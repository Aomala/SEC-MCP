const CH=document.getElementById('chat'),I=document.getElementById('inp'),BT=document.getElementById('btn'),AP=document.getElementById('apanel'),APC=document.getElementById('ap-content');
let _tk=null,_filings=[],_company=null,_avail=[],_curAcc=null,_curData=null;
let _vm='metric_rows',_ff=null,_poll=null,_ci={},_lt=null,_fcPrompt='',_histOpen=true,_isFpi=false;
let _bgSections={};  /* cached filing text sections: {mda:'...', risk_factors:'...', ...} */
let _execSummaries=null; /* cached exec summaries from Claude */
let _chatHistory=[];  /* conversation history for context-aware follow-ups: [{role,content},...] */

/* ═══ Browser-side data cache ═══ */
/* Keyed by "TICKER|accession" — stores {data, summary} so switching periods is instant */
const _dataCache={};
function cacheKey(tk,acc){return(tk||'')+'|'+(acc||'')}
function cacheGet(tk,acc){return _dataCache[cacheKey(tk,acc)]||null}
function cachePut(tk,acc,data,summary){_dataCache[cacheKey(tk,acc)]={data,summary,ts:Date.now()}}
I.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();send()}});
function q(t){I.value=t;send()}

/* ═══ Search bar autocomplete ═══ */
let _searchTimeout=null;
function debSearch(val){
  clearTimeout(_searchTimeout);
  const drop=document.getElementById('sr-drop');
  if(!val||val.length<1){if(drop)drop.style.display='none';return}
  _searchTimeout=setTimeout(async()=>{
    try{
      const r=await fetch('/api/search?q='+encodeURIComponent(val));
      const j=await r.json();
      if(!j.results||!j.results.length){if(drop)drop.style.display='none';return}
      let h='';
      for(const x of j.results){
        h+='<div class="sr-item" onclick="pickAsset(\''+esc(x.ticker)+'\')"><span class="sr-tk">'+esc(x.ticker)+'</span>';
        if(x.exchange)h+='<span class="sr-cik">'+esc(x.exchange)+'</span>';
        else h+='<span class="sr-cik">CIK '+esc(x.cik)+'</span>';
        h+='<span class="sr-nm">'+esc(x.name)+'</span></div>';
      }
      if(drop){drop.innerHTML=h;drop.style.display='block'}
    }catch(e){if(drop)drop.style.display='none'}
  },250);
}
function pickAsset(tk){
  const drop=document.getElementById('sr-drop');if(drop)drop.style.display='none';
  const si=document.getElementById('asrch');if(si)si.value='';
  _tk=tk.toUpperCase();_execSummaries=null;_bgSections={};_chatHistory=[];
  I.value=tk;send();
}
document.addEventListener('click',e=>{
  const drop=document.getElementById('sr-drop');
  if(drop&&!e.target.closest('.srch'))drop.style.display='none';
  const fp=document.getElementById('fc-popup');
  if(fp&&fp.style.display==='block'&&!e.target.closest('.fc-popup')&&!e.target.closest('.fc-btn'))fp.style.display='none';
});

/* ═══ Follow-up detection ═══ */
/* If data is loaded and the message is a question about it (not a new company query), use context-aware chat */
function isFollowUp(msg){
  if(!_curData)return false;
  const low=msg.toLowerCase().trim();
  /* Check for new tickers: 2-5 uppercase letters that aren't common words or finance acronyms.
     Also exclude the currently loaded ticker (asking about it IS a follow-up). */
  const commonWords='THE|FOR|AND|BUT|NOT|ARE|WAS|HAS|HAD|ITS|ALL|CAN|DID|GET|HAS|HER|HIM|HIS|HOW|LET|MAY|NEW|NOW|OLD|OUR|OWN|SAY|SHE|TOO|USE|HER|WAY|WHO|BOY|DAD|MOM';
  const financeAcronyms='CEO|CFO|COO|CTO|ROE|ROA|EPS|IPO|USA|SEC|MDA|DCF|YOY|QOQ|GDP|FCF|ETF|SGA|OCF|TTM|YTD|NAV|AUM|IRR|NPV|PE|PB';
  const excludeRe=new RegExp('\\b('+commonWords+'|'+financeAcronyms+'|'+(_tk||'ZZZZZ')+')\\b');
  const tickerCandidates=msg.match(/\b[A-Z]{2,5}\b/g)||[];
  const realTickers=tickerCandidates.filter(t=>!excludeRe.test(t));
  const hasNewTicker=realTickers.length>0;
  const hasCompanyKeyword=/\b(compare|load|show me|pull up|get|fetch|switch to|look up)\b.*\b[A-Z]{2,5}\b/i.test(msg);
  if(hasCompanyKeyword)return false;
  /* If the message starts with a question word or analysis keyword, it's a follow-up */
  const followUpStarters=['what','why','how','is','are','does','do','can','should','tell','explain',
    'analyze','summarize','break down','walk me through','give me','revenue','margin','profit',
    'cash flow','balance sheet','income','debt','assets','growth','risk','trend','outlook',
    'compare to','year over year','quarter','dividend','eps','roe','valuation','healthy',
    'concern','strength','weakness','insight','opinion','interpret','executive summary'];
  for(const s of followUpStarters){if(low.startsWith(s)||low.includes(s))return !hasNewTicker}
  /* Short questions without tickers are likely follow-ups */
  if(low.length<80&&!hasNewTicker&&(low.includes('?')||low.startsWith('is ')||low.startsWith('are ')))return true;
  return false;
}

/* ═══ Chat send ═══ */
async function send(){
  const m=I.value.trim();if(!m)return;
  const w=document.getElementById('welcome');if(w)w.remove();
  addU(m);I.value='';BT.disabled=true;

  /* Smart routing: if data is loaded and this is a follow-up question, use context-aware AI */
  if(isFollowUp(m)){
    const lid=addLd();
    /* Show thinking indicator */
    const thinkEl=document.createElement('div');thinkEl.className='a-row';
    thinkEl.innerHTML='<div class="a-av think-av">\u2699</div><div class="a-body"><div class="think-card">'
      +'<div class="think-hdr"><span class="think-tool">AI Q&A</span></div>'
      +'<div class="think-steps"><div class="think-step">\u2192 Using loaded '+esc(_tk||'')+' data as context</div>'
      +'<div class="think-step">\u2192 Asking Claude for analysis</div></div></div></div>';
    CH.appendChild(thinkEl);scr();
    try{
      const ctx=Object.assign({},_curData||{});
      if(_bgSections&&Object.keys(_bgSections).length)ctx._filing_sections=_bgSections;
      const body={message:m,ticker:_tk||'',context:ctx,history:_chatHistory.slice(-6)};
      const r=await fetch('/api/chatbot',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify(body)});
      const j=await r.json();rm(lid);
      let answer=j.answer||'No response.';
      if(j.citations&&j.citations.length){
        answer+='\n\n_Source: '+j.citations.map(c=>c.source).join(', ')+'_';
      }
      addA(answer);
      /* Track conversation history for multi-turn context */
      _chatHistory.push({role:'user',content:m});
      _chatHistory.push({role:'assistant',content:answer});
      if(_chatHistory.length>20)_chatHistory=_chatHistory.slice(-12);
    }catch(e){rm(lid);addA('Error: '+e.message,1)}
    BT.disabled=false;I.focus();return;
  }

  /* Standard routing: new company query — fetch fresh data */
  const lid=addLd();showLd();
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})});
    const j=await r.json();rm(lid);
    /* Show thinking/routing first */
    if(j.intent_reasoning&&j.intent_reasoning.length){showThinking(j)}
    if(j.type==='error'){stopLd();addA(j.message,1)}
    else if(j.type==='info'){stopLd();addA(j.message)}
    else if(j.type==='result')handleRes(j);
  }catch(e){rm(lid);stopLd();addA(e.message,1)}
  BT.disabled=false;I.focus();
}

/* ═══ Thinking/tool routing display ═══ */
function showThinking(j){
  const tool=j.intent_tool||'?';
  const reasons=j.intent_reasoning||[];
  const elapsed=j.elapsed_ms?(' \u00b7 '+j.elapsed_ms+'ms'):'';
  const d=document.createElement('div');d.className='a-row';
  let inner='<div class="a-av think-av">\u2699</div><div class="a-body"><div class="think-card">';
  inner+='<div class="think-hdr"><span class="think-tool">'+esc(tool.toUpperCase())+'</span>';
  inner+='<span class="think-time">'+esc(elapsed)+'</span></div>';
  inner+='<div class="think-steps">';
  for(const r of reasons)inner+='<div class="think-step">\u2192 '+esc(r)+'</div>';
  inner+='</div></div></div>';
  d.innerHTML=inner;CH.appendChild(d);scr();
}
function addU(t){const d=document.createElement('div');d.className='u-row';
  d.innerHTML='<div class="u-bub">'+esc(t)+'</div>';CH.appendChild(d);scr()}
function addA(t,err){const d=mkA();const c=d.querySelector('.a-card');
  if(err)c.style.color='var(--red)';c.innerHTML=md(t);CH.appendChild(d);scr()}
function addLd(){const d=mkA();const c=d.querySelector('.a-card');
  c.innerHTML='<div class="ld-wrap"><div class="ld"></div><div class="ld"></div><div class="ld"></div></div>';
  const id='l'+Date.now();d.id=id;CH.appendChild(d);scr();return id}
function rm(id){const e=document.getElementById(id);if(e)e.remove()}
function scr(){CH.scrollTop=CH.scrollHeight}
function esc(s){if(s==null)return'';const d=document.createElement('div');d.textContent=String(s);return d.innerHTML}
function mkA(){const r=document.createElement('div');r.className='a-row';
  r.innerHTML='<div class="a-av">S</div><div class="a-body"><div class="a-card"></div></div>';return r}
function md(s){if(!s)return'';
  let h=esc(s);
  h=h.replace(/### (.+)/g,'<strong style="display:block;margin:10px 0 4px;font-size:13px;color:var(--t1)">$1</strong>');
  h=h.replace(/## (.+)/g,'<strong style="display:block;margin:12px 0 6px;font-size:14px;color:var(--t1)">$1</strong>');
  h=h.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  h=h.replace(/\*(.+?)\*/g,'<em style="color:var(--t3)">$1</em>');
  h=h.replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank">$1</a>');
  h=h.replace(/^- /gm,'<li>').replace(/\n- /g,'</li><li>');
  if(h.includes('<li>'))h='<ul style="margin:6px 0;padding-left:18px">'+h+'</ul>';
  h=h.replace(/\n\n/g,'<br><br>').replace(/\n/g,'<br>');
  return h}

/* ═══ Handle result ═══ */
function handleRes(j){
  const tk=(j.resolved_tickers||[])[0]||'';
  if(j.tool==='financials'||j.tool==='explain'){
    const d=j.data||{};const name=d.company_name||tk||'Unknown';
    const m=d.metrics||{};const fi=d.filing_info||{};
    /* Build a rich summary message for the chat */
    let msg='**'+name+'** loaded\n\n';
    if(fi.form_type)msg+='- Filing: **'+fi.form_type+'** filed '+fi.filing_date+'\n';
    if(m.revenue!=null)msg+='- Revenue: **'+fmtN(m.revenue)+'**\n';
    if(m.net_income!=null)msg+='- Net Income: **'+fmtN(m.net_income)+'**\n';
    if(m.operating_cash_flow!=null)msg+='- Op. Cash Flow: **'+fmtN(m.operating_cash_flow)+'**\n';
    if(m.total_assets!=null)msg+='- Total Assets: **'+fmtN(m.total_assets)+'**\n';
    msg+='\n_Ask me anything about this data — margins, trends, risks, etc._';
    addA(msg);
    _execSummaries=null;_chatHistory=[];
    /* Cache this data for instant switching later */
    if(j.data&&j.data.filing_info){cachePut(tk||_tk,j.data.filing_info.accession_number,j.data,j.narrative||j.summary||'')}
    render(j.data,j.narrative||j.summary||'');
    if(tk){fetchAvail(tk);bgLoadSections(tk);}
  }else if(j.tool==='compare'){
    if(j.error){addA(j.error,1);return}
    const res=j.results||[];
    const valid=res.filter(r=>r&&r.data&&!r.data.error);
    if(valid.length===0){addA('Could not load comparison data for '+(j.resolved_tickers||[]).join(', '),1);return}
    addA('**Comparison** loaded: '+valid.map(r=>r.data.company_name||r.data.ticker_or_cik).join(' vs '));
    renderCompare(j);if((j.resolved_tickers||[])[0])fetchAvail((j.resolved_tickers||[])[0])
  }else if(j.tool==='filing_text'){
    if(j.error)addA(j.error,1);
    else{addA('**'+esc(j.ticker||'')+'** filing section loaded \u2192 analysis panel');renderFiling(j)}
  }else if(j.tool==='historical'){
    addA(j.message||'Historical extraction started.');if(tk){_tk=tk;fetchHist(tk)}
  }else if(j.tool==='entity'){
    if(j.error)addA(j.error,1);
    else{renderEntity(j.profile);addA('**'+esc((j.profile||{}).name||tk)+'** entity profile loaded')}
  }else if(j.tool==='qa'){
    if(j.error)addA(j.error,1);
    else{renderQA(j);if(j.data){render(j.data,'')}}
  }else addA(JSON.stringify(j,null,2));
}

/* ═══ Loading ═══ */
function showLd(){
  if(_lt)clearInterval(_lt);const s=Date.now();
  APC.innerHTML='<div class="ap-ld"><div class="sp"></div><div class="tm" id="ltm">0.0s</div>'
    +'<div class="sub">Fetching from SEC EDGAR\u2026</div></div>';
  _lt=setInterval(()=>{const e=document.getElementById('ltm');if(e)e.textContent=((Date.now()-s)/1000).toFixed(1)+'s'},100);
}
function stopLd(){if(_lt){clearInterval(_lt);_lt=null}}

/* ═══ Fetch available filings for dropdowns ═══ */
async function fetchAvail(tk){
  try{
    const r=await fetch('/api/filings/'+encodeURIComponent(tk));
    const j=await r.json();_avail=j.filings||[];
    /* Detect Foreign Private Issuer: files 20-F/6-K/40-F but NOT 10-K/10-Q */
    const hasDomestic=_avail.some(f=>['10-K','10-Q'].includes(f.form_type));
    const hasForeign=_avail.some(f=>['20-F','6-K','40-F'].includes(f.form_type));
    _isFpi=hasForeign&&!hasDomestic;
    updateFormSelector();populatePeriodDropdown();
  }catch(e){_avail=[]}
}

/* ═══ Update toolbar form selector based on FPI detection ═══ */
function updateFormSelector(){
  const ind=document.getElementById('fpi-indicator');
  if(ind)ind.style.display=_isFpi?'inline-flex':'none';
  const sel=document.getElementById('sel-form');if(!sel)return;
  /* For FPI companies with no data yet, auto-select their primary annual form */
  if(_isFpi&&sel.value===''&&!_curData){
    const has40F=_avail.some(f=>f.form_type==='40-F');
    const has20F=_avail.some(f=>f.form_type==='20-F');
    if(has40F)sel.value='40-F';
    else if(has20F)sel.value='20-F';
  }
}

function populatePeriodDropdown(){
  const sel=document.getElementById('sel-period');if(!sel)return;
  const formSel=document.getElementById('sel-form');
  const formFilter=formSel?formSel.value:'';
  let filtered=_avail;
  if(formFilter){
    const altMap={
      '10-K':['10-K','20-F','40-F'],'10-Q':['10-Q','6-K'],
      '20-F':['20-F','10-K','40-F'],'6-K':['6-K','10-Q'],'40-F':['40-F','10-K','20-F']
    };
    const alts=altMap[formFilter]||[formFilter];
    filtered=_avail.filter(f=>alts.includes(f.form_type));
  }
  let h='<option value="">Select Period</option>';
  for(const f of filtered){
    const sel2=_curAcc===f.accession?' selected':'';
    const label=f.form_type+' \u00b7 '+shortDate(f.filing_date);
    h+='<option value="'+esc(f.accession)+'|'+esc(f.form_type)+'"'+sel2+'>'+label+'</option>';
  }
  sel.innerHTML=h;
}
function onFormSel(){
  populatePeriodDropdown();
  /* Auto-load the most recent period of the selected type — one click, not two */
  const sel=document.getElementById('sel-period');
  if(!sel||!_tk)return;
  /* Skip auto-load if the current data already matches this form type */
  const formSel=document.getElementById('sel-form');
  const newFt=formSel?formSel.value:'';
  const curFt=(_curData&&_curData.filing_info)?_curData.filing_info.form_type:'';
  const altMap={
    '10-K':['10-K','20-F','40-F'],'10-Q':['10-Q','6-K'],
    '20-F':['20-F','10-K','40-F'],'6-K':['6-K','10-Q'],'40-F':['40-F','10-K','20-F']
  };
  const alts=altMap[newFt]||[newFt];
  if(alts.includes(curFt))return; /* already showing this form type */
  /* Select and load the first available period of the new type */
  for(let i=1;i<sel.options.length;i++){
    const parts=sel.options[i].value.split('|');
    if(parts.length===2){sel.selectedIndex=i;loadFiling(parts[0],parts[1]);break}
  }
}
function onPeriodSel(){
  const sel=document.getElementById('sel-period');if(!sel||!sel.value)return;
  const parts=sel.value.split('|');if(parts.length===2)loadFiling(parts[0],parts[1]);
}

async function loadFiling(acc,ft){
  if(!_tk)return;
  _curAcc=acc;_execSummaries=null;
  populatePeriodDropdown();

  /* Check browser cache first — instant switch for previously loaded periods */
  const cached=cacheGet(_tk,acc);
  if(cached){
    addA('Loaded **'+ft+'** filed '+(cached.data?.filing_info?.filing_date||'')+' *(cached)*');
    render(cached.data,cached.summary||'');
    return;
  }

  /* Light loading overlay — preserves toolbar instead of destroying entire panel */
  const overlay=document.createElement('div');
  overlay.id='filing-overlay';
  overlay.innerHTML='<div style="display:flex;align-items:center;justify-content:center;height:100%;background:rgba(17,17,21,0.85);border-radius:12px"><div class="sp"></div><span style="color:var(--t2);margin-left:12px">Loading '+esc(ft)+'...</span></div>';
  overlay.style.cssText='position:absolute;inset:0;z-index:100;pointer-events:all';
  APC.style.position='relative';APC.appendChild(overlay);
  try{
    const r=await fetch('/api/load-filing',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ticker:_tk,accession:acc,form_type:ft})});
    const j=await r.json();
    const ov=document.getElementById('filing-overlay');if(ov)ov.remove();
    /* Cache for instant switching later */
    if(j.data)cachePut(_tk,acc,j.data,j.summary||'');
    addA('Loaded **'+ft+'** filed '+(j.data?.filing_info?.filing_date||''));
    render(j.data,j.summary||'');
  }catch(e){
    const ov=document.getElementById('filing-overlay');if(ov)ov.remove();
    addA('Error loading filing: '+e.message,1);
  }
}

/* ═══ Fact-check feature ═══ */
function factCheck(metric,value,e){
  if(!_curData)return;
  const fi=_curData.filing_info||{};
  const prompt='Verify this SEC filing data for '+(_tk||'?')+': '+metric+' = '+value
    +', from '+(fi.form_type||'10-K')+' filed '+(fi.filing_date||'?')
    +', accession '+(fi.accession_number||'?')+'. Cross-reference the actual EDGAR filing.';
  _fcPrompt=prompt;
  navigator.clipboard.writeText(prompt).then(()=>showToast('Copied to clipboard!'));
  const popup=document.getElementById('fc-popup');
  if(popup){
    const rect=e.target.getBoundingClientRect();
    popup.style.left=Math.min(rect.left,window.innerWidth-180)+'px';
    popup.style.top=(rect.bottom+4)+'px';
    popup.style.display='block';
  }
  if(e)e.stopPropagation();
}
function factCheckChart(chartLabel,e){
  if(!_curData)return;
  const m=_curData.metrics||{},r=_curData.ratios||{},fi=_curData.filing_info||{};
  let data='';
  if(chartLabel==='revenue'){
    data='Revenue='+fmtN(m.revenue)+', Gross Profit='+fmtN(m.gross_profit)+', Op Income='+fmtN(m.operating_income)+', Net Income='+fmtN(m.net_income);
  }else{
    data='Gross Margin='+(r.gross_margin!=null?(r.gross_margin*100).toFixed(1)+'%':'?')
      +', Op Margin='+(r.operating_margin!=null?(r.operating_margin*100).toFixed(1)+'%':'?')
      +', Net Margin='+(r.net_margin!=null?(r.net_margin*100).toFixed(1)+'%':'?');
  }
  const prompt='Verify SEC filing data for '+(_tk||'?')+': '+data
    +'. Source: '+(fi.form_type||'10-K')+' filed '+(fi.filing_date||'?')
    +', accession '+(fi.accession_number||'?')+'. Cross-reference EDGAR.';
  _fcPrompt=prompt;
  navigator.clipboard.writeText(prompt).then(()=>showToast('Copied to clipboard!'));
  const popup=document.getElementById('fc-popup');
  if(popup){
    const rect=e.target.getBoundingClientRect();
    popup.style.left=Math.min(rect.left,window.innerWidth-180)+'px';
    popup.style.top=(rect.bottom+4)+'px';
    popup.style.display='block';
  }
  if(e)e.stopPropagation();
}
function fcGo(target){
  const urls={chatgpt:'https://chat.openai.com/',claude:'https://claude.ai/new',
    perplexity:'https://www.perplexity.ai/',google:'https://www.google.com/search?q='+encodeURIComponent(_fcPrompt)};
  window.open(urls[target]||'#','_blank');
  document.getElementById('fc-popup').style.display='none';
  return false;
}
function showToast(msg){
  const t=document.getElementById('toast');if(!t)return;t.textContent=msg;t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),2000);
}

/* ═══ Main Renderer ═══ */
function render(d,narr){
  stopLd();if(!d){APC.innerHTML='<div class="ap-empty"><div><p>No data</p></div></div>';return}
  Object.values(_ci).forEach(c=>{try{c.destroy()}catch(e){}});_ci={};
  /* Preserve _tk: only update if the incoming value looks like a real ticker */
  const incomingTk=(d.ticker_or_cik||'').toUpperCase();
  if(incomingTk&&!/^\d{7,}$/.test(incomingTk))_tk=incomingTk;
  _execSummaries=null;
  _curData=d;
  if(d.filing_info)_curAcc=d.filing_info.accession_number||null;
  const m=d.metrics||{},r=d.ratios||{},fi=d.filing_info||{},lk=d.sec_links||{};
  const pm=d.prior_metrics||{};
  const toolSel=document.getElementById('sel-tool');
  if(toolSel)toolSel.value='dashboard';

  let h='';

  /* ── 1. Company Hero ── */
  h+=buildHero(d,m,r,fi,lk);

  /* ── 2. KPI Strip ── */
  h+=buildKpiStrip(m,r,pm,d);

  /* ── 3. Main body: charts LEFT + statements RIGHT ── */
  h+='<div class="dash-body">';

  /* LEFT — 3 stacked charts */
  h+='<div class="dash-left">';
  h+='<div class="ch-panel ch-stack"><div class="ch-lbl">Revenue Breakdown <button class="fc-btn" onclick="factCheckChart(\'revenue\',event)" title="Verify">&#x2713;</button></div><canvas id="c-rev"></canvas></div>';
  h+='<div class="ch-panel ch-stack"><div class="ch-lbl">Margins <button class="fc-btn" onclick="factCheckChart(\'margins\',event)" title="Verify">&#x2713;</button></div><canvas id="c-mar"></canvas></div>';
  h+='<div class="ch-panel ch-stack"><div class="ch-lbl">Cash Flows</div><canvas id="c-cf"></canvas></div>';
  h+='</div>';

  /* RIGHT — statement tabs */
  h+='<div class="stmt-section">';
  const ss=[{k:'income_statement',l:'Income Statement'},{k:'balance_sheet',l:'Balance Sheet'},{k:'cash_flow_statement',l:'Cash Flow'}];
  h+='<div class="stmt-tab-bar" id="stmt-tabs">';
  h+='<button class="stmt-tab act" onclick="swStmtTab(this,\'sw-ov\')">&#x229E; Overview</button>';
  h+='<button class="stmt-tab" onclick="swStmtTab(this,\'sw0\')">&#x2191; Income</button>';
  h+='<button class="stmt-tab" onclick="swStmtTab(this,\'sw1\')">&#x25A3; Balance Sheet</button>';
  h+='<button class="stmt-tab" onclick="swStmtTab(this,\'sw2\')">&#x21C4; Cash Flow</button>';
  h+='<button class="stmt-tab" onclick="swStmtTab(this,\'mt-exec2\')">&#x2726; AI</button>';
  const val=d.validation||[];
  if(val.length){
    const errCount=val.filter(w=>w.severity==='error').length;
    h+='<span class="val-pill '+(errCount?'err':'wrn')+'" title="'+esc(val.map(w=>w.message).join('; '))+'">';
    h+=(errCount?'\u26D4':'⚠')+' '+val.length+'</span>';
  }
  h+='</div>';
  h+='<div class="stmt-content">';
  h+='<div class="stmt-pane" id="sw-ov">'+bOverview(d)+'</div>';
  ss.forEach((s,i)=>h+='<div class="stmt-pane" id="sw'+i+'" style="display:none">'+bStmt(d[s.k]||[],s.l)+'</div>');
  h+='<div class="stmt-pane" id="mt-exec2" style="display:none">';
  h+='<div class="exec-inline" id="exec-inline" style="display:none">';
  h+='<div class="exec-inline-hdr"><h3>AI Analysis</h3><span class="exec-badge" id="exec-badge">Generating...</span></div>';
  h+='<div class="exec-inline-body" id="exec-inline-body">';
  h+='<div class="exec-inline-loading"><div class="ld-wrap"><div class="ld"></div><div class="ld"></div><div class="ld"></div></div>';
  h+='<span style="color:var(--t3);font-size:12px;margin-left:8px">Claude is analyzing\u2026</span></div>';
  h+='</div></div></div>';
  h+='</div>'; /* end stmt-content */
  h+='</div>'; /* end stmt-section */
  h+='</div>'; /* end dash-body */

  /* ── 5. Historical ── */
  h+='<div class="hist-sec" id="hist-sec">';
  h+='<div class="hist-toggle'+((_histOpen)?' open':'')+'" id="hist-toggle" onclick="toggleHist()">';
  h+='<span class="arr">&#x25B6;</span><span class="lab">Historical Data</span>';
  h+='<span class="fcnt" id="fcnt" style="margin-left:auto"></span></div>';
  h+='<div class="hist-body'+((_histOpen)?' open':'')+'" id="hist-body">';
  h+='<div class="fbar">';
  h+='<button class="fbtn act" onclick="setFf(null,this)">All</button>';
  h+='<button class="fbtn" onclick="setFf(\'10-K\',this)">Annual</button>';
  h+='<button class="fbtn" onclick="setFf(\'10-Q\',this)">Quarterly/Interim</button>';
  h+='<button class="fbtn" onclick="setFf(\'8-K\',this)">8-K Events</button>';
  h+='<div class="fsep"></div>';
  h+='<button class="fbtn act" onclick="setVm(\'metric_rows\',this)">Metrics\u00d7Time</button>';
  h+='<button class="fbtn" onclick="setVm(\'period_rows\',this)">Time\u00d7Metrics</button>';
  h+='<div class="fright">';
  h+='<button class="fref" onclick="trigRef()">\u21BB Refresh</button>';
  h+='<button class="fref" onclick="popTbl()">\u2922 Expand</button></div></div>';
  h+='<div class="prog" id="prog"><div class="prog-fill" id="pf" style="width:0"></div></div>';
  h+='<div class="tbl-area" id="tbl"></div><div id="extra"></div>';
  h+='</div></div>';

  APC.innerHTML=h;_ff=null;_vm='metric_rows';
  setTimeout(()=>{rCharts(m,r);populatePeriodDropdown()},50);
  fetchHist(_tk);
  if(!_execSummaries&&_curData){setTimeout(()=>genExecSummaries(),600)}
}

function buildHero(d,m,r,fi,lk){
  const name=d.company_name||_tk||'Unknown';
  const industry=d.industry_class?d.industry_class.replace(/_/g,' ').toUpperCase():'';
  const isFpiForm=['20-F','6-K','40-F'].includes(fi.form_type||'');
  const isInterim=['10-Q','6-K'].includes(fi.form_type||'');
  /* Data coverage */
  const req=['revenue','net_income','total_assets','current_assets','operating_cash_flow',
    'capital_expenditures','free_cash_flow','stockholders_equity','total_liabilities','cash_and_equivalents'];
  const filled=req.filter(k=>m[k]!=null).length;
  const pct=Math.round(filled/req.length*100);
  const covClass=pct>=80?'cov-good':pct>=50?'cov-med':'cov-low';

  let h='<div class="hero">';

  /* ── Top row: ticker + name | form tag + filing date ── */
  h+='<div class="hero-top">';
  h+='<div class="hero-id">';
  h+='<div class="hero-ticker-row">';
  h+='<span class="hero-ticker">'+esc(_tk||d.ticker_or_cik||'')+'</span>';
  if(isFpiForm)h+='<span class="hero-badge fpi-badge">FPI</span>';
  if(isInterim)h+='<span class="hero-badge interim-badge">INTERIM</span>';
  h+='</div>';
  h+='<div class="hero-name">'+esc(name)+'</div>';
  h+='</div>';
  h+='<div class="hero-filing-block">';
  if(fi.form_type)h+='<div class="hero-form-tag form-'+esc(fi.form_type.replace('-',''))+'">'+esc(fi.form_type)+'</div>';
  h+='<div class="hero-filing-meta">';
  if(fi.filing_date)h+='<span>Filed <strong>'+esc(fi.filing_date)+'</strong></span>';
  if(fi.period_of_report&&fi.period_of_report!==fi.filing_date)
    h+='<span class="meta-sep">&middot;</span><span>Period '+esc(fi.period_of_report.slice(0,7))+'</span>';
  h+='</div></div>';
  h+='</div>'; /* end hero-top */

  /* ── Bottom row: tags | coverage | links ── */
  h+='<div class="hero-bottom">';
  h+='<div class="hero-tags">';
  if(industry)h+='<span class="htag ind">'+industry+'</span>';
  if(isFpiForm)h+='<span class="htag fpi">FOREIGN FILER</span>';
  else if(fi.form_type)h+='<span class="htag dom">US FILER</span>';
  if(d.exchange)h+='<span class="htag exch">'+esc(d.exchange)+'</span>';
  h+='</div>';
  h+='<div class="hero-right-row">';
  h+='<div class="hero-cov-row">';
  h+='<span class="hero-cov-label">Coverage</span>';
  h+='<div class="hero-cov-bar '+covClass+'"><div style="width:'+pct+'%"></div></div>';
  h+='<span class="hero-cov-pct">'+pct+'%</span>';
  h+='</div>';
  h+='<div class="hero-links">';
  if(lk.filing_index)h+='<a href="'+lk.filing_index+'" target="_blank" class="hero-link edgar-link">EDGAR \u2197</a>';
  if(d.ir_link)h+='<a href="'+esc(d.ir_link)+'" target="_blank" class="hero-link ir-link">IR \u2197</a>';
  h+='</div>';
  h+='</div>';
  h+='</div>'; /* end hero-bottom */

  /* ── Info banner for sparse-data form types ── */
  if(fi.form_type==='6-K'){
    h+='<div class="hero-banner fpi-banner">';
    h+='\u2139\ufe0f &nbsp;<strong>6-K filings</strong> typically contain limited XBRL data. ';
    h+='Financial statements may be partial. Load the <strong>20-F annual filing</strong> for complete data.';
    h+='</div>';
  }else if(fi.form_type==='40-F'){
    h+='<div class="hero-banner fpi-banner">';
    h+='\u2139\ufe0f &nbsp;<strong>40-F</strong> is the annual filing form for Canadian issuers listed on US exchanges.';
    h+='</div>';
  }else if(pct<40&&fi.form_type){
    h+='<div class="hero-banner warn-banner">';
    h+='\u26a0\ufe0f &nbsp;Limited XBRL data available for this filing ('+pct+'% coverage). Some metrics may be missing.';
    h+='</div>';
  }

  h+='</div>'; /* end hero */
  return h;
}

function buildKpiStrip(m,r,pm,d){
  const isQ=d.period_type==='quarterly';
  const periodLabel=isQ?'QoQ':'YoY';
  function deltaStr(cur,prev){
    if(cur==null||prev==null||prev===0)return null;
    const pct=(cur-prev)/Math.abs(prev)*100;
    const up=pct>=0;
    return{pct,up,str:(up?'\u25b2':'\u25bc')+' '+Math.abs(pct).toFixed(1)+'%'};
  }
  /* cat = 'rev' | 'inc' | 'gp' | 'cf' | 'bs' | 'eps' */
  function kpi(label,val,prev,opts){
    opts=opts||{};
    if(val==null)return'';
    const d2=deltaStr(val,prev);
    const isNeg=val<0;
    const cat=opts.cat||'';
    let h='<div class="kpi-card kpi-'+cat+'">';
    h+='<div class="kpi-label">'+label+'</div>';
    h+='<div class="kpi-value'+(isNeg?' neg':'')+'">'+fmtN(val)+'</div>';
    if(d2){
      h+='<div class="kpi-delta '+(d2.up?'up':'dn')+'">';
      h+=esc(d2.str)+'<span class="kpi-period"> '+periodLabel+'</span></div>';
    }else{
      h+='<div class="kpi-delta neutral">\u2014</div>';
    }
    if(opts.sub)h+='<div class="kpi-sub">'+opts.sub+'</div>';
    h+='</div>';
    return h;
  }
  let h='<div class="kpi-strip kpi-8">';
  h+=kpi('Revenue',m.revenue,pm.revenue,{cat:'rev',sub:r.gross_margin!=null?'GM '+(r.gross_margin*100).toFixed(1)+'%':null});
  h+=kpi('Net Income',m.net_income,pm.net_income,{cat:'inc',sub:r.net_margin!=null?'Margin '+(r.net_margin*100).toFixed(1)+'%':null});
  h+=kpi('Gross Profit',m.gross_profit,pm.gross_profit,{cat:'gp',sub:r.operating_margin!=null?'OpMgn '+(r.operating_margin*100).toFixed(1)+'%':null});
  h+=kpi('Free Cash Flow',m.free_cash_flow,pm.free_cash_flow,{cat:'cf'});
  h+=kpi('Total Assets',m.total_assets,pm.total_assets,{cat:'bs',sub:r.return_on_assets!=null?'ROA '+(r.return_on_assets*100).toFixed(1)+'%':null});
  h+=kpi('Current Assets',m.current_assets,pm.current_assets,{cat:'bs'});
  h+=kpi('Total Liabilities',m.total_liabilities,pm.total_liabilities,{cat:'liab'});
  if(m.eps_diluted!=null)h+=kpi('EPS (Diluted)',m.eps_diluted,pm.eps_diluted,{cat:'eps'});
  else if(m.operating_cash_flow!=null)h+=kpi('Op. Cash Flow',m.operating_cash_flow,pm.operating_cash_flow,{cat:'cf'});
  else if(m.ebitda!=null)h+=kpi('EBITDA',m.ebitda,pm.ebitda,{cat:'inc',sub:r.ebitda_margin!=null?'Margin '+(r.ebitda_margin*100).toFixed(1)+'%':null});
  h+='</div>';
  return h;
}

function swMainTab2(el,id){
  /* Legacy — redirect to new tab switcher */
  swStmtTab(el,id);
}

function swStmtTab(el,id){
  /* Update tab active state */
  const bar=document.getElementById('stmt-tabs');
  if(bar)bar.querySelectorAll('.stmt-tab').forEach(t=>t.classList.remove('act'));
  if(el)el.classList.add('act');
  /* Show correct pane */
  const content=document.querySelector('.stmt-content');
  if(content)content.querySelectorAll('.stmt-pane').forEach(p=>p.style.display='none');
  const pane=document.getElementById(id);if(pane)pane.style.display='';
  /* Auto-generate summaries when AI tab opened */
  if(id==='mt-exec2'){
    if(!_execSummaries&&_curData)genExecSummaries();
    /* Show the inline body panel */
    const inl=document.getElementById('exec-inline');
    if(inl)inl.style.display='';
  }
}

function toggleHist(){
  const toggle=document.getElementById('hist-toggle');
  const body=document.getElementById('hist-body');
  if(!toggle||!body)return;
  _histOpen=!_histOpen;
  toggle.classList.toggle('open',_histOpen);
  body.classList.toggle('open',_histOpen);
}

/* ═══ Charts (3-panel) ═══ */
function noDataMsg(canvasId,msg){
  const c=document.getElementById(canvasId);if(!c)return;
  const p=c.parentElement;c.style.display='none';
  const d=document.createElement('div');d.className='ch-no-data';
  d.innerHTML='<span class="ch-nd-ico">\u26a0</span><span>'+esc(msg||'No data')+'</span>';
  p.appendChild(d);
}
function rCharts(m,r){
  const fi=(_curData&&_curData.filing_info)||{};
  const is6K=fi.form_type==='6-K';
  const scaleSuffix=v=>{const a=Math.abs(v);return a>=1e12?'T':a>=1e9?'B':'M'};
  const scaleDiv=v=>{const a=Math.abs(v);return a>=1e12?1e12:a>=1e9?1e9:1e6};
  /* Determine best scale for revenue data */
  const revMax=Math.max(...[m.revenue,m.gross_profit,m.operating_income,m.net_income].filter(v=>v!=null).map(Math.abs));
  const div=scaleDiv(revMax||1e9);
  const suf=scaleSuffix(revMax||1e9);
  const tkColor='rgba(255,255,255,.45)';
  const gridColor='rgba(255,255,255,.04)';

  /* Chart 1: Revenue waterfall */
  const rc=document.getElementById('c-rev');
  if(rc){
    const pairs=[
      ['Revenue',m.revenue,'rgba(74,122,255,.75)'],
      ['Gross Profit',m.gross_profit,'rgba(52,211,153,.75)'],
      ['Op. Income',m.operating_income,'rgba(167,139,250,.75)'],
      ['Net Income',m.net_income,'rgba(99,179,237,.75)']
    ];
    const valid=pairs.filter(p=>p[1]!=null);
    if(valid.length>=2){
      _ci.rev=new Chart(rc,{type:'bar',
        data:{labels:valid.map(p=>p[0]),datasets:[{data:valid.map(p=>p[1]/div),
          backgroundColor:valid.map((p,i)=>p[1]<0?'rgba(255,82,102,.7)':p[2]),
          borderRadius:6,borderSkipped:false,borderWidth:0}]},
        options:{responsive:true,maintainAspectRatio:false,
          plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>'$'+(Math.abs(ctx.raw)).toFixed(1)+suf+(ctx.raw<0?' (loss)':'')}}},
          scales:{y:{ticks:{callback:v=>'$'+v.toFixed(0)+suf,color:tkColor,font:{size:9}},grid:{color:gridColor},border:{display:false}},
            x:{ticks:{color:tkColor,font:{size:9}},grid:{display:false},border:{display:false}}}}});
    }else{noDataMsg('c-rev',is6K?'Revenue data not in this 6-K':'No income data')}
  }

  /* Chart 2: Margins (horizontal bar) */
  const mc=document.getElementById('c-mar');
  if(mc){
    const mp=[];
    if(r.gross_margin!=null)mp.push(['Gross',(r.gross_margin*100).toFixed(1),'rgba(52,211,153,.75)']);
    if(r.operating_margin!=null)mp.push(['Operating',(r.operating_margin*100).toFixed(1),'rgba(74,122,255,.75)']);
    if(r.net_margin!=null)mp.push(['Net',(r.net_margin*100).toFixed(1),'rgba(99,179,237,.75)']);
    if(r.ebitda_margin!=null)mp.push(['EBITDA',(r.ebitda_margin*100).toFixed(1),'rgba(167,139,250,.75)']);
    if(mp.length){
      _ci.mar=new Chart(mc,{type:'bar',
        data:{labels:mp.map(p=>p[0]),datasets:[{data:mp.map(p=>parseFloat(p[1])),
          backgroundColor:mp.map(p=>parseFloat(p[1])<0?'rgba(255,82,102,.7)':p[2]),
          borderRadius:6,borderSkipped:false,borderWidth:0}]},
        options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
          plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>ctx.raw.toFixed(1)+'%'}}},
          scales:{x:{ticks:{callback:v=>v+'%',color:tkColor,font:{size:9}},grid:{color:gridColor},border:{display:false}},
            y:{ticks:{color:tkColor,font:{size:9}},grid:{display:false},border:{display:false}}}}});
    }else{noDataMsg('c-mar',is6K?'Margin data not in this 6-K':'No margin data')}
  }

  /* Chart 3: Cash flows — show OCF / InvCF / FinCF for broader view */
  const cc=document.getElementById('c-cf');
  if(cc){
    const cf=[];
    const cfDiv=scaleDiv(Math.max(
      Math.abs(m.operating_cash_flow||0),Math.abs(m.investing_cash_flow||0),
      Math.abs(m.financing_cash_flow||0),Math.abs(m.free_cash_flow||0))||1e9);
    const cfSuf=scaleSuffix(cfDiv);
    if(m.operating_cash_flow!=null)cf.push(['Op. CF',m.operating_cash_flow/cfDiv,'rgba(52,211,153,.75)']);
    if(m.investing_cash_flow!=null)cf.push(['Investing',m.investing_cash_flow/cfDiv,'rgba(255,179,71,.65)']);
    if(m.financing_cash_flow!=null)cf.push(['Financing',m.financing_cash_flow/cfDiv,'rgba(255,82,102,.65)']);
    if(m.free_cash_flow!=null)cf.push(['Free CF',m.free_cash_flow/cfDiv,'rgba(74,122,255,.75)']);
    if(cf.length){
      _ci.cf=new Chart(cc,{type:'bar',
        data:{labels:cf.map(p=>p[0]),datasets:[{data:cf.map(p=>p[1]),
          backgroundColor:cf.map(p=>p[2]),borderRadius:5,borderSkipped:false,borderWidth:0}]},
        options:{responsive:true,maintainAspectRatio:false,
          plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>'$'+(Math.abs(ctx.raw)).toFixed(1)+cfSuf+(ctx.raw<0?' (outflow)':'')}}},
          scales:{y:{ticks:{callback:v=>'$'+v.toFixed(0)+cfSuf,color:tkColor,font:{size:9}},grid:{color:gridColor},border:{display:false}},
            x:{ticks:{color:tkColor,font:{size:9}},grid:{display:false},border:{display:false}}}}});
    }else{noDataMsg('c-cf',is6K?'Cash flow data not in this 6-K':'No cash flow data')}
  }
}

/* ═══ Filing text view (MD&A, Risk Factors, etc.) ═══ */
function renderFiling(j){
  stopLd();
  const sec=j.section||'full filing';
  const secLabel={'mda':'MD&A (Management Discussion & Analysis)','risk_factors':'Risk Factors',
    'business':'Business Overview','financial_statements':'Financial Statements',
    'legal':'Legal Proceedings','controls':'Controls & Procedures',
    'executive_compensation':'Executive Compensation','full filing':'Full Filing'}[sec]||sec;
  const txt=j.text||'';
  const hasContent=txt.length>200;
  let h='<div class="filing-view">';

  /* Header with ticker + section label + filing info */
  h+='<div class="filing-hdr">';
  h+='<span class="dh-tk">'+esc(j.ticker||_tk||'')+'</span>';
  h+='<span class="filing-sec">'+esc(secLabel)+'</span>';
  h+='<span class="dh-tag">'+esc(j.form_type||'10-K')+' \u00b7 Filed '+esc(j.filing_date||'')+'</span>';
  if(j.text_length)h+='<span class="dh-tag">'+Math.round(j.text_length/1000)+'K chars</span>';
  h+='</div>';

  /* Summary card — shows key metrics from currently loaded data for context */
  if(_curData&&_curData.metrics){
    const m=_curData.metrics,r=_curData.ratios||{},fi=_curData.filing_info||{};
    h+='<div class="filing-summary">';
    h+='<div class="filing-summary-hdr">Financial Context <span class="dh-tag">'+esc(fi.form_type||'')+' '+esc(fi.filing_date||'')+'</span></div>';
    h+='<div class="filing-summary-grid">';
    const kms=[{k:'revenue',l:'Revenue'},{k:'net_income',l:'Net Income'},{k:'operating_income',l:'Op. Income'},
      {k:'total_assets',l:'Assets'},{k:'free_cash_flow',l:'FCF'},{k:'operating_cash_flow',l:'Op. Cash Flow'}];
    for(const km of kms){
      const v=m[km.k];if(v==null)continue;
      h+='<div class="filing-summary-item"><span class="filing-summary-label">'+km.l+'</span>';
      h+='<span class="filing-summary-val'+(v<0?' neg':'')+'">'+fmtN(v)+'</span></div>';
    }
    /* Add key ratios */
    if(r.net_margin!=null)h+='<div class="filing-summary-item"><span class="filing-summary-label">Net Margin</span><span class="filing-summary-val">'+(r.net_margin*100).toFixed(1)+'%</span></div>';
    if(r.current_ratio!=null)h+='<div class="filing-summary-item"><span class="filing-summary-label">Current Ratio</span><span class="filing-summary-val">'+r.current_ratio.toFixed(2)+'x</span></div>';
    h+='</div></div>';
  }

  if(!hasContent){
    h+='<div class="filing-empty"><div class="filing-empty-ico">\u26A0</div>';
    h+='<h3>Section Not Available</h3>';
    h+='<p>Could not extract <strong>'+esc(secLabel)+'</strong> from this filing.</p>';
    h+='<p>This may be because:</p>';
    h+='<ul><li>The section markers were not found in the filing HTML</li>';
    h+='<li>The filing uses non-standard formatting</li>';
    h+='<li>The section does not exist in this filing type</li></ul>';
    h+='<p style="margin-top:12px">Try searching on <a href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company='+esc(j.ticker||_tk||'')+'&type=10-K&dateb=&owner=include&count=5&search_text=&action=getcompany" target="_blank" style="color:var(--acc)">EDGAR</a> directly.</p>';
    h+='</div>';
  }else{
    h+='<div class="filing-body">'+formatFilingText(txt)+'</div>';
  }
  h+='</div>';
  APC.innerHTML=h;
}

function formatFilingText(txt){
  let lines=esc(txt).split('\\n');
  if(lines.length<3)lines=esc(txt).split('\n');
  let out='';
  for(const line of lines){
    const trimmed=line.trim();
    if(!trimmed){out+='<br>';continue}
    if(trimmed.match(/^(Item\s+\d|ITEM\s+\d|Part\s+[IVX])/i)){
      out+='<h3 class="filing-heading">'+trimmed+'</h3>';
    }else if(trimmed.length<80&&trimmed===trimmed.toUpperCase()&&trimmed.length>3){
      out+='<h4 class="filing-subhead">'+trimmed+'</h4>';
    }else{
      out+='<p class="filing-para">'+trimmed+'</p>';
    }
  }
  return out;
}

/* ═══ Entity Profile view ═══ */
function renderEntity(p){
  if(!p)return;
  stopLd();
  /* Don't override _tk — the ticker is already locked from the initial load.
     Only update if _tk is null (first load via entity). */
  if(!_tk)_tk=(p.ticker||'').toUpperCase();
  let h='<div class="entity-view">';
  h+='<div class="entity-hdr">';
  h+='<div class="entity-avatar">'+esc((p.ticker||'?').charAt(0))+'</div>';
  h+='<div class="entity-title">';
  h+='<h2>'+esc(p.name||'')+'</h2>';
  h+='<div class="entity-meta">';
  h+='<span class="dh-tk">'+esc(p.ticker||'')+'</span>';
  if(p.exchange)h+='<span class="dh-tag">'+esc(p.exchange)+'</span>';
  if(p.industry)h+='<span class="dh-tag">'+esc(p.industry)+'</span>';
  h+='</div></div></div>';
  h+='<div class="entity-grid">';
  const fields=[
    {l:'CEO',v:p.ceo,sub:p.ceo_source},
    {l:'CIK',v:p.cik},
    {l:'SIC Code',v:p.sic_code},
    {l:'Website',v:p.website,link:true},
    {l:'Address',v:p.address},
    {l:'Phone',v:p.phone},
    {l:'Fiscal Year End',v:p.fiscal_year_end},
    {l:'State of Incorp.',v:p.state_of_incorporation},
    {l:'EIN',v:p.ein},
  ];
  for(const f of fields){
    if(!f.v)continue;
    h+='<div class="entity-field"><span class="entity-label">'+esc(f.l)+'</span>';
    if(f.link&&f.v){
      const url=f.v.startsWith('http')?f.v:'https://'+f.v;
      h+='<a href="'+esc(url)+'" target="_blank" class="entity-val link">'+esc(f.v)+'</a>';
    }else{
      h+='<span class="entity-val">'+esc(f.v||'\u2014')+'</span>';
    }
    if(f.sub)h+='<span class="entity-sub">'+esc(f.sub)+'</span>';
    h+='</div>';
  }
  h+='</div>';
  /* Financial summary card if data is loaded */
  if(_curData&&_curData.metrics){
    const m=_curData.metrics,r=_curData.ratios||{},fi=_curData.filing_info||{};
    h+='<div class="filing-summary" style="margin-top:16px">';
    h+='<div class="filing-summary-hdr">Loaded Financials <span class="dh-tag">'+esc(fi.form_type||'')+' '+esc(fi.filing_date||'')+'</span></div>';
    h+='<div class="filing-summary-grid">';
    const kms=[{k:'revenue',l:'Revenue'},{k:'net_income',l:'Net Income'},{k:'operating_income',l:'Op. Income'},
      {k:'total_assets',l:'Assets'},{k:'free_cash_flow',l:'FCF'},{k:'operating_cash_flow',l:'Op. Cash Flow'}];
    for(const km of kms){
      const v=m[km.k];if(v==null)continue;
      h+='<div class="filing-summary-item"><span class="filing-summary-label">'+km.l+'</span>';
      h+='<span class="filing-summary-val'+(v<0?' neg':'')+'">'+fmtN(v)+'</span></div>';
    }
    if(r.net_margin!=null)h+='<div class="filing-summary-item"><span class="filing-summary-label">Net Margin</span><span class="filing-summary-val">'+(r.net_margin*100).toFixed(1)+'%</span></div>';
    if(r.current_ratio!=null)h+='<div class="filing-summary-item"><span class="filing-summary-label">Current Ratio</span><span class="filing-summary-val">'+r.current_ratio.toFixed(2)+'x</span></div>';
    h+='</div></div>';
  }

  if(p.latest_filing){
    const lf=p.latest_filing;
    h+='<div class="entity-filing"><h4>Latest Filing</h4>';
    h+='<span class="dh-tag">'+esc(lf.form_type||'')+' \u00b7 Filed '+esc(lf.filing_date||'')+'</span>';
    if(!_curData){
      h+='<button class="entity-load" onclick="I.value=\''+esc(p.ticker)+'\';send()">Load Financials \u2192</button>';
    }else{
      h+='<button class="entity-load" onclick="document.getElementById(\'sel-tool\').value=\'dashboard\';onToolSel()">View Dashboard \u2192</button>';
    }
    h+='</div>';
  }
  if(!p.ceo){
    h+='<div class="entity-note">\u2139 CEO data is not directly available from SEC filings. ';
    h+='Check the company\'s latest <strong>DEF 14A (Proxy Statement)</strong> or investor relations page.</div>';
  }
  h+='</div>';
  APC.innerHTML=h;
  if(!_avail||!_avail.length)fetchAvail(p.ticker);
}

/* ═══ Compare view ═══ */
let _compareRes=[];
function renderCompare(j){
  stopLd();
  const res=(j.results||[]).filter(r=>r&&r.data&&!r.data.error);
  if(!res.length)return;
  _compareRes=res;
  const tickers=(j.resolved_tickers||[]);
  _tk=tickers[0]||(res[0].data.ticker_or_cik||'').toUpperCase();
  _curData=res[0].data;

  const METRICS=['revenue','net_income','gross_profit','operating_income','ebitda',
    'total_assets','total_liabilities','stockholders_equity','operating_cash_flow',
    'free_cash_flow','capital_expenditures','cash_and_equivalents','long_term_debt'];
  const LABELS={revenue:'Revenue',net_income:'Net Income',gross_profit:'Gross Profit',
    operating_income:'Op. Income',ebitda:'EBITDA',total_assets:'Total Assets',
    total_liabilities:'Total Liab.',stockholders_equity:'Equity',
    operating_cash_flow:'Op. Cash Flow',free_cash_flow:'Free Cash Flow',
    capital_expenditures:'CapEx',cash_and_equivalents:'Cash',
    long_term_debt:'LT Debt'};

  let h='<div class="compare-view">';
  h+='<div class="compare-hdr"><h2>Comparison</h2>';
  h+='<div class="compare-pills">';
  for(let i=0;i<res.length;i++){
    const tk=(res[i].data.ticker_or_cik||'').toUpperCase();
    h+='<button class="compare-pill" onclick="loadCompareIdx('+i+')" title="Load full view">'+esc(tk)+'</button>';
  }
  h+='</div></div>';

  h+='<table class="compare-tbl"><thead><tr><th>Metric</th>';
  for(const r of res)h+='<th>'+esc((r.data.company_name||r.data.ticker_or_cik||'?').substring(0,20))+'</th>';
  h+='</tr></thead><tbody>';
  for(const mk of METRICS){
    const hasAny=res.some(r=>(r.data.metrics||{})[mk]!=null);
    if(!hasAny)continue;
    h+='<tr><td class="lbl">'+esc(LABELS[mk]||mk)+'</td>';
    for(const r of res){
      const v=(r.data.metrics||{})[mk];
      h+='<td>'+(v!=null?fmtN(v):'\u2014')+'</td>';
    }
    h+='</tr>';
  }
  h+='</tbody></table>';

  if(j.comparison_narrative){
    h+='<div class="compare-narr"><h3>AI Comparison</h3><div class="narr-box"><p>'+md(j.comparison_narrative)+'</p></div></div>';
  }
  h+='</div>';
  APC.innerHTML=h;
  fetchAvail(tickers[0]);
}
function loadCompareIdx(i){
  const r=_compareRes[i];
  if(!r)return;
  render(r.data,r.summary||'');
}

/* ═══ Q&A answer view ═══ */
function renderQA(j){
  const answer=j.answer||'No answer available.';
  const cits=j.citations||[];
  let citHtml='';
  if(cits.length){
    citHtml='<div class="qa-cits"><strong>Sources:</strong> ';
    for(const c of cits)citHtml+='<span class="qa-cit">'+esc(c.source||'')+'</span> ';
    citHtml+='</div>';
  }
  addA(md(answer)+citHtml);
}

/* ═══ Historical ═══ */
async function fetchHist(tk){
  try{
    const r=await fetch('/api/historical/'+encodeURIComponent(tk));
    const j=await r.json();_company=j.company;_filings=j.filings||[];
    rTbl();rExtra();
    if(j.job&&j.job.status==='processing'){uProg(j.job.progress,j.job.total);sPoll(tk)}else hProg();
    const fc=document.getElementById('fcnt');if(fc)fc.textContent=_filings.length+' filings';
  }catch(e){const t=document.getElementById('tbl');
    if(t)t.innerHTML='<div style="padding:18px;color:var(--t4);font-size:12px">Historical data unavailable</div>'}
}
function sPoll(tk){if(_poll)clearInterval(_poll);
  _poll=setInterval(async()=>{try{const r=await fetch('/api/historical/'+encodeURIComponent(tk)+'/status');
    const j=await r.json();if(j.status==='processing')uProg(j.progress,j.total);
    else{clearInterval(_poll);_poll=null;hProg();fetchHist(tk)}}catch(e){clearInterval(_poll);_poll=null}},3000)}
function uProg(p,t){const e=document.getElementById('pf');if(e)e.style.width=(t>0?p/t*100:0)+'%'}
function hProg(){const e=document.getElementById('pf');if(e)e.style.width='0'}
async function trigRef(){if(!_tk)return;await fetch('/api/historical/'+_tk+'/fetch',{method:'POST'});sPoll(_tk);addA('Refreshing **'+_tk+'** historical data...')}

function setFf(v,el){_ff=v;el.parentElement.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('act'));el.classList.add('act');rTbl();rExtra()}
function setVm(v,el){_vm=v;el.parentElement.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('act'));el.classList.add('act');rTbl()}

const KM=['revenue','net_income','gross_profit','operating_income','ebitda','total_assets','total_liabilities','stockholders_equity',
  'operating_cash_flow','free_cash_flow','capital_expenditures','eps_basic','eps_diluted','cost_of_revenue','operating_expenses',
  'sga_expense','rd_expense','interest_expense','income_tax_expense','current_assets','current_liabilities','long_term_debt',
  'short_term_debt','depreciation_amortization','shares_outstanding'];
const ML={revenue:'Revenue',net_income:'Net Income',gross_profit:'Gross Profit',operating_income:'Op. Income',ebitda:'EBITDA',
  total_assets:'Total Assets',total_liabilities:'Total Liab.',stockholders_equity:'Equity',operating_cash_flow:'Op. CF',
  free_cash_flow:'Free CF',capital_expenditures:'CapEx',eps_basic:'EPS Basic',eps_diluted:'EPS Diluted',
  cost_of_revenue:'Cost of Rev.',operating_expenses:'OpEx',sga_expense:'SG&A',rd_expense:'R&D',interest_expense:'Interest',
  income_tax_expense:'Income Tax',current_assets:'Cur. Assets',current_liabilities:'Cur. Liab.',long_term_debt:'LT Debt',
  short_term_debt:'ST Debt',depreciation_amortization:'D&A',shares_outstanding:'Shares Out.'};

function gFilt(){let f=_filings;
  if(_ff){
    /* Include all FPI equivalents for each category */
    const altMap={'10-K':['10-K','20-F','40-F'],'10-Q':['10-Q','6-K'],'8-K':['8-K']};
    const alts=altMap[_ff]||[_ff];
    f=f.filter(x=>alts.includes(x.form_type));
  }
  return f.sort((a,b)=>(b.filing_date||'').localeCompare(a.filing_date||''))}
function rTbl(){
  const a=document.getElementById('tbl');if(!a)return;
  const fl=gFilt().filter(f=>f.form_type!=='8-K');
  if(!fl.length){a.innerHTML='<div style="padding:18px;text-align:center;color:var(--t4);font-size:12px">'+
    (_ff==='8-K'?'8-K events shown in timeline below':'No historical filings cached. Click Refresh.')+'</div>';return}
  a.innerHTML=_vm==='metric_rows'?bMR(fl):bPR(fl);
  const fc=document.getElementById('fcnt');if(fc)fc.textContent=_filings.length+' filings';
}
function bMR(fl){
  let h='<table class="dt"><thead><tr><th>Metric</th>';
  for(const f of fl)h+='<th>'+esc(shortDate(f.filing_date))+(['10-Q','6-K'].includes(f.form_type)?' Q':'')+'</th>';
  h+='</tr></thead><tbody>';
  for(const mk of KM){
    if(!fl.some(f=>{const m=(f.metrics||{})[mk];return m&&m.value!=null}))continue;
    h+='<tr><td>'+esc(ML[mk]||mk)+'</td>';
    for(const f of fl){const m=(f.metrics||{})[mk];
      if(!m||m.value==null){h+='<td style="color:var(--t4)">\u2014</td>';continue}
      const v=m.value,neg=v<0,url=m.source_url||(f.source_urls||{}).filing_index||'';
      h+='<td'+(neg?' class="neg"':'')+'>'+fmtN(v);
      if(url)h+='<a class="src" href="'+url+'" target="_blank" title="EDGAR source">\u2197</a>';
      h+='</td>'}h+='</tr>'}
  return h+'</tbody></table>';
}
function bPR(fl){
  const am=KM.filter(mk=>fl.some(f=>{const m=(f.metrics||{})[mk];return m&&m.value!=null}));
  let h='<table class="dt"><thead><tr><th>Period</th><th>Type</th>';
  for(const mk of am)h+='<th>'+esc(ML[mk]||mk)+'</th>';h+='</tr></thead><tbody>';
  for(const f of fl){h+='<tr><td>'+esc(shortDate(f.filing_date))+'</td><td>'+esc(f.form_type)+'</td>';
    for(const mk of am){const m=(f.metrics||{})[mk];
      if(!m||m.value==null){h+='<td style="color:var(--t4)">\u2014</td>';continue}
      const v=m.value,neg=v<0,url=m.source_url||(f.source_urls||{}).filing_index||'';
      h+='<td'+(neg?' class="neg"':'')+'>'+fmtN(v);
      if(url)h+='<a class="src" href="'+url+'" target="_blank">\u2197</a>';
      h+='</td>'}h+='</tr>'}
  return h+'</tbody></table>';
}

function rExtra(){
  const sec=document.getElementById('extra');if(!sec)return;let h='';
  const ws=_filings.filter(f=>f.summary);
  if(ws.length)h+='<div class="sum-sec"><h3>AI Summary</h3><p>'+esc(ws[0].summary)+'</p></div>';
  const ek=_filings.filter(f=>f.form_type==='8-K');
  if(ek.length&&(_ff===null||_ff==='8-K')){
    h+='<div class="tl"><div class="tl-title"><span class="dot"></span>8-K Events ('+ek.length+')</div>';
    for(const f of ek.slice(0,40)){const u=f.source_urls||{};
      h+='<div class="tl-item"><span class="tl-date">'+esc(shortDate(f.filing_date))+'</span>';
      h+='<div class="tl-body">'+esc(f.description||'8-K');
      if(f.items_reported&&f.items_reported.length)h+='<div class="items">Items: '+f.items_reported.join(', ')+'</div>';
      if(f.summary)h+='<div style="margin-top:2px;font-size:11px;color:var(--t3)">'+esc(f.summary)+'</div>';
      if(u.filing_index)h+=' <a href="'+u.filing_index+'" target="_blank">View \u2197</a>';
      h+='</div></div>'}h+='</div>'}
  sec.innerHTML=h;
}

/* ═══ Overview tab builder ═══ */
function bOverview(d){
  const m=d.metrics||{},r=d.ratios||{},pm=d.prior_metrics||{};
  let h='';

  function ovRow(label,val,opts){
    opts=opts||{};const isTotal=opts.total;const isNeg=val!=null&&val<0;
    const cls='ov-row'+(isTotal?' total':'');
    let vc='val';if(isNeg)vc+=' neg';else if(val!=null&&val>0&&opts.green)vc+=' pos';
    let valStr='\u2014';
    if(val!=null)valStr=fmtN(val);
    else if(opts.required)valStr='<span style="color:var(--t4);font-size:12px;font-style:italic">N/A</span>';
    h+='<div class="'+cls+'"><span class="lbl">'+esc(label)+'</span>';
    h+='<span class="'+vc+'">'+valStr+'</span></div>';
    const src=(d.metrics_sourced||{})[opts.key];
    const c=(d.confidence_scores||{})[opts.key];
    if(src&&c!=null&&c<0.9){
      h+='<div style="padding:0 12px;font:400 10px var(--ff);color:var(--t4)">'+esc(src)+' ('+Math.round(c*100)+'% conf.)</div>';
    }
  }
  function ratioLine(txt){h+='<div class="ov-ratio">'+txt+'</div>';}

  h+='<div class="ov-grid">';

  /* ── Left column ── */
  h+='<div class="ov-col">';

  h+='<div class="ov-sec"><h4><span class="dot g"></span>Income</h4>';
  ovRow('Revenue',m.revenue,{green:1,required:1,key:'revenue'});
  ovRow('Cost of Revenue',m.cost_of_revenue,{key:'cost_of_revenue'});
  ovRow('Gross Profit',m.gross_profit,{green:1,total:1,required:1,key:'gross_profit'});
  if(r.gross_margin!=null)ratioLine('Gross Margin: <strong>'+(r.gross_margin*100).toFixed(1)+'%</strong>');
  ovRow('Operating Expenses',m.operating_expenses,{key:'operating_expenses'});
  if(m.sga_expense!=null)ovRow('  SG&A',m.sga_expense,{key:'sga_expense'});
  if(m.rd_expense!=null)ovRow('  R&D',m.rd_expense,{key:'rd_expense'});
  ovRow('Operating Income',m.operating_income,{green:1,key:'operating_income'});
  if(r.operating_margin!=null)ratioLine('Operating Margin: <strong>'+(r.operating_margin*100).toFixed(1)+'%</strong>');
  ovRow('Interest Expense',m.interest_expense,{key:'interest_expense'});
  ovRow('Income Tax',m.income_tax_expense,{key:'income_tax_expense'});
  ovRow('Net Income',m.net_income,{green:1,total:1,required:1,key:'net_income'});
  if(r.net_margin!=null)ratioLine('Net Margin: <strong>'+(r.net_margin*100).toFixed(1)+'%</strong>');
  if(m.eps_basic!=null)h+='<div class="ov-row"><span class="lbl">EPS (Basic / Diluted)</span><span class="val">$'+(m.eps_basic||0).toFixed(2)+' / $'+(m.eps_diluted||m.eps_basic||0).toFixed(2)+'</span></div>';
  h+='</div>';

  h+='<div class="ov-sec"><h4><span class="dot g"></span>Cash Flow</h4>';
  ovRow('Operating Cash Flow',m.operating_cash_flow,{green:1,required:1,key:'operating_cash_flow'});
  ovRow('Capital Expenditures',m.capital_expenditures,{required:1,key:'capital_expenditures'});
  ovRow('Free Cash Flow',m.free_cash_flow,{green:1,total:1,required:1,key:'free_cash_flow'});
  if(m.dividends_paid!=null)ovRow('Dividends Paid',m.dividends_paid,{key:'dividends_paid'});
  if(m.shares_repurchased!=null)ovRow('Share Repurchases',m.shares_repurchased,{key:'shares_repurchased'});
  if(m.investing_cash_flow!=null)ovRow('Investing Cash Flow',m.investing_cash_flow,{key:'investing_cash_flow'});
  if(m.financing_cash_flow!=null)ovRow('Financing Cash Flow',m.financing_cash_flow,{key:'financing_cash_flow'});
  h+='</div>';

  h+='</div>'; /* end left col */

  /* ── Right column ── */
  h+='<div class="ov-col">';

  h+='<div class="ov-sec"><h4><span class="dot b"></span>Balance Sheet</h4>';
  ovRow('Cash & Equivalents',m.cash_and_equivalents,{green:1,required:1,key:'cash_and_equivalents'});
  ovRow('Current Assets',m.current_assets,{green:1,required:1,key:'current_assets'});
  ovRow('Total Assets',m.total_assets,{green:1,total:1,required:1,key:'total_assets'});
  if(r.return_on_assets!=null)ratioLine('ROA: <strong>'+(r.return_on_assets*100).toFixed(1)+'%</strong>');
  ovRow('Current Liabilities',m.current_liabilities,{key:'current_liabilities'});
  ovRow('Total Liabilities',m.total_liabilities,{required:1,key:'total_liabilities'});
  ovRow('Stockholders\u2019 Equity',m.stockholders_equity,{green:1,total:1,required:1,key:'stockholders_equity'});
  if(r.return_on_equity!=null)ratioLine('ROE: <strong>'+(r.return_on_equity*100).toFixed(1)+'%</strong>');
  h+='</div>';

  h+='<div class="ov-sec"><h4><span class="dot b"></span>Debt &amp; Leverage</h4>';
  ovRow('Short-term Debt',m.short_term_debt,{key:'short_term_debt'});
  ovRow('Long-term Debt',m.long_term_debt,{key:'long_term_debt'});
  if(m.total_debt!=null)ovRow('Total Debt',m.total_debt,{total:1});
  if(m.net_debt!=null)ovRow('Net Debt',m.net_debt,{total:1});
  if(r.debt_to_equity!=null)ratioLine('Debt/Equity: <strong>'+(r.debt_to_equity).toFixed(2)+'x</strong>');
  if(r.current_ratio!=null)ratioLine('Current Ratio: <strong>'+(r.current_ratio).toFixed(2)+'x</strong>');
  h+='</div>';

  h+='</div>'; /* end right col */
  h+='</div>'; /* end ov-grid */

  /* Period comparison — uses correct labels from backend */
  const yoyLabel=d.yoy_label||'vs Prior Year';
  const compLabel=d.comparison_label||'YoY';
  const qm=d.qoq_metrics||{};
  if(pm&&Object.keys(pm).length>0){
    const yoys=[{k:'revenue',l:'Revenue'},{k:'net_income',l:'Net Income'},{k:'gross_profit',l:'Gross Profit'},
      {k:'operating_income',l:'Op. Income'},{k:'ebitda',l:'EBITDA'},{k:'total_assets',l:'Total Assets'},
      {k:'operating_cash_flow',l:'Op. Cash Flow'},{k:'free_cash_flow',l:'Free CF'}];
    let hasYoy=yoys.some(y=>m[y.k]!=null&&pm[y.k]!=null);
    if(hasYoy){
      h+='<div class="yoy-sec"><h3>'+esc(yoyLabel)+'</h3>';
      for(const y of yoys){
        const cur=m[y.k],prev=pm[y.k];if(cur==null||prev==null)continue;
        const delta=prev!==0?((cur-prev)/Math.abs(prev)*100):0;
        const up=delta>=0;
        h+='<div class="yoy-row"><span class="lbl">'+y.l+'</span>';
        h+='<span class="val">'+fmtN(cur)+'</span>';
        h+='<span class="delta '+(up?'up':'dn')+'">'+(up?'\u25B2':'\u25BC')+' '+Math.abs(delta).toFixed(1)+'%</span>';
        h+='<span class="prev">vs '+fmtN(prev)+'</span></div>';
      }
      h+='</div>';
    }
    /* QoQ comparison for quarterly */
    let hasQoq=Object.keys(qm).length>0&&yoys.some(y=>m[y.k]!=null&&qm[y.k]!=null);
    if(hasQoq){
      h+='<div class="yoy-sec"><h3>vs Prior Quarter (QoQ)</h3>';
      for(const y of yoys){
        const cur=m[y.k],prev=qm[y.k];if(cur==null||prev==null)continue;
        const delta=prev!==0?((cur-prev)/Math.abs(prev)*100):0;
        const up=delta>=0;
        h+='<div class="yoy-row"><span class="lbl">'+y.l+'</span>';
        h+='<span class="val">'+fmtN(cur)+'</span>';
        h+='<span class="delta '+(up?'up':'dn')+'">'+(up?'\u25B2':'\u25BC')+' '+Math.abs(delta).toFixed(1)+'%</span>';
        h+='<span class="prev">vs '+fmtN(prev)+'</span></div>';
      }
      h+='</div>';
    }
  }
  return h;
}

/* ═══ Semantic row color classification ═══ */
/* Revenue → green, Expenses → red, Totals → white bold (handled by isTotal flag) */
const REV_KW=['revenue','net sales','net revenue','total revenue','sales and','service revenue','product revenue','subscription revenue'];
const EXP_KW=['expense','cost of','depreciation','amortization','impairment','write-off','write off','restructuring','provision for'];
/* Expanded total patterns — covers subtotals that contain "loss" or "income" */
const TOT_PATTERNS=[
  /^total/,/^net income/,/^gross profit/,/^operating income/,/^operating loss/,
  /income from operations/,/loss from operations/,/income before/,/income \(loss\) from/,
  /loss \(income\) from/,/^ebitda/,/earnings before/,/^pre-tax/,/^pretax/,
  /^earnings per share/,/net earnings/,/net loss/,/attributable to/,
];
function isSubtotalRow(ll,rec){
  if(rec&&rec.is_total)return true;
  for(const pat of TOT_PATTERNS)if(pat.test(ll))return true;
  return false;
}
function rowClass(label,isTot){
  if(isTot)return''; /* totals get white/bold, no color override */
  const ll=label.toLowerCase();
  for(const w of REV_KW)if(ll.includes(w))return' row-rev';
  for(const w of EXP_KW)if(ll.includes(w))return' row-exp';
  return'';
}

function bStmt(recs,stmtLabel){
  if(!recs||!recs.length)return'<div style="padding:18px;color:var(--t4);font-size:12px">No statement data</div>';
  const skip=new Set(['concept','standard_concept','level','is_abstract','is_total','abstract','units','decimals']);
  const cols=Object.keys(recs[0]).filter(k=>!skip.has(k));
  const lc=cols.find(c=>c==='label'||c==='Label')||cols[0];const vc=cols.filter(c=>c!==lc);
  let h='<div class="stmt-hdr"><h3>'+esc(stmtLabel)+'</h3>';
  if(_tk)h+='<span class="stmt-tk">'+esc(_tk)+'</span>';
  h+='</div>';
  h+='<table class="st"><thead><tr><th>'+esc(lc)+'</th>';
  for(const c of vc)h+='<th>'+esc(sCol(c))+'</th>';h+='</tr></thead><tbody>';
  for(let ri=0;ri<recs.length;ri++){const rec=recs[ri];const lab=String(rec[lc]||'');
    const ll=lab.toLowerCase();
    const isTotal=isSubtotalRow(ll,rec);
    const isSection=rec.is_abstract||(!rec[vc[0]]&&!rec[vc[1]]&&lab&&!isTotal);
    const rc=rowClass(lab,isTotal);
    let cls='';
    if(isSection)cls='sec-hdr';
    else if(isTotal)cls='sub';
    cls+=rc;
    h+='<tr class="'+cls+'">';
    if(isSection){h+='<td colspan="'+(vc.length+1)+'">'+esc(lab)+'</td></tr>';continue}
    h+='<td>'+esc(lab)+'</td>';
    for(const c of vc){const v=rec[c];
      if(v==null||v==='')h+='<td style="color:var(--t4)">\u2014</td>';
      else if(typeof v==='number'||!isNaN(Number(v))){const n=Number(v);
        h+='<td'+(n<0?' class="neg"':'')+'>'+fmtN(n);
        h+='<button class="fc-btn" onclick="factCheck(\''+esc(lab)+'\',\''+fmtN(n)+'\',event)" title="Verify">&#x2713;</button></td>'}
      else h+='<td>'+esc(String(v))+'</td>'}h+='</tr>'}
  return h+'</tbody></table>';
}
function sCol(c){if(/^\d{4}-\d{2}/.test(c))try{return new Date(c).toLocaleDateString('en-US',{month:'short',year:'numeric'})}catch(e){}return c.length>14?c.slice(0,12)+'..':c}
function swMainTab(el,id){
  el.parentElement.querySelectorAll('.main-tab').forEach(t=>t.classList.remove('act'));el.classList.add('act');
  const parent=el.closest('.col-r');if(!parent)return;
  parent.querySelectorAll('.main-pane').forEach(p=>p.style.display='none');
  const pane=document.getElementById(id);if(pane)pane.style.display='';
  /* Auto-generate exec summaries on first click */
  if(id==='mt-exec'&&!_execSummaries&&_curData)genExecSummaries();
}
function swTab(el,id){
  const tabs=el.closest('.dash-stmts')||el.closest('.main-pane')||el.closest('.col-r')||el.parentElement.parentElement;
  el.parentElement.querySelectorAll('.st-tab').forEach(t=>t.classList.remove('act'));el.classList.add('act');
  tabs.querySelectorAll('.st-wrap').forEach(w=>w.style.display='none');document.getElementById(id).style.display=''}

function popTbl(){const a=document.getElementById('tbl');if(!a)return;
  document.getElementById('m-title').textContent=(_tk||'')+' \u2014 '+(_ff||'All')+' Filings';
  document.getElementById('m-body').innerHTML=a.innerHTML;document.getElementById('modal').style.display='flex'}
function closeModal(e){if(e&&e.target!==document.getElementById('modal'))return;document.getElementById('modal').style.display='none'}
document.addEventListener('keydown',e=>{if(e.key==='Escape'){document.getElementById('modal').style.display='none';document.getElementById('fc-popup').style.display='none'}});

function fmtN(n){if(n==null||isNaN(n))return'\u2014';const s=n<0?'-':'',a=Math.abs(n);
  if(a>=1e12)return s+'$'+(a/1e12).toFixed(1)+'T';if(a>=1e9)return s+'$'+(a/1e9).toFixed(1)+'B';
  if(a>=1e6)return s+'$'+(a/1e6).toFixed(0)+'M';if(a>=1e3)return s+'$'+(a/1e3).toFixed(0)+'K';
  if(a>0&&a<1)return(n*100).toFixed(1)+'%';if(a===0)return'\u2014';return s+'$'+a.toLocaleString()}
function shortDate(d){if(!d)return'?';if(d.length>=10)try{return new Date(d).toLocaleDateString('en-US',{month:'short',year:'numeric'})}catch(e){}return d.slice(0,7)}

/* ═══ Tool selector ═══ */
async function onToolSel(){
  const sel=document.getElementById('sel-tool');
  if(!sel||!sel.value||!_tk)return;
  const tool=sel.value;

  /* Dashboard — re-render cached data (no API call) */
  if(tool==='dashboard'){if(_curData)render(_curData,'');return}

  /* Direct API calls — no chat routing, no company misrouting.
     Each view calls its specific endpoint with the locked ticker. */
  showLd();
  try{
    if(tool==='entity'){
      /* Entity profile — direct endpoint */
      const r=await fetch('/api/entity/'+encodeURIComponent(_tk));
      const j=await r.json();
      stopLd();
      if(j.error){addA(j.error,1);return}
      renderEntity(j.profile||j);
    }else if(tool==='mda'||tool==='risk_factors'||tool==='business'||tool==='executive_compensation'){
      /* Filing sections — direct endpoint, uses the currently selected period's filing */
      let url='/api/filing-text/'+encodeURIComponent(_tk)+'/'+encodeURIComponent(tool);
      const params=[];
      if(_curAcc)params.push('accession='+encodeURIComponent(_curAcc));
      if(_curData&&_curData.filing_info)params.push('form_type='+encodeURIComponent(_curData.filing_info.form_type||''));
      if(params.length)url+='?'+params.join('&');
      const r=await fetch(url);
      const j=await r.json();
      stopLd();
      if(j.error){addA(j.error,1);return}
      /* Fill in filing_date from current data if the endpoint didn't return it */
      if(!j.filing_date&&_curData&&_curData.filing_info)j.filing_date=_curData.filing_info.filing_date;
      if(!j.form_type&&_curData&&_curData.filing_info)j.form_type=_curData.filing_info.form_type;
      renderFiling(j);
    }else{
      /* Unknown view — fall back to chat (shouldn't happen) */
      const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({message:_tk+' '+tool})});
      const j=await r.json();
      if(j.type==='error'){stopLd();addA(j.message,1)}
      else if(j.type==='result')handleRes(j);
      else{stopLd();addA(j.message||'Loaded')}
    }
  }catch(e){stopLd();addA('Error: '+e.message,1)}
}

/* ═══ Exec Summary Generation ═══ */
async function genExecSummaries(){
  if(!_curData)return;
  if(_execSummaries)return; /* already generated */
  /* Send only focused context: metrics, ratios, prior, and top statement rows.
     Do NOT send _bgSections (filing text) — too large, makes Claude slow/fail. */
  const focused={
    company_name:_curData.company_name,
    ticker_or_cik:_curData.ticker_or_cik,
    filing_info:_curData.filing_info,
    period_type:_curData.period_type,
    industry_class:_curData.industry_class,
    metrics:_curData.metrics,
    ratios:_curData.ratios,
    prior_metrics:_curData.prior_metrics,
    yoy_label:_curData.yoy_label,
    validation:_curData.validation,
    income_statement:(_curData.income_statement||[]).slice(0,15),
    balance_sheet:(_curData.balance_sheet||[]).slice(0,15),
    cash_flow_statement:(_curData.cash_flow_statement||[]).slice(0,15),
  };
  try{
    const ctrl=new AbortController();
    const timeout=setTimeout(()=>ctrl.abort(),30000); /* 30s timeout */
    const body={ticker:_tk||'',context:focused,sections:{}};
    const r=await fetch('/api/exec-summaries',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:ctrl.signal});
    clearTimeout(timeout);
    const j=await r.json();
    _execSummaries=j;

    /* Update inline exec summary in main view */
    const inlineBody=document.getElementById('exec-inline-body');
    const inlineBadge=document.getElementById('exec-badge');
    if(inlineBody){
      let ih='';
      ih+='<div style="margin-bottom:16px">'+mdCb(j.overall||'')+'</div>';
      if(j.income)ih+='<h4>Income Statement</h4><div style="margin-bottom:12px">'+mdCb(j.income)+'</div>';
      if(j.balance_sheet)ih+='<h4>Balance Sheet</h4><div style="margin-bottom:12px">'+mdCb(j.balance_sheet)+'</div>';
      if(j.cash_flow)ih+='<h4>Cash Flow</h4><div>'+mdCb(j.cash_flow)+'</div>';
      inlineBody.innerHTML=ih;
    }
    if(inlineBadge){inlineBadge.textContent='AI Generated';inlineBadge.classList.add('done')}

    /* Also update the tab pane version */
    const el=document.getElementById('exec-content');
    const ld=document.getElementById('exec-loading');
    if(ld)ld.style.display='none';
    if(el){
      let h='';
      const cards=[
        {key:'overall',title:'Overall Executive Summary',dot:'ov'},
        {key:'income',title:'Income Statement Analysis',dot:'inc'},
        {key:'balance_sheet',title:'Balance Sheet Analysis',dot:'bs'},
        {key:'cash_flow',title:'Cash Flow Analysis',dot:'cf'}
      ];
      for(const c of cards){
        const txt=j[c.key]||'Not available';
        h+='<div class="exec-card"><h4><span class="edot '+c.dot+'"></span>'+c.title+'</h4>';
        h+='<div class="exec-body">'+mdCb(txt)+'</div></div>';
      }
      el.innerHTML=h;
    }
  }catch(e){
    const inlineBody=document.getElementById('exec-inline-body');
    const inlineBadge=document.getElementById('exec-badge');
    const msg=e.name==='AbortError'?'Timed out — try again or check API key':'Failed: '+esc(e.message);
    if(inlineBody)inlineBody.innerHTML='<span style="color:var(--t3);font-size:13px">'+msg+'</span>';
    if(inlineBadge){inlineBadge.textContent='Retry';inlineBadge.style.color='var(--acc)';inlineBadge.style.cursor='pointer';inlineBadge.onclick=()=>{_execSummaries=null;genExecSummaries()}}
    const ld=document.getElementById('exec-loading');
    if(ld)ld.innerHTML='<span style="color:var(--red)">Error: '+esc(e.message)+'</span>';
  }
}

/* ═══ Background section loading ═══ */
async function bgLoadSections(tk){
  if(!tk)return;
  _bgSections={};
  const secs=['mda','risk_factors','business'];
  for(const sec of secs){
    try{
      const r=await fetch('/api/section/'+encodeURIComponent(tk)+'/'+sec);
      const j=await r.json();
      if(j.text&&j.text.length>100)_bgSections[sec]=j.text.slice(0,15000);
    }catch(e){}
  }
}

/* ═══ Right-panel chatbot (AI Assistant) ═══ */
let _cbOpen=false;
function toggleCb(){
  _cbOpen=!_cbOpen;
  const p=document.getElementById('cbpanel');const b=document.getElementById('cb-btn');const fab=document.getElementById('chat-fab');
  if(p)p.classList.toggle('open',_cbOpen);
  if(b)b.classList.toggle('active',_cbOpen);
  if(fab)fab.classList.toggle('active',_cbOpen);
  if(_cbOpen){setTimeout(()=>{const inp=document.getElementById('cb-inp');if(inp)inp.focus()},100)}
}
/* Open chatbot panel and show data-aware welcome */
function openCbWithData(){
  if(!_cbOpen)toggleCb();
  /* Clear previous messages and show data-aware welcome */
  const msgs=document.getElementById('cb-msgs');if(!msgs)return;
  const w=document.getElementById('cb-welcome');if(w)w.remove();
  const d=_curData;if(!d)return;
  const name=d.company_name||_tk||'Company';
  const m=d.metrics||{};
  /* Add data-aware suggestion chips */
  let chips='<div class="cb-chips">';
  chips+='<span class="cb-chip" onclick="cbQ(\'Give me an executive summary of '+esc(name)+'\')">Executive Summary</span>';
  if(m.revenue!=null)chips+='<span class="cb-chip" onclick="cbQ(\'Analyze revenue and margins\')">Revenue Analysis</span>';
  chips+='<span class="cb-chip" onclick="cbQ(\'What are the key risks?\')">Key Risks</span>';
  if(m.operating_cash_flow!=null)chips+='<span class="cb-chip" onclick="cbQ(\'How is cash flow generation?\')">Cash Flow</span>';
  chips+='</div>';
  addCbMsg('**'+name+'** data loaded. '+chips,'ai');
}
function cbQ(t){const inp=document.getElementById('cb-inp');if(inp){inp.value=t;sendCb()}}
document.addEventListener('keydown',e=>{
  if(e.target&&e.target.id==='cb-inp'&&e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendCb()}
});
async function sendCb(){
  const inp=document.getElementById('cb-inp');const btn=document.getElementById('cb-send');
  if(!inp||!inp.value.trim())return;
  const msg=inp.value.trim();inp.value='';if(btn)btn.disabled=true;
  const w=document.getElementById('cb-welcome');if(w)w.remove();
  addCbMsg(msg,'user');
  const tid='cbt'+Date.now();addCbTyping(tid);
  try{
    const ctx=Object.assign({},_curData||{});
    if(_bgSections&&Object.keys(_bgSections).length)ctx._filing_sections=_bgSections;
    const body={message:msg,ticker:_tk||'',context:ctx,history:_chatHistory.slice(-6)};
    const r=await fetch('/api/chatbot',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)});
    const j=await r.json();rmCb(tid);
    const ans=j.answer||'No response.';
    addCbMsg(ans,'ai',j.citations);
    _chatHistory.push({role:'user',content:msg});
    _chatHistory.push({role:'assistant',content:ans});
    if(_chatHistory.length>20)_chatHistory=_chatHistory.slice(-12);
  }catch(e){rmCb(tid);addCbMsg('Error: '+e.message,'ai')}
  if(btn)btn.disabled=false;if(inp)inp.focus();
}
function addCbMsg(text,role,cits){
  const msgs=document.getElementById('cb-msgs');if(!msgs)return;
  const d=document.createElement('div');d.className='cb-msg '+role;
  let inner='<div class="cb-bubble">';
  if(role==='ai'){
    inner+=mdCb(text);
    if(cits&&cits.length){inner+='<div class="cb-cits">';
      for(const c of cits)inner+='<span class="cb-cit">'+esc(c.source||'')+'</span>';
      inner+='</div>'}
  }else{inner+=esc(text)}
  inner+='</div>';d.innerHTML=inner;msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;
}
function addCbTyping(id){
  const msgs=document.getElementById('cb-msgs');if(!msgs)return;
  const d=document.createElement('div');d.id=id;d.className='cb-msg ai';
  d.innerHTML='<div class="cb-bubble"><div class="cb-typing"><span></span><span></span><span></span></div></div>';
  msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;
}
function rmCb(id){const e=document.getElementById(id);if(e)e.remove()}
function mdCb(s){if(!s)return'';
  let h=esc(s);
  h=h.replace(/### (.+)/g,'<h4 style="margin:12px 0 6px;font-size:14px;color:var(--t1)">$1</h4>');
  h=h.replace(/## (.+)/g,'<h3 style="margin:14px 0 8px;font-size:15px;color:var(--t1)">$1</h3>');
  h=h.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  h=h.replace(/\*(.+?)\*/g,'<em>$1</em>');
  h=h.replace(/^- /gm,'<li>').replace(/\n- /g,'</li><li>');
  if(h.includes('<li>'))h='<ul>'+h+'</ul>';
  h=h.replace(/\n\n/g,'<br><br>').replace(/\n/g,'<br>');
  return h;
}

/* ═══ Theme toggle (light/dark) ═══ */
function toggleTheme(){
  const isLight=document.body.classList.toggle('light');
  localStorage.setItem('sec-theme',isLight?'light':'dark');
  const btn=document.getElementById('theme-btn');
  if(btn)btn.innerHTML=isLight?'\u2600':'\u263E';
}
(function initTheme(){
  const saved=localStorage.getItem('sec-theme');
  if(saved==='light'){document.body.classList.add('light');
    const btn=document.getElementById('theme-btn');if(btn)btn.innerHTML='\u2600';}
})();
