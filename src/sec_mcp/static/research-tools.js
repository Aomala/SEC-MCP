/* Comps, filing reader and screener. All values come from the existing SEC API. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const finite = v => v == null || v === '' || !Number.isFinite(Number(v)) ? null : Number(v);
  const validTicker = v => /^[A-Z0-9][A-Z0-9.-]{0,11}$/.test(v);
  const companies = {META:'Meta Platforms',MSFT:'Microsoft',AAPL:'Apple',GOOGL:'Alphabet',GOOG:'Alphabet',AMZN:'Amazon',NVDA:'NVIDIA',AMD:'AMD',INTC:'Intel',TSLA:'Tesla',JPM:'JPMorgan Chase',BAC:'Bank of America',C:'Citigroup',DELL:'Dell Technologies'};
  const historyCache = new Map(), historyPending = new Map(), textCache = new Map();
  const params = new URLSearchParams(location.search);
  const initial = window.FineasShell?.context() || {};
  const state = {
    initialized:false, view:null,
    compare:{tickers:[],mode:'annual',year:'',quarter:'Q4',rows:[],seq:0,sort:'revenue',direction:-1,peersSeq:0},
    reader:{ticker:'',filings:[],selected:null,section:'business',raw:'',matches:[],match:-1,seq:0,textSeq:0,controller:null,textController:null,started:false},
    screen:{rows:[],selection:new Set(),seq:0,controller:null,sectorsLoaded:false,ran:false}
  };
  function node(tag,text,className) { const el=document.createElement(tag); if(text!=null)el.textContent=text; if(className)el.className=className; return el; }
  function status(id,text,error=false) { $(id).textContent=text; $(id).classList.toggle('error',error); }
  function empty(tbody,title,detail='') { const tr=node('tr'),td=node('td',null,'tools-empty'); td.colSpan=7; td.append(node('strong',title),document.createTextNode(detail)); tr.append(td); tbody.replaceChildren(tr); }
  function compact(v) { if(v==null)return '—'; const a=Math.abs(v); return (v<0?'−':'')+(a>=1e12?(a/1e12).toFixed(2)+'T':a>=1e9?(a/1e9).toFixed(1)+'B':a>=1e6?(a/1e6).toFixed(1)+'M':a.toLocaleString(undefined,{maximumFractionDigits:2})); }
  function money(v,currency) { if(v==null)return '—'; let symbol=currency+' '; try{symbol=new Intl.NumberFormat('en',{style:'currency',currency,currencyDisplay:'narrowSymbol'}).formatToParts(0).find(p=>p.type==='currency').value;}catch{} return (v<0?'−':'')+symbol+compact(Math.abs(v)); }
  function percent(v) { return v==null?'—':(v*100).toFixed(1)+'%'; }
  function ratio(n,d) { return n!=null&&d!=null&&d!==0?n/d:null; }
  function periodLabel(row) { return (row.fiscalPeriod==='FY'?'FY':row.fiscalPeriod+' FY')+' '+row.fiscalYear; }
  function safeUrl(value) {try{const u=new URL(value); return /^https?:$/.test(u.protocol)&&!u.username&&!u.password?u.href:null;}catch{return null;} }
  function updateURL(values) { const u=new URL(location.href); Object.entries(values).forEach(([key,value])=>value==null||value===''?u.searchParams.delete(key):u.searchParams.set(key,String(value))); history.replaceState(null,'',u); window.FineasShell?.refresh(state.view); }
  function companyHref(ticker,row) {const p=new URLSearchParams({ticker,period:row?.fiscalPeriod==='FY'?'annual':'quarterly'}); if(!row)p.set('period','annual'); if(row){p.set('year',row.fiscalYear);if(row.fiscalPeriod!=='FY')p.set('quarter',row.fiscalPeriod);} return '/explorer?'+p;}
  async function request(path,options={}) {
    const controller=new AbortController(), parent=options.signal;
    const abort=()=>controller.abort(); if(parent?.aborted)abort(); else parent?.addEventListener('abort',abort,{once:true});
    const timeout=setTimeout(abort,65000);
    try {
      const response=await fetch(API_BASE+path,{...options,signal:controller.signal});
      const data=await response.json().catch(()=>null);
      if(!response.ok)throw new Error(response.status===429?'The data service is busy or its request limit was reached. Please try again shortly.':'The data service could not complete this request (HTTP '+response.status+'). Try again.');
      if(!data||typeof data!=='object'||data.error)throw new Error('The data service returned an incomplete response. Please try again.');
      return data;
    } finally {clearTimeout(timeout);parent?.removeEventListener('abort',abort);}
  }
  function message(error){return error.name==='AbortError'?'This request took too long. Try a smaller group or a single sector.':error instanceof TypeError?'The connection was interrupted. Please try again.':error.message;}
  function periods(data,mode) {
    const cfs=new Map((data.cashflow||[]).map(r=>[r.date,r])), result=new Map();
    for(const r of data.income||[]) {
      const meta=r._meta||{},year=String(r.fiscalYear||'');
      if(!/^\d{4}$/.test(year)||!/^\d{4}-\d{2}-\d{2}$/.test(r.date||''))continue;
      if(mode==='quarter'? !/^Q[1-4]$/.test(r.period)||meta.isYtd||meta.quality==='ytd_fallback':r.period!=='FY')continue;
      const currency=r.reportedCurrency||'USD',cf=cfs.get(r.date)||{};
      const match=String(cf.fiscalYear)===year&&cf.period===r.period&&(cf.reportedCurrency||'USD')===currency&&!(mode==='quarter'&&(cf._meta?.isYtd||cf._meta?.quality==='ytd_fallback'));
      const ocf=match?finite(cf.operatingCashFlow):null,capex=match?finite(cf.capitalExpenditure):null;
      const revenue=finite(r.revenue),netIncome=finite(r.netIncome);
      const row={fiscalYear:year,fiscalPeriod:r.period,endDate:r.date,currency,accession:meta.accession,quality:meta.quality,metrics:{revenue,netIncome,grossMargin:ratio(finite(r.grossProfit),revenue),netMargin:ratio(netIncome,revenue),freeCashFlow:ocf!=null&&capex!=null?ocf-Math.abs(capex):match?finite(cf.freeCashFlow):null,eps:finite(r.epsdiluted??r.epsDiluted)}};
      const key=year+r.period;if(!result.has(key)||result.get(key).endDate<r.date)result.set(key,row);
    }
    return [...result.values()].sort((a,b)=>b.endDate.localeCompare(a.endDate));
  }
  async function financials(ticker,mode) {
    const key=ticker+'|'+mode;
    if(historyCache.has(key))return historyCache.get(key);
    // A year or quarter change only selects another row from the same history.
    // Share the bounded request so quick filter changes do not restart it.
    if(!historyPending.has(key))historyPending.set(key,(async()=>{
      try {
        const data=await request('/api/financials-history/'+encodeURIComponent(ticker)+'?period='+mode+'&limit=12');
        if(!Array.isArray(data.income))throw new Error('No financial history was returned for '+ticker+'.');
        const rows=periods(data,mode);historyCache.set(key,rows);return rows;
      } finally {historyPending.delete(key);}
    })());
    return historyPending.get(key);
  }
  function comparisonSettings(){const c=state.compare;c.mode=$('compare-mode').value;c.year=$('compare-year').value;c.quarter=$('compare-quarter').value;$('compare-quarter-label').hidden=c.mode!=='quarter';updateURL({view:'comps',ticker:c.tickers[0],tickers:c.tickers.join(','),period:c.mode==='quarter'?'quarterly':'annual',year:c.year,quarter:c.mode==='quarter'?c.quarter:null});}
  function compareYears(){const c=state.compare,years=new Set(c.rows.flatMap(r=>(r.periods||[]).map(p=>p.fiscalYear)));if(c.year)years.add(c.year);$('compare-year').replaceChildren(new Option('Latest available',''),...[...years].sort().reverse().map(y=>new Option('FY '+y,y)));$('compare-year').value=c.year;}
  function renderChips(){const c=state.compare;$('compare-chips').replaceChildren(...c.tickers.map(t=>{const chip=node('span',t,'tool-chip'),button=node('button','×');button.type='button';button.setAttribute('aria-label','Remove '+t);button.onclick=()=>{c.tickers=c.tickers.filter(v=>v!==t);c.peersSeq++;renderChips();loadComparison();};chip.append(button);return chip;}));}
  function pickPeriod(entry){const c=state.compare;return (entry.periods||[]).find(p=>(!c.year||p.fiscalYear===c.year)&&(c.mode==='annual'||p.fiscalPeriod===c.quarter));}
  function renderComparison(){
    const c=state.compare,body=$('compare-rows');
    if(!c.tickers.length){empty(body,'Choose companies to compare','Add a ticker or start with a group above.');$('compare-summary').hidden=true;return;}
    const entries=c.rows.map(r=>({...r,selected:pickPeriod(r)}));
    entries.sort((a,b)=>{const x=a.selected?.metrics[c.sort],y=b.selected?.metrics[c.sort];return x==null?(y==null?0:1):y==null?-1:c.direction*(x-y);});
    document.querySelectorAll('#compare-table [data-sort]').forEach(th=>th.setAttribute('aria-sort',th.dataset.sort===c.sort?(c.direction===1?'ascending':'descending'):'none'));
    const valid=entries.filter(r=>r.selected),currencies=new Set(valid.map(r=>r.selected.currency));
    const max=currencies.size===1?Math.max(0,...valid.map(r=>r.selected.metrics.revenue||0)):0;
    body.replaceChildren(...entries.map(entry=>{
      const tr=node('tr');tr.dataset.ticker=entry.ticker;const name=node('td'),link=node('a',entry.ticker);link.href=companyHref(entry.ticker,entry.selected);name.append(link,node('small',companies[entry.ticker]||'Open company financials'));tr.append(name);
      if(!entry.selected){const td=node('td',entry.loading?'Loading financials…':entry.error||'This fiscal period is unavailable.');td.colSpan=6;td.style.textAlign='left';if(entry.error){const retry=node('button','Retry','tool-button');retry.type='button';retry.setAttribute('aria-label','Retry '+entry.ticker);retry.style.marginLeft='12px';retry.onclick=loadComparison;td.append(retry);}tr.append(td);return tr;}
      const r=entry.selected,m=r.metrics,p=node('td',periodLabel(r)+(r.quality==='q4_synthesized'?'*':''));p.append(node('small',r.endDate+' · '+r.currency));tr.append(p);
      for(const [key,text] of [['revenue',money(m.revenue,r.currency)],['grossMargin',percent(m.grossMargin)],['netMargin',percent(m.netMargin)],['freeCashFlow',money(m.freeCashFlow,r.currency)],['eps',m.eps==null?'—':money(m.eps,r.currency)]]){const cell=node('td',text);if(key==='revenue'&&m.revenue>0&&max){const bar=node('span',null,'number-bar');bar.style.width=(m.revenue/max*100)+'%';bar.setAttribute('aria-hidden','true');cell.append(bar);}tr.append(cell);}
      return tr;
    }));
    const summary=$('compare-summary');summary.replaceChildren();summary.hidden=!valid.length;
    summary.append(node('span',valid.length+' of '+c.tickers.length+' companies have this period'));
    if(valid.length>1&&currencies.size===1){const top=valid.filter(x=>x.selected.metrics.revenue!=null).sort((a,b)=>b.selected.metrics.revenue-a.selected.metrics.revenue)[0];if(top){const item=node('span','Largest reported revenue: ');item.append(node('strong',top.ticker+' · '+money(top.selected.metrics.revenue,top.selected.currency)));summary.append(item);}}
    if(currencies.size>1)summary.append(node('span','Mixed currencies — amounts are not normalized.'));
    if(!c.year)summary.append(node('span','Latest periods may differ; check each row.'));
  }
  async function loadComparison(){
    const c=state.compare;comparisonSettings();const mode=c.mode,seq=++c.seq;
    c.rows=c.tickers.map(ticker=>({ticker,loading:true}));renderComparison();
    if(!c.tickers.length){status('compare-status','Add up to eight companies.');return;}
    status('compare-status','Loading financials. Each company appears as soon as it is ready.');let done=0;
    await Promise.all(c.rows.map(async entry=>{try{const rows=await financials(entry.ticker,mode);if(seq!==c.seq)return;entry.periods=rows;}catch(e){if(seq!==c.seq)return;entry.error=message(e);}finally{if(seq===c.seq){entry.loading=false;done++;compareYears();renderComparison();status('compare-status',done<c.rows.length?done+' of '+c.rows.length+' companies loaded…':c.rows.filter(r=>r.error).length?'Some companies could not load. Use Retry beside a company to try again.':'Financials ready. Select a column heading to sort or a company to explore.');}}}));
  }
  function setTickers(values){const tickers=[...new Set(values.map(v=>v.trim().toUpperCase()).filter(Boolean))];if(tickers.some(t=>!validTicker(t))||tickers.length>8){status('compare-status','Use up to eight valid ticker symbols, separated by commas.',true);return false;}state.compare.tickers=tickers;state.compare.peersSeq++;renderChips();loadComparison();return true;}
  async function findPeers(){const c=state.compare,tk=c.tickers[0];if(!tk){status('compare-status','Add your main company first.',true);return;}const seq=++c.peersSeq;status('compare-status','Finding peers for '+tk+'…');try{const d=await request('/api/peers/'+encodeURIComponent(tk));if(seq!==c.peersSeq)return;const peers=(d.peers||[]).map(p=>typeof p==='string'?p:p.ticker).filter(p=>validTicker(p||''));if(!peers.length){status('compare-status','No suggested peers returned. Add tickers manually.');return;}setTickers([...new Set([...c.tickers,...peers])].slice(0,8));}catch(e){if(seq===c.peersSeq)status('compare-status',message(e),true);}}

  function clearDocument(text){const r=state.reader;r.raw='';r.matches=[];r.match=-1;$('reader-matches').textContent='';$('reader-doc').replaceChildren(node('p',text,'reader-status'));$('reader-prev').disabled=$('reader-next').disabled=true;}
  function filingType(f){return /^(10-K|20-F|40-F)/.test(f.form_type||'')?'annual':/^(10-Q|6-K)/.test(f.form_type||'')?'quarter':'other';}
  function visibleFilings(){return state.reader.filings.filter(f=>($('reader-type').value==='all'||filingType(f)===$('reader-type').value)&&(!$('reader-year').value||(f.filing_date||'').slice(0,4)===$('reader-year').value));}
  function renderFilingList(){const r=state.reader,rows=visibleFilings(),list=$('reader-list');list.replaceChildren();if(!rows.length){list.append(node('p','No filings match these filters. Try another year or report type.','reader-status'));return;}for(const f of rows){const b=node('button');b.type='button';b.append(node('strong',f.form_type),node('span','Filed '+f.filing_date));b.setAttribute('aria-current',String(r.selected?.accession===f.accession));b.onclick=()=>selectFiling(f);list.append(b);}}
  function selectFiling(f){const r=state.reader;r.selected=f;renderFilingList();const mode=filingType(f),period=historyCache.get(r.ticker+'|'+mode)?.find(p=>p.accession===f.accession);$('reader-explorer').href=companyHref(r.ticker,period);updateURL({view:'filings',ticker:r.ticker,accession:f.accession,section:r.section,form:$('reader-type').value,filingYear:$('reader-year').value,period:mode==='quarter'?'quarterly':'annual',year:period?.fiscalYear,quarter:period?.fiscalPeriod!=='FY'?period?.fiscalPeriod:null});loadSection();}
  function filterFilings(){const r=state.reader,rows=visibleFilings();if(rows.length)selectFiling(rows.find(f=>f.accession===r.selected?.accession)||rows[0]);else{r.selected=null;r.textSeq++;r.textController?.abort();renderFilingList();clearDocument('No report matches the current filters.');$('reader-meta').replaceChildren();updateURL({accession:null,form:$('reader-type').value,filingYear:$('reader-year').value});}}
  async function loadReader(){
    const r=state.reader,tk=$('reader-company').value.trim().toUpperCase();if(!validTicker(tk)){status('reader-status','Choose a company ticker, such as META or MSFT.',true);return;}
    r.controller?.abort();r.textController?.abort();r.textSeq++;const controller=r.controller=new AbortController(),seq=++r.seq;r.ticker=tk;r.selected=null;r.filings=[];
    const first=!r.started;r.started=true;$('reader-list').replaceChildren(node('p','Loading available reports…','reader-status'));$('reader-meta').replaceChildren();clearDocument('Loading the filing list…');status('reader-status','Finding SEC filings for '+tk+'…');$('reader-explorer').href=companyHref(tk);updateURL({view:'filings',ticker:tk,accession:null});
    try {
      const [data,hist]=await Promise.all([request('/api/filings/'+encodeURIComponent(tk)+'?limit=80',{signal:controller.signal}),first&&initial.year?financials(tk,initial.period==='quarterly'?'quarter':'annual').catch(()=>[]):Promise.resolve([])]);
      if(seq!==r.seq)return;
      if(!Array.isArray(data.filings))throw new Error('The service did not return a filing list.');r.filings=data.filings.filter(f=>typeof f.accession==='string'&&/^\d{10}-\d{2}-\d{6}$/.test(f.accession)).sort((a,b)=>(b.filing_date||'').localeCompare(a.filing_date||''));
      const years=[...new Set(r.filings.map(f=>(f.filing_date||'').slice(0,4)).filter(y=>/^\d{4}$/.test(y)))].sort().reverse();
      const year=first?params.get('filingYear')||'':'';$('reader-year').replaceChildren(new Option('All years',''),...years.map(y=>new Option(y,y)));$('reader-year').value=years.includes(year)?year:'';
      const requested=first?(params.get('accession')||hist.find(p=>p.fiscalYear===initial.year&&(!initial.quarter||p.fiscalPeriod===initial.quarter))?.accession):null;
      let chosen=requested?r.filings.find(f=>f.accession===requested):visibleFilings()[0];
      status('reader-status',r.filings.length+' reports available for '+tk+'. Dates below are filing dates, not fiscal year labels.');
      if(chosen){if(requested){$('reader-type').value=filingType(chosen)==='other'?'all':filingType(chosen);$('reader-year').value='';}selectFiling(chosen);}else{renderFilingList();clearDocument(requested?'The linked report is not in the available list. Choose another filing.':'Choose a report type or year with available filings.');}
    }catch(e){if(seq!==r.seq)return;status('reader-status',message(e),true);$('reader-list').replaceChildren(node('p','Use Open filings to retry.','reader-status'));clearDocument('The filing list could not load.');}
  }
  async function loadSection(){
    const r=state.reader;if(!r.selected)return;r.textController?.abort();const controller=r.textController=new AbortController(),seq=++r.textSeq,f=r.selected,tk=r.ticker,section=r.section;
    clearDocument('Loading '+($('reader-tabs').querySelector('[data-section="'+section+'"]')?.textContent||'section')+'…');
    document.querySelectorAll('#reader-tabs button').forEach(b=>{b.setAttribute('aria-selected',String(b.dataset.section===section));b.tabIndex=b.dataset.section===section?0:-1;});
    const meta=$('reader-meta');meta.replaceChildren(document.createTextNode(tk+' · '+f.form_type+' · filed '+f.filing_date));const url=safeUrl(f.edgar_url);if(url){const a=node('a','Open original SEC filing');a.href=url;a.target='_blank';a.rel='noopener noreferrer';meta.append(a);}
    updateURL({section,accession:f.accession});
    try{const key=[tk,f.accession,section].join('|');let d=textCache.get(key);if(!d){d=await request('/api/filing-text/'+encodeURIComponent(tk)+'/'+section+'?'+new URLSearchParams({accession:f.accession,form_type:f.form_type}),{signal:controller.signal});if(d.text)textCache.set(key,d);}if(seq!==r.textSeq)return;r.raw=typeof d.text==='string'?d.text:'';if(!r.raw.trim()){clearDocument('This section is not available in the extracted text. Try another section or open the original SEC filing above.');return;}renderDocument();if(d.text_length>r.raw.length||section==='full')meta.append(node('span','Extracted text; use the original filing for complete tables and exhibits.'));}
    catch(e){if(seq!==r.textSeq)return;clearDocument(message(e)+' Select the section again to retry, or open the original filing.');}
  }
  function renderDocument(){
    const r=state.reader;if(!r.raw)return;const doc=$('reader-doc'),query=$('reader-find').value.trim(),lower=query.toLocaleLowerCase();r.matches=[];r.match=-1;
    const paragraphs=r.raw.split(/\n\s*\n/).map(t=>t.trim()).filter(t=>t.length>1&&!/^(<\?xml|xml version|XBRL Document Created|Copyright \d{4} Workiva|r:[0-9a-f-]{8,})/i.test(t));
    const fragment=document.createDocumentFragment();
    for(const text of paragraphs){const p=node('p');if(text.length<100&&!/[.!?]$/.test(text))p.className='reader-subheading';let offset=0,index=-1;
      while(lower&&(index=text.toLocaleLowerCase().indexOf(lower,offset))!==-1){p.append(document.createTextNode(text.slice(offset,index)));const mark=node('mark',text.slice(index,index+query.length));r.matches.push(mark);p.append(mark);offset=index+query.length;}
      p.append(document.createTextNode(text.slice(offset)));fragment.append(p);
    }
    doc.replaceChildren(fragment);$('reader-matches').textContent=query?r.matches.length+' matches':'';$('reader-prev').disabled=$('reader-next').disabled=!r.matches.length;if(r.matches.length)moveMatch(1);
  }
  function moveMatch(direction){const r=state.reader;if(!r.matches.length)return;r.matches[r.match]?.classList.remove('active');r.match=(r.match+direction+r.matches.length)%r.matches.length;const m=r.matches[r.match];m.classList.add('active');m.scrollIntoView({block:'center',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});$('reader-matches').textContent=(r.match+1)+' / '+r.matches.length;}

  function screenFilters(){const f={};for(const [id,key] of [['screen-revenue','revenue_min'],['screen-gross','gross_margin_min'],['screen-margin','net_margin_min'],['screen-debt','de_ratio_max']])if($(id).value!=='')f[key]=Number($(id).value);if($('screen-sector').value)f.sector=$('screen-sector').value;return f;}
  async function loadSectors(){if(state.screen.sectorsLoaded)return;try{const d=await request('/api/screener/sectors');if(!Array.isArray(d.sectors))return;const current=$('screen-sector').value;for(const s of d.sectors)if(typeof s==='string'&&!Array.from($('screen-sector').options).some(o=>o.value===s))$('screen-sector').append(new Option(s.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase()),s));$('screen-sector').value=current;state.screen.sectorsLoaded=true;}catch{} }
  function screenSelection(){const n=state.screen.selection.size;$('screen-compare').disabled=n<2;$('screen-compare').textContent='Compare selected ('+n+')';}
  function renderScreen(){
    const s=state.screen,key=$('screen-sort').value,rows=[...s.rows].sort((a,b)=>key==='ticker'?a.ticker.localeCompare(b.ticker):a[key]==null?(b[key]==null?0:1):b[key]==null?-1:(key==='de_ratio'?1:-1)*(a[key]-b[key]));
    if(!rows.length){empty($('screen-rows'),s.ran?'No companies match these filters':'Build your shortlist',s.ran?'Try a lower minimum, a higher debt limit, or another sector.':'Choose a screen above. Select up to eight results to compare.');return;}
    $('screen-rows').replaceChildren(...rows.map(r=>{const tr=node('tr'),name=node('td'),check=document.createElement('input');check.type='checkbox';check.checked=s.selection.has(r.ticker);check.setAttribute('aria-label','Compare '+r.ticker);check.onchange=()=>{if(check.checked&&s.selection.size>=8){check.checked=false;status('screen-status','Select up to eight companies for a comparison.',true);return;}check.checked?s.selection.add(r.ticker):s.selection.delete(r.ticker);screenSelection();};const a=node('a',r.ticker);a.href=companyHref(r.ticker);name.append(check,a,node('small',r.company||r.ticker));tr.append(name);for(const value of [(r.sector||'').replace(/_/g,' '),compact(r.revenue),compact(r.net_income),percent(r.gross_margin),percent(r.net_margin),r.de_ratio==null?'—':r.de_ratio.toFixed(2)+'×'])tr.append(node('td',value));return tr;}));
  }
  function screenBusy(busy){$('screen-run').disabled=busy;$('screen-stop').hidden=!busy;$('screen-run').textContent=busy?'Screening…':'Run screen';}
  async function runScreen(){
    const s=state.screen;s.controller?.abort();const controller=s.controller=new AbortController(),seq=++s.seq,filters=screenFilters();s.rows=[];s.selection.clear();s.ran=true;screenSelection();screenBusy(true);empty($('screen-rows'),'Looking through the covered companies','A single sector usually loads faster than the full universe.');$('screen-result-title').textContent='Results';status('screen-status','Applying your filters to cached annual filings…');
    const slow=setTimeout(()=>{if(seq===s.seq)status('screen-status','Still screening the covered universe. You can stop and choose a single sector for a smaller search.');},10000);
    try{const d=await request('/api/screener',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filters,limit:50}),signal:controller.signal});if(seq!==s.seq)return;const rows=Array.isArray(d.matches)?d.matches:d.results;if(!Array.isArray(rows))throw new Error('The service did not return screening results.');const seen=new Set();s.rows=rows.filter(r=>validTicker(r.ticker||'')&&!seen.has(r.ticker)&&seen.add(r.ticker)).map(r=>({...r,...Object.fromEntries(['revenue','net_income','gross_margin','net_margin','de_ratio'].map(k=>[k,finite(r[k])]))}));renderScreen();$('screen-result-title').textContent=s.rows.length+' companies returned';status('screen-status',(s.rows.length>=50?'Showing up to 50 matches. Narrow the filters to refine the list.':'Screen complete.')+(d.cached_tickers!=null?' '+d.cached_tickers+' companies have cached data across the covered universe.':''));}
    catch(e){if(seq!==s.seq)return;status('screen-status',message(e),true);empty($('screen-rows'),'The screen could not finish','Try a single sector or run the screen again.');}
    finally{clearTimeout(slow);if(seq===s.seq){s.controller=null;screenBusy(false);}}
  }
  function stopScreen(){const s=state.screen;s.seq++;s.controller?.abort();s.controller=null;screenBusy(false);status('screen-status','Screen stopped. Adjust your filters and run it again.');empty($('screen-rows'),'Screen stopped','Choose a sector for a smaller search.');}
  function preset(name){$('screen-form').reset();const config={tech:{sector:'mega_tech'},chips:{sector:'semiconductors'},margin:{revenue:'10000000000',margin:'0.2'}}[name]||{};for(const [key,value] of Object.entries(config)){const el=$('screen-'+key);if(el.tagName==='SELECT'&&!Array.from(el.options).some(o=>o.value===value))el.append(new Option(value.replace(/_/g,' '),value));el.value=value;}document.querySelectorAll('[data-screen-preset]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.screenPreset===name)));runScreen();}
  function init(){
    if(state.initialized)return;state.initialized=true;
    const list=document.createElement('datalist');list.id='tool-companies';for(const [ticker,name] of Object.entries(companies)){const o=node('option');o.value=ticker;o.label=name;list.append(o);}document.body.append(list);
    let searchSeq=0,searchTimer;for(const id of ['compare-add','reader-company'])$(id).addEventListener('input',()=>{clearTimeout(searchTimer);const seq=++searchSeq,q=$(id).value.trim();if(q.length<2)return;searchTimer=setTimeout(async()=>{try{const d=await request('/api/search?q='+encodeURIComponent(q));if(seq!==searchSeq)return;for(const r of d.results||[])if(validTicker(r.ticker||'')&&!Array.from(list.options).some(o=>o.value===r.ticker)){const o=node('option');o.value=r.ticker;o.label=r.name||r.ticker;list.append(o);companies[r.ticker]=r.name||r.ticker;}}catch{}},220);});
    const c=state.compare;c.mode=initial.period==='quarterly'?'quarter':'annual';c.year=/^\d{4}$/.test(initial.year||'')?initial.year:'';c.quarter=/^Q[1-4]$/.test(initial.quarter||'')?initial.quarter:'Q4';$('compare-mode').value=c.mode;$('compare-quarter').value=c.quarter;compareYears();
    c.tickers=(params.get('tickers')||initial.ticker||'MSFT,GOOGL,META').split(',').map(s=>s.trim().toUpperCase()).filter(validTicker).slice(0,8);c.tickers=[...new Set(c.tickers)];
    $('compare-form').onsubmit=e=>{e.preventDefault();const raw=$('compare-add').value.trim();if(!raw)return;if(setTickers([...c.tickers,...raw.split(/[\s,]+/)]))$('compare-add').value='';};
    for(const id of ['compare-mode','compare-year','compare-quarter'])$(id).onchange=loadComparison;
    document.querySelectorAll('[data-peer-group]').forEach(b=>b.onclick=()=>setTickers(b.dataset.peerGroup.split(',')));$('compare-peers').onclick=findPeers;
    document.querySelectorAll('#compare-table [data-sort] button').forEach(b=>b.onclick=()=>{const key=b.parentNode.dataset.sort;c.direction=c.sort===key?-c.direction:-1;c.sort=key;renderComparison();});
    const r=state.reader;$('reader-company').value=validTicker(initial.ticker||'')?initial.ticker:'META';$('reader-type').value=['annual','quarter','all'].includes(params.get('form'))?params.get('form'):initial.period==='quarterly'?'quarter':'annual';r.section=Array.from(document.querySelectorAll('#reader-tabs [data-section]')).some(b=>b.dataset.section===params.get('section'))?params.get('section'):'business';
    $('reader-form').onsubmit=e=>{e.preventDefault();loadReader();};$('reader-type').onchange=$('reader-year').onchange=filterFilings;
    const tabs=[...document.querySelectorAll('#reader-tabs button')];tabs.forEach((b,i)=>{b.onclick=()=>{r.section=b.dataset.section;loadSection();};b.onkeydown=e=>{if(!['ArrowRight','ArrowLeft','Home','End'].includes(e.key))return;e.preventDefault();const n=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[n].focus();tabs[n].click();};});
    let findTimer;$('reader-find').oninput=()=>{clearTimeout(findTimer);findTimer=setTimeout(renderDocument,120);};$('reader-find').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();moveMatch(e.shiftKey?-1:1);}if(e.key==='Escape'){$('reader-find').value='';renderDocument();}};$('reader-prev').onclick=()=>moveMatch(-1);$('reader-next').onclick=()=>moveMatch(1);
    $('screen-form').onsubmit=e=>{e.preventDefault();runScreen();};$('screen-form').onchange=()=>{document.querySelectorAll('[data-screen-preset]').forEach(b=>b.setAttribute('aria-pressed','false'));status('screen-status','Filters changed. Run the screen to apply them.');};$('screen-sort').onchange=renderScreen;$('screen-stop').onclick=stopScreen;
    $('screen-reset').onclick=()=>{if(state.screen.controller)stopScreen();$('screen-form').reset();state.screen.rows=[];state.screen.selection.clear();state.screen.ran=false;renderScreen();screenSelection();$('screen-result-title').textContent='Results';status('screen-status','Filters reset. Choose a preset or run a new screen.');document.querySelectorAll('[data-screen-preset]').forEach(b=>b.setAttribute('aria-pressed','false'));};
    document.querySelectorAll('[data-screen-preset]').forEach(b=>b.onclick=()=>preset(b.dataset.screenPreset));$('screen-compare').onclick=()=>{location.href='/?'+new URLSearchParams({view:'comps',tickers:[...state.screen.selection].join(','),period:'annual'});};
  }
  function activate(view){if(!['comps','filings','screener'].includes(view))return false;init();state.view=view;document.body.dataset.tool=view;updateURL({view});if(view==='comps'){renderChips();loadComparison();}if(view==='filings'&&!state.reader.started)loadReader();if(view==='screener')loadSectors();return true;}
  window.FineasTools={activate};
})();
