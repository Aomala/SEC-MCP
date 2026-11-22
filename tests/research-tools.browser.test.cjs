const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {JSDOM}=require('jsdom');
const root=path.join(__dirname,'../src/sec_mcp/static');
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
const script=name=>fs.readFileSync(path.join(root,name),'utf8');
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const json=(data,status=200)=>new Response(JSON.stringify(data),{status,headers:{'content-type':'application/json'}});
function history(mode='annual',ticker='META'){
 const entries=mode==='quarter'?[['2026','Q1','2026-03-31',100],['2025','Q1','2025-03-31',80],['2025','Q4','2025-12-31',90]]:[['2025','FY','2025-12-31',400],['2024','FY','2024-12-31',300]];
 const income=entries.map(([fiscalYear,period,date,revenue])=>({fiscalYear,period,date,revenue,netIncome:0,grossProfit:50,epsDiluted:0,reportedCurrency:ticker==='SAP'?'EUR':'USD',_meta:{accession:fiscalYear==='2024'?'0001326801-25-000017':'0001628280-26-003942',formType:'10-K',quality:'standalone'}}));
 return {income,cashflow:income.map(r=>({...r,operatingCashFlow:30,capitalExpenditure:-10}))};
}
const filings=[{accession:'0001628280-26-003942',form_type:'10-K',filing_date:'2026-01-29',edgar_url:'https://www.sec.gov/Archives/current'},{accession:'0001326801-25-000017',form_type:'10-K',filing_date:'2025-01-30',edgar_url:'https://www.sec.gov/Archives/prior'},{accession:'0001628280-26-028526',form_type:'10-Q',filing_date:'2026-04-30',edgar_url:'https://www.sec.gov/Archives/quarter'}];
async function setup(t,{url='https://fineasmcp.vercel.app/?view=comps&tickers=META,MSFT&period=annual',respond,saved}={}){
 const dom=new JSDOM(html,{url,runScripts:'outside-only'}),w=dom.window;t.after(()=>w.close());
 w.HTMLElement.prototype.scrollIntoView=()=>{};w.matchMedia=()=>({matches:true});w.lucide={createIcons(){}};w.Request=Request;w.Response=Response;w.Headers=Headers;w.AbortController=AbortController;w.AbortSignal=AbortSignal;w.FineasAccount={apiBase:'https://sec-mcp-production.up.railway.app'};
 if(saved)w.localStorage.setItem('fineas_explorer_selection_v1',JSON.stringify(saved));w.localStorage.setItem('fineas-tos-accepted','1');
 const requests=[];
 w.fetch=async(input,init)=>{if(input instanceof Request)init={method:input.method,signal:input.signal,body:input.method==='GET'?undefined:await input.clone().text()};const u=new URL(input instanceof Request?input.url:input);requests.push({u,init});const custom=respond?.(u,init);if(custom!==undefined)return custom;
 if(u.pathname.includes('/financials-history/'))return json(history(u.searchParams.get('period'),u.pathname.split('/').at(-1)));
 if(u.pathname.includes('/api/filings/'))return json({filings});
 if(u.pathname.includes('/filing-text/'))return json({text:'Business\n\nRevenue is growing. Revenue is reported here. <script>alert(1)</script> & debt.',text_length:100});
 if(u.pathname==='/api/peers/META')return json({peers:['MSFT','GOOGL']});
 if(u.pathname==='/api/screener/sectors')return json({sectors:['mega_tech','semiconductors']});
 if(u.pathname==='/api/screener')return json({matches:[{ticker:'META',company:'Meta',sector:'mega_tech',revenue:400,net_income:0,gross_margin:0.5,net_margin:0,de_ratio:0},{ticker:'MSFT',company:'Microsoft',sector:'mega_tech',revenue:500,net_income:50,gross_margin:0.6,net_margin:0.1,de_ratio:null}],cached_tickers:754});
 return json({});};
 w.eval(['workspace-shell.js','app.js','research-tools.js'].map(script).join('\n;\n'));
 await tick();await tick();return {w,dom,doc:w.document,requests};
}
function change(ctx,id,value){const el=ctx.doc.getElementById(id);el.value=value;el.dispatchEvent(new ctx.w.Event('change',{bubbles:true}));}
function submit(ctx,id){ctx.doc.getElementById(id).dispatchEvent(new ctx.w.Event('submit',{cancelable:true}));}
function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return{promise,resolve:data=>resolve(json(data))};}

test('deep links open the requested tool with shared navigation and explicit periods override saved selection',async t=>{
 const c=await setup(t,{saved:{ticker:'AAPL',year:'2024',quarter:'Q4',period:'quarterly'}});
 assert.equal(c.doc.body.dataset.tool,'comps');assert.equal(c.doc.getElementById('panel-comps').style.display,'block');assert.equal(c.doc.getElementById('compare-year').value,'');
 assert.equal(c.doc.querySelector('[data-nav-tool="comps"]').getAttribute('aria-current'),'page');assert.equal(c.doc.querySelectorAll('.fineas-topbar [data-nav-tool]').length,4);
 assert.equal(c.doc.querySelectorAll('#compare-rows tr').length,2);assert.match(c.doc.getElementById('compare-rows').textContent,/0.0%/);assert.match(c.doc.getElementById('compare-rows').textContent,/\$20/);
});

test('comparison renders fast rows immediately and labels currencies without normalizing unlike amounts',async t=>{
 const slow=deferred();const c=await setup(t,{url:'https://fineasmcp.vercel.app/?view=comps&tickers=META,SAP',respond:u=>u.pathname.endsWith('/SAP')?slow.promise:undefined});
 assert.match(c.doc.getElementById('compare-rows').textContent,/\$400/);assert.match(c.doc.getElementById('compare-rows').textContent,/Loading financials/);
 slow.resolve(history('annual','SAP'));await tick();await tick();assert.match(c.doc.getElementById('compare-summary').textContent,/Mixed currencies/);assert.doesNotMatch(c.doc.getElementById('compare-summary').textContent,/Largest reported revenue/);assert.match(c.doc.getElementById('compare-rows').textContent,/EUR/);
});

test('changing period requests quarterly history and never substitutes another fiscal year',async t=>{
 const c=await setup(t,{url:'https://fineasmcp.vercel.app/?view=comps&tickers=META&period=annual&year=2024'});
 assert.match(c.doc.getElementById('compare-rows').textContent,/\$300/);
 change(c,'compare-mode','quarter');await tick();await tick();assert.match(c.doc.getElementById('compare-rows').textContent,/fiscal period is unavailable/);assert.doesNotMatch(c.doc.getElementById('compare-rows').textContent,/\$100/);
 change(c,'compare-year','2026');change(c,'compare-quarter','Q1');await tick();await tick();assert.match(c.doc.getElementById('compare-rows').textContent,/Q1 FY 2026/);assert.match(c.doc.getElementById('compare-rows').textContent,/\$100/);assert(c.requests.some(r=>r.u.searchParams.get('period')==='quarter'));
});

test('removing a company during a request prevents its late row from returning',async t=>{
 const slow=deferred();const c=await setup(t,{respond:u=>u.pathname.endsWith('/MSFT')?slow.promise:undefined});c.doc.querySelector('[aria-label="Remove MSFT"]').click();await tick();slow.resolve(history());await tick();assert.doesNotMatch(c.doc.getElementById('compare-rows').textContent,/MSFT/);assert.equal(c.doc.querySelectorAll('#compare-rows tr').length,1);
});

test('comparison excludes year-to-date rows and keeps absent cash flow unknown',async t=>{
 const h=history('quarter');h.income[0]._meta.isYtd=true;h.cashflow=[];
 const c=await setup(t,{url:'https://fineasmcp.vercel.app/?view=comps&tickers=META&period=quarterly&quarter=Q1',respond:u=>u.pathname.includes('/financials-history/')?json(h):undefined});
 assert.match(c.doc.getElementById('compare-rows').textContent,/Q1 FY 2025/);assert.doesNotMatch(c.doc.getElementById('compare-rows').textContent,/Q1 FY 2026/);assert.equal(c.doc.querySelector('#compare-rows tr').children[5].textContent,'—');
});

test('reader opens the fiscal period’s accession, supports filing-year filtering, and searches literal text safely',async t=>{
 const c=await setup(t,{url:'https://fineasmcp.vercel.app/?view=filings&ticker=META&period=annual&year=2024'});await tick();
 assert.match(c.doc.getElementById('reader-meta').textContent,/2025-01-30/);assert.match(c.requests.find(r=>r.u.pathname.includes('/filing-text/')).u.search,/0001326801-25-000017/);
 c.doc.getElementById('reader-find').value='<script>';c.doc.getElementById('reader-find').dispatchEvent(new c.w.Event('input'));await new Promise(r=>setTimeout(r,160));
 assert.equal(c.doc.querySelectorAll('#reader-doc script').length,0);assert.equal(c.doc.querySelector('#reader-doc mark').textContent,'<script>');assert.match(c.doc.getElementById('reader-matches').textContent,/1 \/ 1/);
 change(c,'reader-year','2026');await tick();await tick();assert.match(c.doc.getElementById('reader-meta').textContent,/2026-01-29/);assert.equal(c.doc.querySelectorAll('#reader-list button').length,1);
});

test('reader ignores a stale section response after the user changes sections',async t=>{
 const slow=deferred();const c=await setup(t,{url:'https://fineasmcp.vercel.app/?view=filings&ticker=META',respond:u=>u.pathname.endsWith('/business')?slow.promise:undefined});
 c.doc.querySelector('[data-section="risk_factors"]').click();await tick();await tick();slow.resolve({text:'Obsolete business response'});await tick();assert.doesNotMatch(c.doc.getElementById('reader-doc').textContent,/Obsolete/);assert.match(new URL(c.w.location.href).search,/section=risk_factors/);
});

test('empty filing filters clear the previous document and keep source navigation safe',async t=>{
 const c=await setup(t,{url:'https://fineasmcp.vercel.app/?view=filings&ticker=META',respond:u=>u.pathname.includes('/api/filings/')?json({filings:filings.filter(f=>f.form_type==='10-K').map(f=>({...f,edgar_url:'javascript:alert(1)'}))}):undefined});await tick();assert.equal(c.doc.querySelector('#reader-meta a'),null);change(c,'reader-type','quarter');assert.match(c.doc.getElementById('reader-doc').textContent,/No report matches/);assert.doesNotMatch(c.doc.getElementById('reader-doc').textContent,/Revenue is/);
});

test('screener consumes matches, preserves zero values and can shortlist companies',async t=>{
 const c=await setup(t,{url:'https://fineasmcp.vercel.app/?view=screener'});c.doc.querySelector('[data-screen-preset="tech"]').click();await tick();await tick();
 const req=c.requests.find(r=>r.u.pathname==='/api/screener');assert.equal(JSON.parse(req.init.body).filters.sector,'mega_tech');assert.equal(c.doc.querySelectorAll('#screen-rows tr').length,2);assert.match(c.doc.getElementById('screen-rows').textContent,/0.00×/);assert.match(c.doc.getElementById('screen-status').textContent,/754/);
 for(const input of c.doc.querySelectorAll('#screen-rows input')){input.checked=true;input.dispatchEvent(new c.w.Event('change'));}assert.equal(c.doc.getElementById('screen-compare').disabled,false);assert.match(c.doc.getElementById('screen-compare').textContent,/2/);assert.match(c.doc.querySelector('#screen-rows a').href,/explorer\?ticker=MSFT/);
});

test('screener errors are not shown as an empty successful result',async t=>{
 const c=await setup(t,{url:'https://fineasmcp.vercel.app/?view=screener',respond:u=>u.pathname==='/api/screener'?new Response('The page could not be found',{status:502}):undefined});submit(c,'screen-form');await tick();await tick();assert.match(c.doc.getElementById('screen-status').textContent,/HTTP 502/);assert.match(c.doc.getElementById('screen-rows').textContent,/could not finish/);assert.doesNotMatch(c.doc.getElementById('screen-rows').textContent,/No companies match/);
});

test('screener cancellation prevents old results from overwriting a newer screen',async t=>{
 const slow=deferred();let calls=0;const c=await setup(t,{url:'https://fineasmcp.vercel.app/?view=screener',respond:u=>u.pathname==='/api/screener'&&++calls===1?slow.promise:undefined});submit(c,'screen-form');await tick();c.doc.getElementById('screen-stop').click();c.doc.querySelector('[data-screen-preset="tech"]').click();await tick();await tick();slow.resolve({matches:[{ticker:'OLD',revenue:1}]});await tick();assert.doesNotMatch(c.doc.getElementById('screen-rows').textContent,/OLD/);assert.match(c.doc.getElementById('screen-rows').textContent,/META/);
});

test('all requested tools open through existing navigation without invoking the old empty-data renderers',async t=>{
 const c=await setup(t);c.doc.querySelector('.nav-item[data-view="filings"]').click();await tick();await tick();assert.equal(c.doc.body.dataset.tool,'filings');assert.match(c.doc.getElementById('reader-doc').textContent,/Revenue/);c.doc.querySelector('.nav-item[data-view="screener"]').click();await tick();assert.equal(c.doc.body.dataset.tool,'screener');assert.match(c.w.location.search,/view=screener/);assert.equal(c.doc.querySelector('[data-nav-tool="screener"]').getAttribute('aria-current'),'page');
});

test('quick quarter changes share an in-flight history request and render the latest selection',async t=>{
 const slow=deferred();const c=await setup(t,{url:'https://fineasmcp.vercel.app/?view=comps&tickers=META&period=annual&year=2025',respond:u=>u.searchParams.get('period')==='quarter'?slow.promise:undefined});
 change(c,'compare-mode','quarter');change(c,'compare-quarter','Q1');change(c,'compare-quarter','Q4');
 assert.equal(c.requests.filter(r=>r.u.searchParams.get('period')==='quarter').length,1);
 slow.resolve(history('quarter'));await tick();await tick();assert.match(c.doc.getElementById('compare-rows').textContent,/Q4 FY 2025/);assert.match(c.doc.getElementById('compare-rows').textContent,/\$90/);assert.doesNotMatch(c.doc.getElementById('compare-rows').textContent,/Q1 FY/);
});

test('failed comparison requests can retry without reloading successful companies',async t=>{
 let calls=0;const c=await setup(t,{respond:u=>u.pathname.endsWith('/MSFT')&&++calls===1?Promise.reject(new TypeError('Failed to fetch')):undefined});
 assert.match(c.doc.getElementById('compare-rows').textContent,/\$400/);c.doc.querySelector('[aria-label="Retry MSFT"]').click();await tick();await tick();
 assert.equal(c.requests.filter(r=>r.u.pathname.endsWith('/META')).length,1);assert.equal(c.requests.filter(r=>r.u.pathname.endsWith('/MSFT')).length,2);assert.equal(c.doc.querySelector('[aria-label="Retry MSFT"]'),null);assert.match(c.doc.getElementById('compare-summary').textContent,/2 of 2/);
});
