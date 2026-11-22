const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { JSDOM } = require('jsdom');

// NODE_PATH=/path/to/node_modules node --test tests/explorer.browser.test.cjs
const root = path.join(__dirname, '../src/sec_mcp/static/');
const html = fs.readFileSync(root + 'explorer.html', 'utf8');
const api = 'https://sec-mcp-production.up.railway.app';
const overview = {
  quote: { price: 123.45, session: 'closed' },
  latest: { formType: '10-Q', fiscalPeriod: 'Q3', fiscalYear: '2026', currency: 'USD',
    metrics: { revenue: 1000000000, netMargin: 0.2, freeCashFlow: 250000000 } },
};
function historyData(mode = 'quarter') {
  const entries = mode === 'annual'
    ? [['2025-09-27', '2025', 'FY', 4000000000], ['2024-09-28', '2024', 'FY', 3000000000]]
    : [['2026-06-27', '2026', 'Q3', 1000000000], ['2026-03-28', '2026', 'Q2', 900000000],
       ['2025-06-28', '2025', 'Q3', 800000000], ['2025-03-29', '2025', 'Q2', 600000000]];
  const income = entries.map(([date, fiscalYear, period, revenue]) => ({
    date, fiscalYear, period, revenue, cik: 1326801, grossProfit: revenue * .5, operatingIncome: revenue * .3,
    netIncome: revenue * .2, reportedCurrency: 'USD',
    _meta: { formType: mode === 'annual' ? '10-K' : '10-Q', accession: '0000000000-26-000001', quality: 'standalone' },
  }));
  return { income, cashflow: income.map(row => ({ ...row, operatingCashFlow: row.revenue * .3,
    capitalExpenditure: -row.revenue * .05 })), source: 'sec', period: mode };
}
async function setup(options = {}) {
  const dom = new JSDOM(html, {
    url: options.url || 'https://fineasmcp.vercel.app/explorer?ticker=AAPL&period=quarterly',
    runScripts: 'outside-only',
  });
  const w = dom.window;
  if (options.savedSelection) w.localStorage.setItem('fineas_explorer_selection_v1', JSON.stringify(options.savedSelection));
  const calls = [], requests = [], charts = new Map(), chartEvents = new Map();
  w.matchMedia = () => ({ matches: options.reducedMotion !== false });
  w.HTMLElement.prototype.scrollIntoView = () => {};
  w.echarts = { init: el => ({
    setOption: data => charts.set(el.id, data), clear: () => charts.delete(el.id), resize() {},
    on: (name, handler) => chartEvents.set(el.id + ':' + name, handler), off: name => chartEvents.delete(el.id + ':' + name),
  }) };
  // Account access is covered separately; retain its API-origin contract here.
  w.FineasAccount = { apiBase: /vercel\.app$/.test(w.location.hostname) ? api : '' };
  w.fetch = async (input, init) => {
    const url = new URL(input, w.location.href);
    calls.push(url);
    requests.push({ url, signal: init?.signal, init });
    if (options.respond) return options.respond(url, init);
    if (url.hostname === 'gemini-search-api-production.up.railway.app') return new Response(JSON.stringify(options.answer || { answer: 'Revenue rose **25%** year over year.', urls: [{ title: 'Company filing', url: 'https://www.sec.gov/example' }] }));
    const data = url.pathname === '/api/search' ? { results: options.searchResults || [] }
      : url.pathname.includes('/overview/') ? options.overview || overview
      : options.history || historyData(url.searchParams.get('period'));
    return new Response(JSON.stringify(data), { headers: { 'Content-Type': 'application/json' } });
  };
  for (const script of w.document.querySelectorAll('script:not([src])')) {
    // Exercise the prepared integration separately from the production outage state.
    w.eval(options.researchEnabled ? script.textContent.replace('const RESEARCH_ENABLED = false;', 'const RESEARCH_ENABLED = true;') : script.textContent);
  }
  await new Promise(resolve => setImmediate(resolve));
  return { dom, w, calls, requests, charts, chartEvents, status: () => w.document.getElementById('status').textContent };
}

const settle = () => new Promise(resolve => setImmediate(resolve));
function selectPeriod(ctx, id, value) {
  const select = ctx.w.document.getElementById(id);
  select.value = value;
  select.dispatchEvent(new ctx.w.Event('change', { bubbles: true }));
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve: data => resolve(new Response(JSON.stringify(data))), reject };
}

test('annual to quarterly switches the data requests and remains on the company Explorer', async t => {
  const ctx = await setup({ url: 'https://fineasmcp.vercel.app/explorer?ticker=META&period=annual&year=2025' });
  t.after(() => ctx.dom.window.close());
  selectPeriod(ctx, 'period', 'quarter'); await settle();
  assert.equal(ctx.w.document.getElementById('kpi-rev').textContent, '$800.0M');
  assert.equal(ctx.w.document.getElementById('fiscal-year').value, '2025');
  assert.equal(ctx.calls.at(-1).searchParams.get('period'), 'quarter');
  assert.equal(ctx.calls.at(-2).searchParams.get('period'), 'quarterly');
  assert.equal(new URL(ctx.w.location.href).pathname, '/explorer');
  assert.equal(new URL(ctx.w.location.href).searchParams.get('quarter'), 'Q3');
  assert.equal(ctx.w.document.getElementById('chat-context').textContent, 'META · Q3 FY 2025');
});

test('landing-page Explore link resumes the last selection while explicit links take precedence', async t => {
  const savedSelection = {ticker:'META',period:'quarterly',year:'2025',quarter:'Q2'};
  const ctx = await setup({url:'https://fineasmcp.vercel.app/explorer',savedSelection});
  t.after(() => ctx.dom.window.close());
  assert.equal(ctx.w.document.getElementById('ticker').value,'META');
  assert.equal(ctx.w.document.getElementById('kpi-rev').textContent,'$600.0M');
  assert.equal(ctx.w.document.getElementById('fiscal-quarter').value,'Q2');
  const explicit = await setup({url:'https://fineasmcp.vercel.app/explorer?ticker=MSFT&period=annual&year=2024',savedSelection});
  t.after(() => explicit.dom.window.close());
  assert.equal(explicit.w.document.getElementById('ticker').value,'MSFT');
  assert.equal(explicit.w.document.getElementById('period').value,'annual');
  assert.equal(explicit.w.document.getElementById('kpi-rev').textContent,'$3.0B');
});

test('financial table, chart clicks and period arrows share one selected period', async t => {
  const ctx = await setup(); t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document;
  doc.getElementById('tab-financials').click();
  assert.equal(doc.getElementById('view-financials').hidden, false);
  assert.equal(doc.getElementById('view-overview').hidden, true);
  [...doc.querySelectorAll('.financial-table button')].find(button => button.textContent === 'Q2 FY 2025').click();
  assert.equal(doc.getElementById('kpi-rev').textContent, '$600.0M');
  assert.equal(doc.getElementById('previous-period').disabled, true);
  doc.getElementById('next-period').click();
  assert.equal(doc.getElementById('kpi-rev').textContent, '$800.0M');
  assert.equal(doc.querySelector('.financial-table [aria-pressed="true"]').textContent, 'Q3 FY 2025');
  ctx.chartEvents.get('c-trend:click')({dataIndex:0});
  assert.equal(doc.getElementById('fiscal-quarter').value, 'Q2');
  assert.equal(doc.getElementById('fiscal-year').value, '2025');
  doc.getElementById('tab-overview').dispatchEvent(new ctx.w.KeyboardEvent('keydown',{key:'ArrowRight'}));
  assert.equal(doc.activeElement.id, 'tab-financials');
});

test('cash flow view retains missing values and historical breakdowns have no blank map space', async t => {
  const history = historyData(); history.cashflow = [];
  const ctx = await setup({history}); t.after(() => ctx.dom.window.close());
  selectPeriod(ctx, 'trend-metric', 'cashflow');
  assert(ctx.charts.get('c-trend').series.every(series => series.data.every(value => value === null)));
  assert.equal(ctx.w.document.getElementById('c-geo').hidden, true);
  assert.equal(ctx.w.document.getElementById('c-seg').hidden, true);
  assert.equal(ctx.w.document.querySelectorAll('#financial-table tbody tr').length, 8);
});

async function ask(ctx, question) {
  const doc = ctx.w.document;
  doc.getElementById('chat-input').value = question;
  doc.getElementById('chat-form').dispatchEvent(new ctx.w.Event('submit',{cancelable:true}));
  await settle();
}
test('the live outage state keeps financials usable without sending requests to the broken AI service', async t => {
  const ctx = await setup(); t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document;
  assert.match(doc.getElementById('chat-messages').textContent, /AI research is temporarily unavailable/);
  assert.equal(doc.getElementById('chat-form').hidden, true);
  assert.equal(doc.getElementById('chat-send').disabled, true);
  await ask(ctx, 'Explain these figures');
  assert.equal(ctx.calls.length, 2);
  selectPeriod(ctx, 'fiscal-quarter', 'Q2');
  assert.equal(doc.getElementById('kpi-rev').textContent, '$900.0M');
  assert.match(doc.getElementById('chat-messages').textContent, /AI research is temporarily unavailable/);
});
test('chat uses the existing Railway service, selected quarter, filing and full active conversation', async t => {
  const ctx = await setup({ researchEnabled: true, url:'https://fineasmcp.vercel.app/explorer?ticker=META&period=quarterly&year=2025&quarter=Q2' });
  t.after(() => ctx.dom.window.close());
  await ask(ctx, 'Explain this revenue.');
  const request = ctx.requests.at(-1);
  assert.equal(request.url.href, 'https://gemini-search-api-production.up.railway.app/search');
  assert.equal(request.init.method,'POST');
  assert.equal(request.init.credentials,'omit');
  const query = JSON.parse(request.init.body).query;
  assert.match(query, /META, Q2 FY 2025/);
  assert.match(query, /"revenue":600000000/);
  assert.match(query, /"freeCashFlow":150000000/);
  assert.match(query, /0000000000-26-000001-index\.htm/);
  assert.equal(ctx.w.document.querySelector('.answer-text strong').textContent, '25%');
  assert.match(ctx.w.document.querySelector('.chat-sources').textContent, /Company filing/);
  await ask(ctx, 'And what about margins?');
  const followup = JSON.parse(ctx.requests.at(-1).init.body).query;
  assert.match(followup, /Explain this revenue/);
  assert.match(followup, /Revenue rose/);
  assert.match(followup, /And what about margins/);
});

test('switching periods aborts chat and ignores an old answer', async t => {
  const slow = deferred();
  const ctx = await setup({ researchEnabled: true, respond:(url) => url.hostname === 'gemini-search-api-production.up.railway.app'
    ? slow.promise : new Response(JSON.stringify(url.pathname.includes('/overview/') ? overview : historyData())) });
  t.after(() => ctx.dom.window.close());
  await ask(ctx, 'Explain Q3.');
  const chatRequest = ctx.requests.at(-1);
  assert.equal(ctx.w.document.getElementById('chat-stop').hidden, false);
  selectPeriod(ctx, 'fiscal-quarter', 'Q2');
  assert.equal(chatRequest.signal.aborted, true);
  slow.resolve({answer:'Obsolete Q3 answer',urls:[]}); await settle();
  assert.doesNotMatch(ctx.w.document.getElementById('chat-messages').textContent,/Obsolete|Explain Q3/);
  assert.match(ctx.w.document.getElementById('chat-context').textContent,/Q2/);
});

test('chat rejects unsafe source links, renders HTML as text and recovers from service failure', async t => {
  const ctx = await setup({researchEnabled: true, answer:{answer:'<img src=x onerror=alert(1)> **Safe text**',urls:[{url:'javascript:alert(1)'},{url:'https://trusted.example/source',title:'<img onerror=alert(1)>'}]}});
  t.after(() => ctx.dom.window.close());
  await ask(ctx,'Explain');
  assert.equal(ctx.w.document.querySelector('#chat-messages img'),null);
  assert.equal(ctx.w.document.querySelector('#chat-messages a[href^="javascript"]'),null);
  assert.equal(ctx.w.document.querySelector('.chat-sources a').href,'https://trusted.example/source');
  ctx.w.fetch = async () => new Response('Bad gateway',{status:502});
  await ask(ctx,'Another question');
  assert.match(ctx.w.document.getElementById('chat-messages').textContent,/research service is unavailable/);
  assert.equal(ctx.w.document.getElementById('chat-send').disabled,false);
  assert.equal(ctx.w.document.getElementById('chat-stop').hidden,true);
});

test('exact figures and period controls are usable while the slower overview is still pending', async t => {
  const slow = deferred();
  const ctx = await setup({ reducedMotion: false, respond: url => url.pathname.includes('/overview/')
    ? slow.promise : new Response(JSON.stringify(historyData())) });
  t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document;
  assert.equal(doc.getElementById('kpi-rev').textContent, '$1.0B');
  assert.equal(doc.getElementById('kpi-margin').textContent, '20.0%');
  assert.equal(doc.getElementById('kpi-fcf').textContent, '$250.0M');
  assert.equal(doc.getElementById('fiscal-year').disabled, false);
  assert.equal(doc.getElementById('fiscal-quarter').disabled, false);
  assert.equal(doc.getElementById('card-revenue').getAttribute('aria-busy'), 'false');
  assert.equal(doc.getElementById('card-price').getAttribute('aria-busy'), 'true');
  assert.equal(doc.getElementById('sec-geo').getAttribute('aria-busy'), 'true');
  assert.match(ctx.status(), /Figures ready/);
  assert.match(doc.getElementById('seg-list').textContent, /Loading breakdowns for Q3 FY 2026/);
  assert.match(doc.getElementById('prov').textContent, /Checking sources/);
  const trend = ctx.charts.get('c-trend');
  slow.resolve(overview); await settle();
  assert.equal(doc.getElementById('kpi-price').textContent, '$123.45');
  assert.equal(doc.getElementById('card-price').getAttribute('aria-busy'), 'false');
  assert.equal(doc.getElementById('sec-geo').getAttribute('aria-busy'), 'false');
  assert.doesNotMatch(ctx.status(), /loading/i);
  assert.equal(ctx.charts.get('c-trend'), trend, 'overview must not restart the chart');
});

test('a fast quote renders independently while financial history is still pending', async t => {
  const slow = deferred();
  const ctx = await setup({ respond: url => url.pathname.includes('/overview/')
    ? new Response(JSON.stringify(overview)) : slow.promise });
  t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document;
  assert.equal(doc.getElementById('kpi-price').textContent, '$123.45');
  assert.equal(doc.getElementById('card-price').getAttribute('aria-busy'), 'false');
  assert.equal(doc.getElementById('card-revenue').getAttribute('aria-busy'), 'true');
  assert.equal(doc.getElementById('kpi-rev').textContent, '—');
  assert.equal(doc.getElementById('fiscal-year').disabled, true);
  slow.resolve(historyData()); await settle();
  assert.equal(doc.getElementById('kpi-rev').textContent, '$1.0B');
  assert.equal(doc.getElementById('fiscal-year').disabled, false);
});

test('late overview data respects a period changed while it was loading', async t => {
  const slow = deferred();
  const ctx = await setup({ respond: url => url.pathname.includes('/overview/')
    ? slow.promise : new Response(JSON.stringify(historyData())) });
  t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document;
  selectPeriod(ctx, 'fiscal-year', '2025');
  selectPeriod(ctx, 'fiscal-quarter', 'Q2');
  const trend = ctx.charts.get('c-trend');
  slow.resolve({ ...overview,
    latest: { ...overview.latest, endDate: '2026-06-27' },
    segments: { sourceMeta: { reportDate: '2026-06-27', fiscalYear: '2026', fiscalPeriod: 'Q3', currency: 'USD' },
      product: [{ name: 'Latest product', value: 1000000000 }] } });
  await settle();
  assert.equal(doc.getElementById('fiscal-year').value, '2025');
  assert.equal(doc.getElementById('fiscal-quarter').value, 'Q2');
  assert.equal(doc.getElementById('kpi-rev').textContent, '$600.0M');
  assert.match(doc.getElementById('seg-list').textContent, /Q2 FY 2025.*not available/);
  assert.equal(doc.getElementById('t-seg').textContent, 'Revenue by Segment · Q2 FY 2025');
  assert.doesNotMatch(doc.getElementById('seg-list').textContent, /Latest product/);
  assert.equal(ctx.charts.get('c-trend'), trend);
  assert.match(ctx.status(), /Q2 2025/);
  assert.equal(new URL(ctx.w.location.href).searchParams.get('quarter'), 'Q2');
});

test('failed overview leaves financials and period selection available', async t => {
  const slow = deferred();
  const ctx = await setup({ respond: url => url.pathname.includes('/overview/')
    ? slow.promise : new Response(JSON.stringify(historyData())) });
  t.after(() => ctx.dom.window.close());
  slow.reject(new Error('Overview unavailable')); await settle();
  const doc = ctx.w.document;
  assert.equal(doc.getElementById('kpi-rev').textContent, '$1.0B');
  assert.match(doc.getElementById('sub-price').textContent, /unavailable/);
  assert.match(doc.getElementById('seg-list').textContent, /temporarily unavailable/);
  assert.equal(doc.getElementById('sec-seg').getAttribute('aria-busy'), 'false');
  assert.doesNotMatch(ctx.status(), /error:|loading/i);
  selectPeriod(ctx, 'fiscal-quarter', 'Q2');
  assert.equal(doc.getElementById('kpi-rev').textContent, '$900.0M');
});

test('history failure is shown immediately and late overview cannot mask it', async t => {
  const slow = deferred();
  const ctx = await setup({ respond: url => url.pathname.includes('/overview/')
    ? slow.promise : new Response('The page could not be found', { status: 404 }) });
  t.after(() => ctx.dom.window.close());
  assert.match(ctx.status(), /HTTP 404/);
  assert.equal(ctx.w.document.getElementById('card-revenue').getAttribute('aria-busy'), 'false');
  slow.resolve(overview); await settle();
  assert.match(ctx.status(), /HTTP 404/);
  assert.equal(ctx.w.document.getElementById('kpi-rev').textContent, '—');
  assert.equal(ctx.w.document.getElementById('kpi-price').textContent, '$123.45');
});

test('switching companies cancels previous requests and ignores a late old overview', async t => {
  const old = deferred();
  const ctx = await setup({ respond: url => url.pathname.endsWith('/AAPL') && url.pathname.includes('/overview/')
    ? old.promise : new Response(JSON.stringify(url.pathname.includes('/overview/')
      ? { ...overview, quote: { price: 456.78 } } : historyData())) });
  t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document;
  doc.getElementById('ticker').value = 'MSFT';
  doc.getElementById('ticker').dispatchEvent(new ctx.w.KeyboardEvent('keydown', { key: 'Enter' }));
  await settle();
  assert(ctx.requests.slice(0, 2).every(request => request.signal.aborted));
  assert(ctx.requests.slice(2).every(request => !request.signal.aborted));
  old.resolve(overview); await settle();
  assert.match(ctx.status(), /MSFT/);
  assert.equal(doc.getElementById('kpi-price').textContent, '$456.78');
});

test('late overview does not populate an explicitly unavailable requested year', async t => {
  const slow = deferred();
  const ctx = await setup({ url: 'https://fineasmcp.vercel.app/explorer?ticker=AAPL&period=annual&year=2000',
    respond: url => url.pathname.includes('/overview/') ? slow.promise
      : new Response(JSON.stringify(historyData('annual'))) });
  t.after(() => ctx.dom.window.close());
  slow.resolve(overview); await settle();
  assert.match(ctx.status(), /No data for AAPL.*FY 2000/);
  assert.equal(ctx.w.document.getElementById('kpi-rev').textContent, '—');
  assert.equal(ctx.w.document.getElementById('kpi-price').textContent, '—');
  assert.equal(ctx.w.document.getElementById('sec-geo').getAttribute('aria-busy'), 'false');
});

test('selecting an annual fiscal year changes metrics, provenance, chart and URL without another request', async t => {
  const ctx = await setup({ url: 'https://fineasmcp.vercel.app/explorer?ticker=AAPL&period=annual' });
  t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document, calls = ctx.calls.length;
  assert.equal(doc.getElementById('fiscal-year').value, '2025');
  selectPeriod(ctx, 'fiscal-year', '2024');
  assert.equal(doc.getElementById('kpi-rev').textContent, '$3.0B');
  assert.equal(doc.getElementById('kpi-fcf').textContent, '$750.0M');
  assert.match(doc.getElementById('prov').textContent, /2024-09-28/);
  assert.deepEqual(Array.from(ctx.charts.get('c-trend').xAxis.data), ['FY 2024']);
  assert.match(ctx.status(), /10-K FY 2024/);
  assert.equal(new URL(ctx.w.location.href).searchParams.get('year'), '2024');
  assert.equal(doc.getElementById('quarter-filter').hidden, true);
  assert.equal(ctx.calls.length, calls);
});

test('quarter selection compares against the same fiscal quarter a year earlier', async t => {
  const ctx = await setup(); t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document;
  selectPeriod(ctx, 'fiscal-quarter', 'Q2');
  assert.equal(doc.getElementById('kpi-rev').textContent, '$900.0M');
  assert.match(doc.getElementById('sub-rev').textContent, /50\.0%.*YoY/);
  assert.equal(doc.getElementById('kpi-fcf').textContent, '$225.0M');
  assert.equal(new URL(ctx.w.location.href).searchParams.get('quarter'), 'Q2');
  selectPeriod(ctx, 'fiscal-year', '2025');
  assert.equal(doc.getElementById('fiscal-quarter').value, 'Q2');
  assert.equal(doc.getElementById('kpi-rev').textContent, '$600.0M');
  assert.equal(doc.getElementById('sub-rev').textContent, '');
  assert.match(ctx.status(), /Q2 2025/);
});

test('deep links restore the selected year and quarter and keep the stock quote labeled current', async t => {
  const ctx = await setup({ url: 'https://fineasmcp.vercel.app/explorer?ticker=AAPL&period=quarterly&year=2025&quarter=Q3' });
  t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document;
  assert.equal(doc.getElementById('fiscal-year').value, '2025');
  assert.equal(doc.getElementById('fiscal-quarter').value, 'Q3');
  assert.equal(doc.getElementById('kpi-rev').textContent, '$800.0M');
  assert.match(doc.getElementById('lbl-price').textContent, /^Current price/);
  assert.equal(ctx.charts.get('c-trend').xAxis.data.at(-1), 'Q3 FY 2025');
  assert.match(doc.getElementById('geo-list').textContent, /Q3 FY 2025.*not available/);
});

test('an unavailable requested year never falls back to different-period metrics', async t => {
  const ctx = await setup({ url: 'https://fineasmcp.vercel.app/explorer?ticker=AAPL&period=annual&year=2000' });
  t.after(() => ctx.dom.window.close());
  assert.match(ctx.status(), /No data for AAPL.*FY 2000/);
  assert.equal(ctx.w.document.getElementById('kpi-rev').textContent, '—');
  selectPeriod(ctx, 'fiscal-year', '2024');
  assert.equal(ctx.w.document.getElementById('kpi-rev').textContent, '$3.0B');
});

test('a quarter-only deep link selects that quarter in the latest available fiscal year', async t => {
  const ctx = await setup({ url: 'https://fineasmcp.vercel.app/explorer?ticker=AAPL&quarter=Q2' });
  t.after(() => ctx.dom.window.close());
  assert.equal(ctx.w.document.getElementById('period').value, 'quarter');
  assert.equal(ctx.w.document.getElementById('fiscal-year').value, '2026');
  assert.equal(ctx.w.document.getElementById('fiscal-quarter').value, 'Q2');
  assert.equal(ctx.w.document.getElementById('kpi-rev').textContent, '$900.0M');
});

test('an older failed company request cannot overwrite a newer loaded selection', async t => {
  const ctx = await setup(); t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document;
  let rejectOld;
  ctx.w.fetch = async input => {
    const url = new URL(input);
    if (url.pathname.includes('/overview/')) return new Response(JSON.stringify(overview));
    if (url.pathname.endsWith('/OLD')) return new Promise((resolve, reject) => { rejectOld = reject; });
    return new Response(JSON.stringify(historyData()));
  };
  doc.getElementById('ticker').value = 'OLD';
  doc.getElementById('ticker').dispatchEvent(new ctx.w.KeyboardEvent('keydown', { key: 'Enter' }));
  doc.getElementById('ticker').value = 'MSFT';
  doc.getElementById('ticker').dispatchEvent(new ctx.w.KeyboardEvent('keydown', { key: 'Enter' }));
  await settle();
  rejectOld(new Error('Old request failed'));
  await settle();
  assert.match(ctx.status(), /MSFT · 10-Q Q3 2026/);
  assert.equal(doc.getElementById('kpi-rev').textContent, '$1.0B');
});

test('cash flow joins by period date rather than array position and missing values stay blank', async t => {
  const history = historyData();
  history.cashflow = history.cashflow.filter(row => row.date !== '2025-03-29').reverse();
  const ctx = await setup({ history }); t.after(() => ctx.dom.window.close());
  assert.equal(ctx.w.document.getElementById('kpi-fcf').textContent, '$250.0M');
  selectPeriod(ctx, 'fiscal-year', '2025');
  selectPeriod(ctx, 'fiscal-quarter', 'Q2');
  assert.equal(ctx.w.document.getElementById('kpi-fcf').textContent, '—');
});

test('latest segment and cross-check data are removed when a historical period is selected', async t => {
  const latest = { ...overview.latest, endDate: '2026-06-27', metrics: { ...overview.latest.metrics, netIncome: 200000000 } };
  const ctx = await setup({ overview: { ...overview, latest,
    crossCheck: { status: 'ok' }, segments: {
      sourceMeta: { reportDate: '2026-06-27', fiscalYear: '2026', fiscalPeriod: 'Q3', currency: 'USD' },
      product: [{ name: 'Latest product', value: 1000000000, pct: 100 }],
    } } });
  t.after(() => ctx.dom.window.close());
  assert.match(ctx.w.document.getElementById('seg-list').textContent, /Latest product/);
  assert.match(ctx.w.document.getElementById('prov').textContent, /cross-checked/);
  selectPeriod(ctx, 'fiscal-year', '2025');
  assert.doesNotMatch(ctx.w.document.getElementById('seg-list').textContent, /Latest product/);
  assert.match(ctx.w.document.getElementById('seg-list').textContent, /Q3 FY 2025.*not available/);
  assert.doesNotMatch(ctx.w.document.getElementById('prov').textContent, /cross-checked/);
});

test('fiscal quarter labels follow issuer metadata rather than the calendar date', async t => {
  const history = historyData();
  history.income[0] = { ...history.income[0], date: '2025-12-31', fiscalYear: '2026', period: 'Q2' };
  history.income = history.income.slice(0, 1);
  const ctx = await setup({ history }); t.after(() => ctx.dom.window.close());
  assert.equal(ctx.w.document.getElementById('fiscal-year').value, '2026');
  assert.equal(ctx.w.document.getElementById('fiscal-quarter').value, 'Q2');
  assert.match(ctx.w.document.getElementById('period-note').textContent, /2025-12-31/);
});

test('YTD fallback rows are not offered as standalone quarters and derived Q4 is labeled', async t => {
  const history = historyData();
  history.income[1]._meta = { ...history.income[1]._meta, isYtd: true, quality: 'ytd_fallback' };
  history.income[0].period = 'Q4';
  history.income[0]._meta.quality = 'q4_synthesized';
  const ctx = await setup({ history }); t.after(() => ctx.dom.window.close());
  assert.deepEqual(Array.from(ctx.w.document.getElementById('fiscal-quarter').options).map(option => option.value), ['Q4']);
  assert.match(ctx.w.document.getElementById('prov').textContent, /Q4 derived/);
});
async function searchCompany(ctx, query) {
  const input = ctx.w.document.getElementById('company-search');
  input.value = query;
  input.dispatchEvent(new ctx.w.Event('input', { bubbles: true }));
  await new Promise(resolve => setTimeout(resolve, 275));
}

test('company dropdown supports keyboard selection and preserves the reporting period', async t => {
  const ctx = await setup(); t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document;
  doc.getElementById('company-toggle').click();
  assert.equal(doc.getElementById('company-popover').hidden, false);
  assert.equal(doc.activeElement.id, 'company-search');
  assert.equal(doc.querySelectorAll('[role="option"]').length, 8);
  const input = doc.getElementById('company-search');
  input.dispatchEvent(new ctx.w.KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
  input.dispatchEvent(new ctx.w.KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
  assert.match(doc.getElementById(input.getAttribute('aria-activedescendant')).textContent, /MSFT/);
  input.dispatchEvent(new ctx.w.KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  await settle();
  assert.equal(doc.getElementById('ticker').value, 'MSFT');
  assert.equal(doc.getElementById('company-popover').hidden, true);
  assert.equal(doc.activeElement.id, 'company-toggle');
  assert.equal(new URL(ctx.w.location.href).searchParams.get('ticker'), 'MSFT');
  assert.equal(new URL(ctx.w.location.href).searchParams.get('period'), 'quarterly');
  assert.equal(ctx.calls.at(-2).pathname, '/api/v2/overview/MSFT');
  assert.match(ctx.status(), /MSFT · 10-Q/);
});

test('searching by company name uses the API and clicking a result loads its ticker', async t => {
  const ctx = await setup({ searchResults: [{ ticker: 'TSLA', name: 'Tesla, Inc.' }] });
  t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document;
  doc.getElementById('company-toggle').click();
  await searchCompany(ctx, 'tesla');
  assert.equal(ctx.calls.at(-1).origin, api);
  assert.equal(ctx.calls.at(-1).pathname, '/api/search');
  assert.equal(ctx.calls.at(-1).searchParams.get('q'), 'tesla');
  const option = doc.querySelector('[role="option"]');
  assert.match(option.textContent, /TSLATesla, Inc\./);
  option.click(); await settle();
  assert.equal(doc.getElementById('ticker').value, 'TSLA');
  assert.match(ctx.status(), /TSLA · 10-Q/);
});

test('search distinguishes no results from unavailable service', async t => {
  const ctx = await setup(); t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document;
  doc.getElementById('company-toggle').click();
  await searchCompany(ctx, 'no-such-company');
  assert.match(doc.getElementById('company-search-status').textContent, /No companies found/);
  ctx.w.fetch = async () => new Response('Bad Gateway', { status: 502 });
  await searchCompany(ctx, 'tesla');
  assert.match(doc.getElementById('company-search-status').textContent, /Search unavailable/);
  assert.equal(doc.getElementById('ticker').value, 'AAPL');
});

test('late search results cannot replace a newer query or reopen a closed dropdown', async t => {
  const ctx = await setup(); t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document, pending = [];
  ctx.w.fetch = () => new Promise(resolve => pending.push(resolve));
  doc.getElementById('company-toggle').click();
  await searchCompany(ctx, 'apple');
  await searchCompany(ctx, 'tesla');
  pending[1](new Response('{"results":[{"ticker":"TSLA","name":"Tesla"}]}'));
  await settle();
  pending[0](new Response('{"results":[{"ticker":"AAPL","name":"Apple"}]}'));
  await settle();
  assert.match(doc.getElementById('company-options').textContent, /TSLA/);
  assert.doesNotMatch(doc.getElementById('company-options').textContent, /AAPL/);
  await searchCompany(ctx, 'microsoft');
  doc.getElementById('company-search').dispatchEvent(new ctx.w.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  pending[2](new Response('{"results":[{"ticker":"MSFT","name":"Microsoft"}]}'));
  await settle();
  assert.equal(doc.getElementById('company-popover').hidden, true);
  assert.equal(doc.getElementById('company-toggle').getAttribute('aria-expanded'), 'false');
});

test('company names render as text and clicking outside dismisses the dropdown', async t => {
  const ctx = await setup({ searchResults: [{ ticker: 'SAFE', name: '<img src=x onerror=alert(1)>' }] });
  t.after(() => ctx.dom.window.close());
  const doc = ctx.w.document;
  doc.getElementById('company-toggle').click();
  await searchCompany(ctx, 'safe');
  assert.equal(doc.querySelector('#company-options img'), null);
  assert.match(doc.getElementById('company-options').textContent, /<img/);
  doc.body.click();
  assert.equal(doc.getElementById('company-popover').hidden, true);
});

test('quarterly deep link routes both requests to Railway and renders the data', async t => {
  const ctx = await setup(); t.after(() => ctx.dom.window.close());
  assert.equal(ctx.calls.length, 2);
  assert(ctx.calls.every(url => url.origin === api));
  assert.equal(ctx.calls[0].pathname, '/api/v2/overview/AAPL');
  assert.equal(ctx.calls[0].searchParams.get('period'), 'quarterly');
  assert.equal(ctx.calls[1].searchParams.get('period'), 'quarter');
  assert.equal(ctx.w.document.getElementById('period').value, 'quarter');
  assert.match(ctx.status(), /AAPL · 10-Q Q3 2026/);
  assert.equal(ctx.w.document.getElementById('kpi-rev').textContent, '$1.0B');
  assert.equal(ctx.charts.get('c-trend').series[0].data.at(-1), 1000000000);
});

for (const origin of [api, 'http://localhost:8000', 'http://127.0.0.1:8000']) {
  test('backend deployment keeps data requests on ' + origin, async t => {
    const ctx = await setup({ url: origin + '/explorer?ticker=AAPL&period=annual' });
    t.after(() => ctx.dom.window.close());
    assert(ctx.calls.every(url => url.origin === origin));
    assert(ctx.calls.every(url => url.searchParams.get('period') === 'annual'));
    assert.doesNotMatch(ctx.status(), /error:/);
  });
}

for (const [name, body, status, message] of [
  ['Vercel text 404', 'The page could not be found\nNOT_FOUND', 404, /HTTP 404/],
  ['proxy HTML 502', '<html>Bad Gateway</html>', 502, /HTTP 502/],
  ['malformed success', 'This is not JSON', 200, /invalid response/],
  ['empty success', '', 200, /invalid response/],
  ['JSON null', 'null', 200, /invalid response/],
  ['structured error', '{"detail":"Ticker not found"}', 404, /Ticker not found/],
  ['structured validation error', '{"detail":[{"msg":"invalid"}]}', 422, /HTTP 422/],
]) {
  test(name + ' produces a readable error', async t => {
    const ctx = await setup({ respond: () => new Response(body, { status }) });
    t.after(() => ctx.dom.window.close());
    assert.match(ctx.status(), message);
    assert.doesNotMatch(ctx.status(), /Unexpected token|JSON\.parse|\[object Object\]/);
  });
}

test('ADR source uses the issuer CIK when a filing agent submitted the report', async t => {
  const history = historyData();
  Object.assign(history.income[0], { cik: '0000937966', date: '2026-06-28', period: 'Q2' });
  history.income[0]._meta = { accession: '0001628280-26-048235', formType: '6-K', quality: 'standalone' };
  const ctx = await setup({ history, url: 'https://fineasmcp.vercel.app/explorer?ticker=ASML&period=quarterly&year=2026&quarter=Q2' });
  t.after(() => ctx.dom.window.close());
  assert.equal(ctx.w.document.getElementById('filing-link').href,
    'https://www.sec.gov/Archives/edgar/data/937966/000162828026048235/0001628280-26-048235-index.htm');
});

test('a quarter without normalized data offers access to the original filings', async t => {
  const ctx = await setup({ history: { income: [], cashflow: [] }, url: 'https://fineasmcp.vercel.app/explorer?ticker=TSM&period=quarterly' });
  t.after(() => ctx.dom.window.close());
  const link = ctx.w.document.getElementById('filing-link');
  assert.match(ctx.status(), /No standalone quarterly figures/);
  assert.equal(link.hidden, false);
  assert.equal(new URL(link.href).searchParams.get('CIK'), 'TSM');
});

test('6-K exhibit metrics display in the filed currency and revalidate stale HTTP history', async t => {
  const history = historyData();
  const row = history.income[0];
  Object.assign(row, { cik: 1046179, date: '2026-06-30', period: 'Q2', revenue: 38519222301,
    netIncome: 21423666402, _meta: { formType: '6-K', accession: '0001046179-26-000451',
      quality: 'standalone', filingCurrency: 'TWD', fxRate: 0.030321,
      sources: { revenue: 'https://www.sec.gov/Archives/edgar/data/1046179/000104617926000451/a2q26e_withguidancexfinal.htm' } } });
  const ctx = await setup({ history, url: 'https://fineasmcp.vercel.app/explorer?ticker=TSM&period=quarterly&year=2026&quarter=Q2' });
  t.after(() => ctx.dom.window.close());
  assert.equal(ctx.w.document.getElementById('kpi-rev').textContent, 'NT$1270.4B');
  assert.match(ctx.w.document.getElementById('prov').textContent, /SEC filing exhibit/);
  assert.equal(ctx.requests.find(req => req.url.pathname.includes('/financials-history/')).init.cache, 'no-cache');
});
