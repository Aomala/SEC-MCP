/* One navigation surface across the SEC research tools. */
(() => {
  try { document.documentElement.dataset.theme = localStorage.getItem('fineas-theme') === 'light' ? 'light' : 'dark'; } catch {}
  const names = { explorer:'Company explorer', comps:'Comps', filings:'10-K reader', screener:'Screener' };
  function context() {
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem('fineas_explorer_selection_v1') || '{}'); } catch {}
    const q = new URLSearchParams(location.search);
    const explicit = ['ticker','tickers','period','year','quarter'].some(key => q.has(key));
    const result = {};
    for (const key of ['ticker','period','year','quarter']) {
      const value = explicit ? q.get(key) : saved[key];
      if (value) result[key] = value;
    }
    return result;
  }
  function href(tool, extra = {}) {
    const q = new URLSearchParams(context());
    for (const [key,value] of Object.entries(extra)) value == null ? q.delete(key) : q.set(key,value);
    if (tool !== 'explorer') q.set('view',tool);
    return (tool === 'explorer' ? '/explorer' : '/') + '?' + q;
  }
  function refresh(active) {
    active ||= location.pathname === '/explorer' ? 'explorer' : new URLSearchParams(location.search).get('view');
    document.querySelectorAll('[data-nav-tool]').forEach(a => {
      a.href = href(a.dataset.navTool);
      if (a.dataset.navTool === active) a.setAttribute('aria-current','page'); else a.removeAttribute('aria-current');
    });
  }
  const host = document.getElementById('fineas-navigation');
  if (host) {
    host.className = 'fineas-topbar';
    const brand = document.createElement('a'); brand.href = '/explorer'; brand.className='fineas-brand';
    const logo = document.createElement('img'); logo.src='/static/logo-32.png'; logo.width=26; logo.height=26; logo.alt='';
    brand.append(logo,document.createTextNode('Fineas')); host.append(brand);
    const nav = document.createElement('nav'); nav.setAttribute('aria-label','Research tools');
    for (const [key,name] of Object.entries(names)) { const a=document.createElement('a'); a.dataset.navTool=key; a.textContent=name; nav.append(a); }
    const more=document.createElement('a'); more.href='/'; more.textContent='More tools'; nav.append(more);
    host.append(nav);
    const home=document.createElement('a'); home.href='https://fineas.ai'; home.className='fineas-home'; home.textContent='Fineas AI'; host.append(home);
  }
  window.FineasShell={href,refresh,context};
  refresh();
})();
