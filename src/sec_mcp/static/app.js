/* ============================================================
   Fineas.ai V2 — SEC Financial Intelligence Platform
   ============================================================
   Production-grade JavaScript application for SEC filing analysis.
   
   Modules:
   1. State management (tickers, data cache, chat history)
   2. DOM initialization (DOMContentLoaded, Lucide icons)
   3. Theme management (dark/light mode, localStorage)
   4. Search & autocomplete (debounced fetch, dropdown)
   5. Navigation & view switching (sidebar, header tabs)
   6. Main query flow (/api/chat, response handling)
   7. Context-aware chat (/api/chatbot with filing context)
   8. Dashboard rendering (KPI cards, provenance)
   9. Chart rendering (Chart.js: revenue, segments, margins)
   10. Period/form selectors (filing dropdown, load-filing)
   11. Utilities (formatting, markdown, escaping, analytics)
   ============================================================ */

// ========================================
// 1. STATE MANAGEMENT
// ========================================

/** Current selected ticker (uppercase) */
let _tk = null;

/** Current financial data from API */
let _curData = null;

/** Current accession number for loaded filing */
let _curAcc = null;

/** Available filings for the current ticker */
let _avail = [];

/** Foreign Private Issuer flag (true if 20-F/6-K/40-F detected) */
let _isFpi = false;

/** Chat conversation history for context */
let _chatHistory = [];

/** Cached filing sections (fetched from /api/filing_text) */
let _bgSections = {};

/** Chart.js instances: { revenue, segment, margin } */
const _charts = {};

/** Browser-side data cache: "TICKER|accession" -> {data, summary} */
const _dataCache = {};

/**
 * API base URL: use Railway directly when deployed on Vercel to avoid
 * Vercel's 10s rewrite timeout. On localhost, use relative paths.
 */
const API_BASE = (function() {
  const host = window.location.hostname;
  if (host === 'localhost' || host === '127.0.0.1') return '';
  // Deployed on Vercel or elsewhere → call Railway API directly
  return 'https://sec-mcp-production.up.railway.app';
})();

/** Current main view ('dashboard', 'compare', 'filing') */
let _activeView = 'dashboard';

/** Date range state for multi-year charts */
let _chartRange = 3;          // years (0 = all)
let _chartPeriod = 'annual';  // 'annual' or 'quarter'
let _fmpHistory = null;       // cached FMP history data

/** Debounce timer for search input */
let _searchTimeout = null;

/** Chart color palette (dark theme optimized) */
const CHART_COLORS = {
  brand: '#6366f1',
  brandBg: 'rgba(99,102,241,0.15)',
  success: '#10b981',
  successBg: 'rgba(16,185,129,0.1)',
  info: '#3b82f6',
  infoBg: 'rgba(59,130,246,0.1)',
  danger: '#ef4444',
  warning: '#f59e0b',
  text: '#94a3b8',
  grid: 'rgba(148,163,184,0.08)',
  palette: ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899'],
};

// ========================================
// 2. DOM INITIALIZATION
// ========================================

/**
 * Initialize the application on DOMContentLoaded.
 * Wires up all event listeners and renders initial state.
 */
// Register Chart.js datalabels plugin globally
if (typeof ChartDataLabels !== 'undefined') {
  Chart.register(ChartDataLabels);
  // Default: datalabels OFF globally (opt-in per chart)
  Chart.defaults.plugins.datalabels = { display: false };
}

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Lucide icons (convert data-lucide to SVG)
  lucide.createIcons();

  // Setup theme from localStorage
  initTheme();

  // Wire up all event handlers
  initKeyboardShortcuts();
  initNavigation();
  initSearch();
  initChat();

  // Terms of Service gate
  initTos();

  // Load research folders from localStorage
  loadResearchFolders();
  initResearchQueryClicks();

  // Log initialization complete
  console.log('[Fineas.ai] Application initialized');
});

// ========================================
// TERMS OF SERVICE GATE
// ========================================

function initTos() {
  const overlay = document.getElementById('tos-overlay');
  if (!overlay) return;

  // Auto-accept — let people use the app immediately
  // T&C is still accessible via a link if needed
  if (localStorage.getItem('fineas-tos-accepted') === '1') {
    overlay.style.display = 'none';
    return;
  }

  // Auto-accept after 1s with fade out (non-blocking)
  overlay.style.display = 'flex';
  setTimeout(() => {
    localStorage.setItem('fineas-tos-accepted', '1');
    overlay.style.opacity = '0';
    overlay.style.transition = 'opacity 0.5s ease';
    setTimeout(() => { overlay.style.display = 'none'; }, 500);
  }, 800);
}

function acceptTos() {
  localStorage.setItem('fineas-tos-accepted', '1');
  const overlay = document.getElementById('tos-overlay');
  if (overlay) {
    overlay.style.opacity = '0';
    overlay.style.transition = 'opacity 0.3s ease';
    setTimeout(() => { overlay.style.display = 'none'; }, 300);
  }
}

// ========================================
// DATE RANGE (controls SEC XBRL chart period display)
// ========================================

function setDateRange(years) {
  _chartRange = years;
  document.querySelectorAll('.range-presets .range-btn').forEach(b => {
    b.classList.toggle('active', parseInt(b.dataset.range) === years);
  });
  // Re-render the XBRL revenue chart with the current data
  if (_curData) renderRevenueChart(_curData);
}

function setChartPeriod(period) {
  _chartPeriod = period;
  document.querySelectorAll('.range-period .range-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.period === period);
  });
  if (_curData) renderRevenueChart(_curData);
}

function applyCustomRange() {
  const from = document.getElementById('range-from')?.value;
  const to = document.getElementById('range-to')?.value;
  if (from && to && parseInt(from) <= parseInt(to)) {
    _chartRange = -1; // custom
    document.querySelectorAll('.range-presets .range-btn').forEach(b => b.classList.remove('active'));
    if (_curData) renderRevenueChart(_curData);
  }
}

// ========================================
// FMP FOOTNOTE COMPARISON (background load only)
// ========================================

function loadFmpFootnote() {
  if (!_tk) return;
  const el = document.getElementById('data-footnote');
  if (!el) return;

  fetch(API_BASE + '/api/financials-history/' + encodeURIComponent(_tk) + '?period=annual&limit=5')
    .then(r => r.json())
    .then(j => {
      if (j.error || !j.income || !j.income.length) {
        el.style.display = 'none';
        return;
      }
      _fmpHistory = j;
      renderFootnote(j);
      // Re-render revenue chart with multi-year FMP data
      if (_curData) renderRevenueChart(_curData);
    })
    .catch(() => { el.style.display = 'none'; });
}

function renderFootnote(hist) {
  const el = document.getElementById('data-footnote');
  if (!el) return;

  const m = _curData?.metrics || {};
  const fmp = hist.income[0] || {};
  const poly = _curData?._crossCheck || {};

  // Polygon metric key mapping
  const polyMap = {
    'Revenue': 'revenue',
    'Net Income': 'net_income',
    'Gross Profit': null,
    'Operating Income': null,
    'EPS (Diluted)': 'eps',
    'Total Assets': 'total_assets',
  };

  // Compare SEC-MCP XBRL vs FMP vs Polygon for latest period
  const rows = [];
  const pairs = [
    ['Revenue', m.revenue, fmp.revenue],
    ['Net Income', m.net_income, fmp.netIncome],
    ['Gross Profit', m.gross_profit, fmp.grossProfit],
    ['Operating Income', m.operating_income, fmp.operatingIncome],
    ['EPS (Diluted)', m.eps_diluted, fmp.epsDiluted],
    ['Total Assets', m.total_assets, fmp.totalAssets],
  ];

  for (const [label, xbrl, fmpVal] of pairs) {
    if (xbrl == null && fmpVal == null) continue;
    const xbrlStr = xbrl != null ? fmtN(xbrl) : '—';
    const fmpStr = fmpVal != null ? fmtN(fmpVal) : '—';

    // Polygon value
    const polyKey = polyMap[label];
    const polyEntry = polyKey ? poly[polyKey] : null;
    const polyVal = polyEntry?.polygon;
    const polyStr = polyVal != null ? fmtN(polyVal) : '—';

    // Status: best match across all available sources
    let status = '';
    let statusTitle = '';
    const diffs = [];
    if (xbrl != null && xbrl !== 0) {
      if (fmpVal != null) diffs.push({ src: 'FMP', diff: Math.abs((xbrl - fmpVal) / xbrl * 100) });
      if (polyVal != null) diffs.push({ src: 'Polygon', diff: Math.abs((xbrl - polyVal) / xbrl * 100) });
    }
    if (diffs.length) {
      const maxDiff = Math.max(...diffs.map(d => d.diff));
      const details = diffs.map(d => d.src + ': ' + d.diff.toFixed(1) + '% diff').join(', ');
      if (maxDiff < 5) {
        status = '<span class="fn-match" style="color:var(--success)">✓</span>';
        statusTitle = 'All sources agree within 5%. ' + details;
      } else if (maxDiff < 20) {
        status = '<span class="fn-close" style="color:var(--warning)">⚠</span>';
        statusTitle = 'Sources differ >5%. ' + details;
      } else {
        status = '<span class="fn-mismatch" style="color:var(--danger)">✗</span>';
        statusTitle = 'Sources differ >20%. ' + details;
      }
    } else if (xbrl != null) {
      status = '<span class="fn-mismatch" style="color:var(--danger)">✗</span>';
      statusTitle = 'No cross-validation data available for this metric.';
    }

    rows.push('<tr><td>' + esc(label) + '</td><td class="right mono">' + xbrlStr +
      '</td><td class="right mono">' + polyStr +
      '</td><td class="right mono">' + fmpStr +
      '</td><td class="right" title="' + esc(statusTitle) + '">' + status + '</td></tr>');
  }

  if (!rows.length) { el.style.display = 'none'; return; }

  const fmpDate = fmp.date ? ' (' + fmp.date.slice(0, 4) + ')' : '';
  el.style.display = 'block';
  el.innerHTML =
    '<div class="card-header"><div><h3>Data Source Comparison</h3>' +
    '<p class="card-subtitle">SEC XBRL (primary) vs FMP' + fmpDate + ' vs Polygon.io</p></div></div>' +
    '<table class="data-table"><thead><tr><th>Metric</th><th class="right">SEC XBRL</th>' +
    '<th class="right">Polygon</th><th class="right">FMP</th><th class="right">Status</th></tr></thead>' +
    '<tbody>' + rows.join('') + '</tbody></table>' +
    '<p class="fn-note">Revenue & profitability charts use SEC XBRL data extracted directly from 10-K/10-Q filings. ' +
    'FMP and Polygon data shown for cross-reference only.</p>';
}

// ========================================
// 3. THEME MANAGEMENT
// ========================================

/**
 * Initialize theme from localStorage (default: dark).
 */
function initTheme() {
  const saved = localStorage.getItem('fineas-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  updateThemeUI(saved);
}

/**
 * Toggle between dark and light themes.
 */
function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('fineas-theme', next);
  updateThemeUI(next);
  
  // Redraw charts if they exist (colors may differ)
  Object.values(_charts).forEach(chart => {
    if (chart && typeof chart.update === 'function') {
      chart.update();
    }
  });
}

/**
 * Update theme UI (icon + label).
 */
function updateThemeUI(theme) {
  const icon = document.getElementById('theme-icon');
  const label = document.getElementById('theme-label');
  if (icon) {
    icon.setAttribute('data-lucide', theme === 'dark' ? 'sun' : 'moon');
  }
  if (label) {
    label.textContent = theme === 'dark' ? 'Light Mode' : 'Dark Mode';
  }
  lucide.createIcons();
}

// ========================================
// 4. SEARCH & AUTOCOMPLETE
// ========================================

/**
 * Wire up search input: debounced autocomplete + Enter to submit.
 */
function initSearch() {
  const inp = document.getElementById('search-input');
  if (!inp) return;
  
  // Debounced input → fetch autocomplete results
  inp.addEventListener('input', (e) => {
    debSearch(e.target.value);
  });
  
  // Enter key submits search
  inp.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      send(inp.value);
    }
  });
}

/**
 * Debounced search: fetch /api/search with autocomplete query.
 */
function debSearch(val) {
  clearTimeout(_searchTimeout);
  const drop = document.getElementById('search-results');
  
  // Hide dropdown if input is empty
  if (!val || val.length < 1) {
    if (drop) drop.style.display = 'none';
    return;
  }
  
  // Debounce for 250ms to avoid hammering API
  _searchTimeout = setTimeout(async () => {
    try {
      const r = await fetch(API_BASE + '/api/search?q=' + encodeURIComponent(val));
      if (!r.ok) throw new Error('Search failed: ' + r.status);
      
      const j = await r.json();
      if (!j.results || !j.results.length) {
        if (drop) drop.style.display = 'none';
        return;
      }
      
      // Build dropdown HTML
      let h = '';
      for (const x of j.results) {
        h += '<div class="sr-item" onclick="pickAsset(\'' + esc(x.ticker) + '\')" title="' + esc(x.name || '') + '">';
        h += '<span class="sr-ticker">' + esc(x.ticker) + '</span>';
        h += '<span class="sr-name">' + esc(x.name || '') + '</span>';
        if (x.exchange) {
          h += '<span class="sr-exchange">' + esc(x.exchange) + '</span>';
        }
        h += '</div>';
      }
      
      if (drop) {
        drop.innerHTML = h;
        drop.style.display = 'block';
      }
    } catch (e) {
      console.error('[Search Error]', e);
      if (drop) drop.style.display = 'none';
    }
  }, 250);
}

/**
 * Handle autocomplete item click: load ticker and run query.
 */
function pickAsset(tk) {
  // Close dropdown and clear input
  const drop = document.getElementById('search-results');
  if (drop) drop.style.display = 'none';
  
  const inp = document.getElementById('search-input');
  if (inp) inp.value = '';
  
  // Reset state and load company
  _tk = tk.toUpperCase();
  _bgSections = {};
  _chatHistory = [];
  _curAcc = null;
  _curData = null;
  _fmpHistory = null;
  
  // Trigger main query
  send(tk);
}

/**
 * Close search dropdown on outside click.
 */
document.addEventListener('click', (e) => {
  const drop = document.getElementById('search-results');
  const searchBar = e.target.closest('.search-bar');
  if (drop && !searchBar) {
    drop.style.display = 'none';
  }
});

// ========================================
// 5. NAVIGATION & VIEW SWITCHING
// ========================================

/**
 * Wire up sidebar nav items and header view tabs.
 */
function initNavigation() {
  // Sidebar nav items
  document.querySelectorAll('.nav-item[data-view]').forEach((item) => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      switchView(item.dataset.view);
    });
  });
  
  // Header view tabs
  document.querySelectorAll('.view-tab[data-view]').forEach((tab) => {
    tab.addEventListener('click', () => {
      switchView(tab.dataset.view);
    });
  });
  
  // Period toggle buttons (annual/quarterly)
  document.querySelectorAll('.toggle-btn[data-period]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.toggle-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      
      // Update form selector based on period
      const formSel = document.getElementById('sel-form');
      if (formSel) {
        formSel.value = btn.dataset.period === 'quarterly' ? '10-Q' : '10-K';
        onFormSel();
      }
    });
  });
  
  // Form selector (10-K, 10-Q, 20-F, etc.)
  const formSel = document.getElementById('sel-form');
  if (formSel) {
    formSel.addEventListener('change', onFormSel);
  }
  
  // Period dropdown (specific filing)
  const periodSel = document.getElementById('sel-period');
  if (periodSel) {
    periodSel.addEventListener('change', onPeriodSel);
  }
  
  // Export button
  const exportBtn = document.getElementById('btn-export');
  if (exportBtn) {
    exportBtn.addEventListener('click', exportData);
  }
  
  // Refresh button
  const refreshBtn = document.getElementById('btn-refresh');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', refreshData);
  }
  
  // Theme toggle
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', toggleTheme);
  }
}

/**
 * Switch to a different view (dashboard, statements, comps, filings, insights).
 */
function switchView(view) {
  _activeView = view;

  // Update sidebar active state
  document.querySelectorAll('.nav-item').forEach((i) => i.classList.remove('active'));
  const activeNav = document.querySelector('.nav-item[data-view="' + view + '"]');
  if (activeNav) activeNav.classList.add('active');

  // Update header tabs active state
  document.querySelectorAll('.view-tab').forEach((t) => t.classList.remove('active'));
  const activeTab = document.querySelector('.view-tab[data-view="' + view + '"]');
  if (activeTab) activeTab.classList.add('active');

  // Hide dashboard and all view panels
  const dashboard = document.getElementById('dashboard');
  if (dashboard) dashboard.style.display = 'none';
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';
  document.querySelectorAll('.view-panel').forEach((p) => p.style.display = 'none');

  if (view === 'dashboard') {
    // Show dashboard if data is loaded, otherwise show welcome
    if (_curData) {
      if (dashboard) { dashboard.style.display = 'block'; dashboard.classList.remove('fade-in'); void dashboard.offsetWidth; dashboard.classList.add('fade-in'); }
    } else {
      if (welcome) { welcome.style.display = 'flex'; welcome.classList.remove('fade-in'); void welcome.offsetWidth; welcome.classList.add('fade-in'); }
    }
  } else {
    // Show the matching panel
    const panel = document.getElementById('panel-' + view);
    if (panel) { panel.style.display = 'block'; panel.classList.remove('fade-in'); void panel.offsetWidth; panel.classList.add('fade-in'); }

    // Trigger data loading for each view
    if (view === 'statements' && _curData) renderFullStatement('income');
    if (view === 'comps') loadCompsView();
    if (view === 'filings') loadFilingsView();
    if (view === 'insights') { /* content stays until Generate is clicked */ }

    // Invalidate Leaflet map size when switching back to dashboard
    if (view === 'dashboard' && _geoMap) {
      setTimeout(() => { _geoMap.invalidateSize(); }, 100);
    }
  }
}

// ========================================
// 6. MAIN QUERY FLOW
// ========================================

/**
 * Shorthand: pass a ticker/query string directly to send().
 */
function q(t) {
  const inp = document.getElementById('search-input');
  if (inp && !inp.value) inp.value = t;
  send(t);
}

/**
 * Main query: send message to /api/chat and handle response.
 * This is the primary entry point for user queries.
 */
async function send(msg) {
  msg = msg ? msg.trim() : (document.getElementById('search-input')?.value?.trim() || '');
  if (!msg) return;
  
  // Hide welcome panel, show dashboard
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';
  
  const dashboard = document.getElementById('dashboard');
  if (dashboard) dashboard.style.display = 'block';
  
  // Show loading spinner
  const content = document.getElementById('content');
  if (content) {
    const spinner = document.createElement('div');
    spinner.className = 'loading-spinner';
    spinner.id = 'main-spinner';
    spinner.innerHTML = '<div class="spinner"></div><p>Loading financial data...</p>';
    content.appendChild(spinner);
  }
  
  try {
    const r = await fetch(API_BASE + '/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg }),
    });
    
    if (!r.ok) throw new Error('API error: ' + r.status);
    
    const j = await r.json();
    
    // Remove loading spinner
    const spinner = document.getElementById('main-spinner');
    if (spinner) spinner.remove();
    
    if (j.type === 'error') {
      showError(j.message || 'Unknown error');
    } else if (j.type === 'result') {
      handleResult(j);
    }
  } catch (e) {
    const spinner = document.getElementById('main-spinner');
    if (spinner) spinner.remove();
    showError('Request failed: ' + e.message);
    console.error('[Send Error]', e);
  }
  
  // Clear search input
  const inp = document.getElementById('search-input');
  if (inp) inp.value = '';
}

/**
 * Handle API response from /api/chat.
 * Routes to appropriate rendering function based on tool type.
 */
function handleResult(j) {
  const tk = (j.resolved_tickers && j.resolved_tickers.length) ? j.resolved_tickers[0] : '';
  
  // Financials or explain tools: render dashboard
  if (j.tool === 'financials' || j.tool === 'explain') {
    const d = j.data || {};

    // Attach provenance metadata from API response onto the data object
    if (j.sources) d._sources = j.sources;
    if (j.cross_check) d._crossCheck = j.cross_check;

    // Update ticker
    if (tk) _tk = tk.toUpperCase();
    
    // Store current data
    _curData = d;
    if (d.filing_info) {
      _curAcc = d.filing_info.accession_number || null;
    }
    
    // Update company pill in header
    updateCompanyPill(d);

    // Save to research folders
    saveResearch(_tk, d.company_name);

    // Switch to dashboard view
    switchView('dashboard');
    
    // Render all dashboard components
    renderDashboard(d);
    
    // Fetch available filings for period dropdown
    if (tk) fetchAvail(tk);
    
    // Cache the data for fast period switching
    if (d.filing_info) {
      _dataCache[tk + '|' + d.filing_info.accession_number] = {
        data: d,
        summary: j.summary || '',
      };
    }
    
    // Add chat message summarizing the load
    addChatMessage('assistant', buildLoadedSummary(d));
  } else if (j.tool === 'compare') {
    // Switch to Comparables tab and render comparison table
    switchView('comps');
    renderComparison(j);
  } else if (j.tool === 'filing_text') {
    // Handle raw filing text
    renderFilingText(j);
  } else {
    // Generic fallback: show response in chat
    addChatMessage('assistant', j.summary || JSON.stringify(j, null, 2));
  }
}

// ========================================
// 7. CONTEXT-AWARE CHAT
// ========================================

/**
 * Initialize chat input: wire up Enter key.
 */
function initChat() {
  const inp = document.getElementById('chat-inp');
  if (!inp) return;

  inp.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendCb();
    }
  });
}

/**
 * Set chat input and send immediately.
 */
function cbQ(text) {
  const inp = document.getElementById('chat-inp');
  if (inp) inp.value = text;
  sendCb();
}

/**
 * Send chat message: POST to /api/chatbot with context.
 */
async function sendCb() {
  const inp = document.getElementById('chat-inp');
  const msg = inp?.value?.trim();
  if (!msg) return;
  
  inp.value = '';

  // Add user message to UI
  addChatMessage('user', msg);

  // Track query in research folders
  if (_tk) addResearchQuery(_tk, msg);

  // Show typing indicator
  addChatLoading();
  
  try {
    // If no data is loaded, try to extract a ticker from the message and load it first
    if (!_curData && !_tk) {
      // Check if message mentions a known ticker pattern (1-5 uppercase letters)
      const tkMatch = msg.match(/\b([A-Z]{1,5})\b/);
      if (tkMatch) {
        removeChatLoading();
        addChatMessage('assistant', 'Loading **' + tkMatch[1] + '** data first — one moment...');
        addChatLoading();
        // Trigger a load via the main search, then re-send the question
        await new Promise((resolve) => {
          const origHandler = handleResult;
          send(tkMatch[1]);
          // Wait for data to load (poll _curData)
          const check = setInterval(() => {
            if (_curData) { clearInterval(check); resolve(); }
          }, 500);
          setTimeout(() => { clearInterval(check); resolve(); }, 15000); // 15s timeout
        });
      }
    }

    // Build context from current data
    const ctx = Object.assign({}, _curData || {});
    if (_bgSections && Object.keys(_bgSections).length) {
      ctx._filing_sections = _bgSections;
    }

    // Send to /api/chatbot
    const body = {
      message: msg,
      ticker: _tk || '',
      context: ctx,
      history: _chatHistory.slice(-6), // Last 6 messages for context
    };
    
    const r = await fetch(API_BASE + '/api/chatbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    
    if (!r.ok) throw new Error('Chat API error: ' + r.status);
    
    const j = await r.json();
    
    // Remove typing indicator
    removeChatLoading();
    
    // Build answer with citations
    let answer = j.answer || 'No response.';
    if (j.citations && j.citations.length) {
      const cites = j.citations.map((c) => esc(c.source)).join(', ');
      answer += '\n\n_Sources: ' + cites + '_';
    }
    
    // Add assistant response
    addChatMessage('assistant', answer);
    
    // Update history
    _chatHistory.push({ role: 'user', content: msg });
    _chatHistory.push({ role: 'assistant', content: answer });
    
    // Trim history to last 12 messages (6 round-trips)
    if (_chatHistory.length > 12) {
      _chatHistory = _chatHistory.slice(-12);
    }
  } catch (e) {
    removeChatLoading();
    addChatMessage('assistant', 'Error: ' + e.message);
    console.error('[Chat Error]', e);
  }
}

/**
 * Toggle chat panel visibility.
 */
function toggleChat() {
  const panel = document.getElementById('chat-panel');
  if (!panel) return;
  
  panel.classList.toggle('open');
  
  const btn = document.getElementById('chat-toggle');
  if (btn) btn.classList.toggle('active');
  
  // Focus input when opening
  if (panel.classList.contains('open')) {
    setTimeout(() => {
      document.getElementById('chat-inp')?.focus();
    }, 100);
  }
}

/**
 * Start a new chat conversation. Clears history and messages.
 */
function newChat() {
  _chatHistory = [];
  const container = document.getElementById('chat-messages');
  if (container) {
    container.innerHTML = '';
    // Re-add the welcome message
    const welcome = document.createElement('div');
    welcome.className = 'chat-welcome';
    welcome.id = 'chat-welcome';
    welcome.innerHTML = '<p><strong>Ask anything</strong> about the loaded data</p><p class="small">Executive summaries, comparisons, insights</p>';
    container.appendChild(welcome);
  }
  // Reinitialize lucide icons for the new welcome content
  if (typeof lucide !== 'undefined') lucide.createIcons();
  // Focus chat input
  document.getElementById('chat-inp')?.focus();
}

/**
 * Add a message to the chat panel.
 */
function addChatMessage(role, text) {
  const container = document.getElementById('chat-messages');
  if (!container) return;
  
  // Hide welcome message on first message
  const welcome = document.getElementById('chat-welcome');
  if (welcome) welcome.style.display = 'none';
  
  // Create message div
  const div = document.createElement('div');
  div.className = 'msg msg-' + role;
  div.innerHTML = '<div class="msg-bubble">' + md(text) + '</div>';

  // Wrap tables in scrollable container and add styling class
  div.querySelectorAll('table').forEach(t => {
    t.classList.add('data-table');
    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    t.parentNode.insertBefore(wrap, t);
    wrap.appendChild(t);
  });

  container.appendChild(div);

  // Auto-scroll to bottom
  container.scrollTop = container.scrollHeight;
}

/**
 * Add typing indicator to chat.
 */
function addChatLoading() {
  const container = document.getElementById('chat-messages');
  if (!container) return;
  
  const div = document.createElement('div');
  div.className = 'msg msg-assistant msg-loading';
  div.id = 'chat-loading';
  div.innerHTML = '<div class="msg-bubble"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>';
  
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

/**
 * Remove typing indicator from chat.
 */
function removeChatLoading() {
  const loading = document.getElementById('chat-loading');
  if (loading) loading.remove();
}

// ========================================
// 8. DASHBOARD RENDERING
// ========================================

/**
 * Update company pill in header with ticker and name.
 */
function updateCompanyPill(d) {
  const pill = document.getElementById('company-pill');
  if (!pill) return;
  
  pill.style.display = 'flex';
  
  const ticker = document.getElementById('pill-ticker');
  const name = document.getElementById('pill-name');
  
  if (ticker) ticker.textContent = _tk || d.ticker_or_cik || '';
  if (name) name.textContent = d.company_name || '';
}

/**
 * Render entire dashboard: KPIs, charts, tables, provenance.
 */
function renderDashboard(d) {
  const dashboard = document.getElementById('dashboard');
  if (!dashboard) return;
  dashboard.style.display = 'block';
  
  const m = d.metrics || {};
  const r = d.ratios || {};
  const pm = d.prior_metrics || {};
  
  // 1. Render KPI cards (pass cross_check for source badges)
  renderKPIs(m, r, pm, d._crossCheck || {});

  // 2. Render company overview
  renderCompanyOverview(d);

  // 3. Show date range bar
  const rangeBar = document.getElementById('date-range-bar');
  if (rangeBar) rangeBar.style.display = 'flex';

  // 4. Render charts with slight delay to allow DOM rendering
  setTimeout(() => {
    renderRevenueChart(d);
    renderSegmentChart(d);
    renderMarginChart(d);
    renderGeoMap(d);
    // Load FMP data for footnote comparison only (not for charts)
    loadFmpFootnote();
  }, 100);

  // 4. Render peer comparison table
  renderPeerTable(d);

  // 4. Render provenance/validation bar
  renderProvenance(d);

  // 4b. Render web context panel (Perplexity market context)
  renderWebContext(d);

  // 5. Render confidence scores + validation details
  renderConfidence(d);

  // 6. Render financial statement preview
  renderStatementPreview(d);
}

/**
 * Render KPI cards in a grid.
 */
function renderKPIs(m, r, pm, crossCheck) {
  const grid = document.getElementById('kpi-grid');
  if (!grid) return;
  const cc = crossCheck || {};
  
  // Derived metrics
  const operatingIncome = m.operating_income || null;
  const ebitda = m.ebitda || (operatingIncome && m.depreciation ? operatingIncome + m.depreciation : (m.net_income && m.income_tax && m.interest_expense && m.depreciation ? m.net_income + m.income_tax + m.interest_expense + m.depreciation : null));
  const workingCapital = (m.current_assets != null && m.current_liabilities != null) ? m.current_assets - m.current_liabilities : null;
  const netDebt = m.net_debt != null ? m.net_debt : ((m.long_term_debt != null || m.total_debt != null) && m.cash != null ? (m.total_debt || m.long_term_debt || 0) - m.cash : null);
  const roic = (m.net_income != null && m.total_equity != null && m.long_term_debt != null && (m.total_equity + m.long_term_debt) !== 0) ? (m.net_income / (m.total_equity + m.long_term_debt)) * 100 : null;
  const interestCoverage = (operatingIncome != null && m.interest_expense != null && m.interest_expense !== 0) ? operatingIncome / Math.abs(m.interest_expense) : null;

  // Define KPIs: label, value, prior value, icon, color
  const kpis = [
    {
      label: 'Revenue',
      key: 'revenue',
      value: m.revenue,
      prior: pm?.revenue,
      icon: 'dollar-sign',
      color: 'brand',
    },
    {
      label: 'Net Income',
      key: 'net_income',
      value: m.net_income,
      prior: pm?.net_income,
      icon: 'trending-up',
      color: 'emerald',
    },
    {
      label: 'Operating Income',
      value: operatingIncome,
      prior: pm?.operating_income,
      icon: 'bar-chart-3',
      color: 'blue',
    },
    {
      label: 'EBITDA',
      value: ebitda,
      icon: 'layers',
      color: 'violet',
    },
    {
      label: 'Free Cash Flow',
      value: m.free_cash_flow,
      prior: pm?.free_cash_flow,
      icon: 'activity',
      color: 'blue',
    },
    {
      label: 'Gross Margin',
      value: r.gross_margin != null ? r.gross_margin * 100 : null,
      fmt: 'pct',
      icon: 'pie-chart',
      color: 'violet',
    },
    {
      label: 'EPS',
      value: m.eps_diluted,
      prior: pm?.eps_diluted,
      fmt: 'eps',
      icon: 'users',
      color: 'rose',
    },
    {
      label: 'Total Assets',
      key: 'total_assets',
      value: m.total_assets,
      prior: pm?.total_assets,
      icon: 'building-2',
      color: 'amber',
    },
    {
      label: 'Working Capital',
      value: workingCapital,
      icon: 'wallet',
      color: 'emerald',
    },
    {
      label: 'Net Debt',
      value: netDebt,
      icon: 'credit-card',
      color: 'rose',
    },
    {
      label: 'ROIC',
      value: roic,
      fmt: 'pct',
      icon: 'target',
      color: 'brand',
    },
    {
      label: 'D/E Ratio',
      value: r.debt_to_equity,
      fmt: 'ratio',
      icon: 'scale',
      color: 'amber',
    },
    {
      label: 'Interest Coverage',
      value: interestCoverage,
      fmt: 'ratio',
      icon: 'shield',
      color: 'emerald',
    },
  ];
  
  let h = '';
  kpis.forEach((kpi, i) => {
    if (kpi.value == null) return;

    const change = calcChange(kpi.value, kpi.prior);
    const fmtVal =
      kpi.fmt === 'pct'
        ? kpi.value.toFixed(1) + '%'
        : kpi.fmt === 'eps'
          ? '$' + kpi.value.toFixed(2)
          : kpi.fmt === 'ratio'
            ? kpi.value.toFixed(2) + 'x'
            : fmtN(kpi.value);

    // Source badge: SEC always, verified if Polygon cross-check matched this metric
    const metricKey = kpi.key || kpi.label.toLowerCase().replace(/\s+/g, '_');
    const verified = cc && cc[metricKey] && cc[metricKey].match;
    h += buildKpiCard(kpi.label, fmtVal, change, kpi.icon, kpi.color, i, verified);
  });
  
  grid.innerHTML = h;
  lucide.createIcons();
}

const KPI_TOOLTIPS = {
  'Revenue': 'Total sales from all business segments. Source: SEC EDGAR XBRL filing.',
  'Net Income': 'Profit after all expenses, taxes, and interest. Source: SEC EDGAR.',
  'Free Cash Flow': 'Cash from operations minus capital expenditures. Measures cash generation ability.',
  'Gross Margin': 'Revenue minus cost of goods sold, as a percentage of revenue.',
  'Total Assets': 'Sum of all assets on the balance sheet.',
  'EPS': 'Earnings per share (diluted). Net income divided by diluted share count.',
  'Operating Income': 'Profit from core business operations before interest and taxes.',
  'EBITDA': 'Earnings before interest, taxes, depreciation & amortization. Proxy for operating cash flow.',
  'Working Capital': 'Current assets minus current liabilities. Measures short-term liquidity.',
  'Net Debt': 'Total debt minus cash and equivalents. Negative means net cash position.',
  'ROIC': 'Return on invested capital. Net income / (equity + long-term debt). Measures capital efficiency.',
  'D/E Ratio': 'Debt-to-equity ratio. Total debt / total equity. Measures financial leverage.',
  'Interest Coverage': 'Operating income / interest expense. Higher = more ability to service debt.',
};

/**
 * Build a single KPI card HTML.
 */
function buildKpiCard(label, value, change, icon, color, index, verified) {
  let changeHtml = '';
  if (change != null) {
    const trendIcon = change >= 0 ? 'arrow-up-right' : 'arrow-down-right';
    const trendClass = change >= 0 ? 'positive' : 'negative';
    changeHtml =
      '<span class="kpi-change ' +
      trendClass +
      '">' +
      '<i data-lucide="' +
      trendIcon +
      '"></i>' +
      (change >= 0 ? '+' : '') +
      change.toFixed(1) +
      '%</span>';
  }

  // Source badges: SEC always present; Verified or Unverified based on cross-check
  let badges = '<span class="source-badge source-sec" title="Data extracted from SEC EDGAR XBRL filing"><i data-lucide="shield-check"></i>SEC</span>';
  if (verified === true) {
    badges += '<span class="source-badge source-verified" title="Cross-validated against Polygon.io — values match within 5%"><i data-lucide="check-circle"></i>Verified</span>';
  } else if (verified === false) {
    badges += '<span class="source-badge source-unverified" title="No cross-validation data available for this metric"><i data-lucide="alert-circle"></i>Unverified</span>';
  }

  // Build tooltip for the KPI card
  let cardTip = KPI_TOOLTIPS[label] || label;
  if (verified === true) cardTip += ' [Polygon verified]';
  else if (verified === false) cardTip += ' [Unverified]';

  return (
    '<div class="kpi-card animate-in" title="' + esc(cardTip) + '" style="animation-delay:' +
    index * 0.05 +
    's">' +
    '<div class="kpi-top">' +
    '<div class="kpi-icon ' +
    color +
    '"><i data-lucide="' +
    icon +
    '"></i></div>' +
    changeHtml +
    '</div>' +
    '<div class="kpi-value">' +
    value +
    '</div>' +
    '<div class="kpi-label">' +
    label +
    '</div>' +
    '<div class="kpi-source-badges">' + badges + '</div>' +
    '</div>'
  );
}

/**
 * SIC code to business description mapping (top ~50 codes).
 */
const SIC_DESCRIPTIONS = {
  '3571': 'Electronic computers',
  '3572': 'Computer storage devices',
  '3674': 'Semiconductors & related devices',
  '3672': 'Printed circuit boards',
  '7372': 'Prepackaged software',
  '7371': 'Computer programming & data processing',
  '7374': 'Computer processing & data preparation',
  '5045': 'Computers & peripherals (wholesale)',
  '5961': 'Catalog & mail-order houses',
  '5065': 'Electronic parts (wholesale)',
  '4813': 'Telephone communications',
  '4812': 'Radiotelephone communications',
  '4841': 'Cable & pay television services',
  '4911': 'Electric services',
  '4931': 'Electric & other services combined',
  '2834': 'Pharmaceutical preparations',
  '2836': 'Biological products',
  '2835': 'In-vitro diagnostics',
  '2860': 'Industrial chemicals',
  '3841': 'Surgical & medical instruments',
  '6022': 'State commercial banks',
  '6020': 'National commercial banks',
  '6021': 'National commercial banks',
  '6035': 'Savings institutions (federally chartered)',
  '6036': 'Savings institutions (not federally chartered)',
  '6311': 'Life insurance',
  '6321': 'Fire, marine & casualty insurance',
  '6331': 'Accident & health insurance',
  '6199': 'Finance services',
  '6211': 'Security brokers & dealers',
  '6141': 'Personal credit institutions',
  '6153': 'Short-term business credit',
  '6159': 'Federal-sponsored credit agencies',
  '6798': 'Real estate investment trusts',
  '5411': 'Grocery stores',
  '5311': 'Department stores',
  '5331': 'Variety stores',
  '5912': 'Drug stores & proprietary stores',
  '5812': 'Eating places',
  '3711': 'Motor vehicles & passenger car bodies',
  '3714': 'Motor vehicle parts & accessories',
  '3721': 'Aircraft',
  '3724': 'Aircraft engines & parts',
  '3760': 'Guided missiles & space vehicles',
  '1311': 'Crude petroleum & natural gas',
  '2911': 'Petroleum refining',
  '1040': 'Gold mining',
  '1000': 'Metal mining',
  '7370': 'Services — computer programming, data processing',
};

/**
 * Render company overview card — data-driven summary with business description,
 * size metrics, performance highlights.
 */
function renderCompanyOverview(d) {
  const el = document.getElementById('company-overview');
  const titleEl = document.getElementById('overview-title');
  const subEl = document.getElementById('overview-subtitle');
  if (!el) return;

  const m = d.metrics || {};
  const r = d.ratios || {};
  const pm = d.prior_metrics || {};
  const fi = d.filing_info || {};
  const name = d.company_name || _tk || 'Company';
  const sic = d.sic_code || '';

  if (titleEl) titleEl.textContent = name;
  if (subEl) subEl.textContent = _tk + ' · ' + (fi.form_type || '10-K') + ' · ' + (fi.filing_date || '');

  // Business description from SIC
  const sicDesc = SIC_DESCRIPTIONS[sic] || d.industry_class || 'Public company';

  // Revenue growth
  let revGrowth = null;
  if (m.revenue && pm.revenue) {
    revGrowth = ((m.revenue - pm.revenue) / Math.abs(pm.revenue)) * 100;
  }

  // Net income growth
  let niGrowth = null;
  if (m.net_income != null && pm.net_income != null && pm.net_income !== 0) {
    niGrowth = ((m.net_income - pm.net_income) / Math.abs(pm.net_income)) * 100;
  }

  // Market cap estimate (shares × EPS × ~PE, or just note shares)
  const shares = m.shares_outstanding;
  const eps = m.eps_diluted;

  // Performance assessment
  let perfLabel = 'stable';
  let perfColor = 'var(--text-secondary)';
  if (revGrowth != null) {
    if (revGrowth > 15) { perfLabel = 'strong growth'; perfColor = 'var(--success)'; }
    else if (revGrowth > 5) { perfLabel = 'moderate growth'; perfColor = 'var(--success)'; }
    else if (revGrowth > -5) { perfLabel = 'stable'; perfColor = 'var(--warning)'; }
    else { perfLabel = 'declining'; perfColor = 'var(--danger)'; }
  }

  // Derived overview metrics
  const workingCapital = (m.current_assets != null && m.current_liabilities != null) ? m.current_assets - m.current_liabilities : null;
  const netDebt = m.net_debt != null ? m.net_debt : ((m.long_term_debt != null || m.total_debt != null) && m.cash != null ? (m.total_debt || m.long_term_debt || 0) - m.cash : null);
  const interestCoverage = (m.operating_income != null && m.interest_expense != null && m.interest_expense !== 0) ? m.operating_income / Math.abs(m.interest_expense) : null;

  // Build HTML — Bloomberg-style structured grid
  let h = '';

  // Business description (compact)
  h += '<div class="overview-section">';
  h += '<div class="overview-label">BUSINESS <span style="color:' + perfColor + ';font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:0.5px">' + perfLabel + '</span></div>';
  h += '<p class="overview-text">' + esc(name) + ' — <strong>' + esc(sicDesc) + '</strong>';
  h += ' (SIC ' + esc(sic) + ')';
  if (m.revenue) {
    h += ' — ' + fmtN(m.revenue) + ' revenue';
    if (fi.filing_date && fi.filing_date.length >= 4) h += ' (' + fi.filing_date.substring(0, 4) + ')';
  }
  h += '</p></div>';

  // Structured metrics grid — Bloomberg Terminal style
  // Helper to build a metric cell
  function mCell(label, value) {
    if (value == null || value === '—') return '';
    const cls = typeof value === 'string' && value.startsWith('-') ? ' negative' : (typeof value === 'string' && value.startsWith('+') ? ' positive' : '');
    return '<div class="metric-item"><span class="metric-label">' + label + '</span><span class="metric-value' + cls + '">' + value + '</span></div>';
  }

  h += '<div class="metrics-grid">';

  // Row 1: Growth & Margins
  h += mCell('Rev Growth', revGrowth != null ? (revGrowth >= 0 ? '+' : '') + revGrowth.toFixed(1) + '%' : null);
  h += mCell('NI Growth', niGrowth != null ? (niGrowth >= 0 ? '+' : '') + niGrowth.toFixed(1) + '%' : null);
  h += mCell('Gross Margin', r.gross_margin != null ? (r.gross_margin * 100).toFixed(1) + '%' : null);

  // Row 2: Profitability
  h += mCell('Net Margin', r.net_margin != null ? (r.net_margin * 100).toFixed(1) + '%' : null);
  h += mCell('ROE', r.roe != null ? (r.roe * 100).toFixed(1) + '%' : null);
  h += mCell('EPS (Diluted)', eps != null ? '$' + eps.toFixed(2) : null);

  // Row 3: Financial Health
  h += mCell('Current Ratio', r.current_ratio != null ? r.current_ratio.toFixed(2) + 'x' : null);
  h += mCell('D/E Ratio', r.debt_to_equity != null ? r.debt_to_equity.toFixed(2) + 'x' : null);
  h += mCell('Free Cash Flow', m.free_cash_flow ? fmtN(m.free_cash_flow) : null);

  // Row 4: Leverage & Capital
  h += mCell('Net Debt', netDebt != null ? fmtN(netDebt) : null);
  h += mCell('Interest Coverage', interestCoverage != null ? interestCoverage.toFixed(1) + 'x' : null);
  h += mCell('Working Capital', workingCapital != null ? fmtN(workingCapital) : null);

  h += '</div>';

  el.innerHTML = h;
}

// ========================================
// 9. CHART RENDERING (Chart.js)
// ========================================

/**
 * Get default Chart.js config for dark theme.
 */
function getChartDefaults() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1e293b',
        titleColor: '#f1f5f9',
        bodyColor: '#94a3b8',
        borderColor: '#334155',
        borderWidth: 1,
        cornerRadius: 8,
        padding: 10,
        usePointStyle: true,
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: CHART_COLORS.text, font: { size: 11 } },
      },
      y: {
        grid: { color: CHART_COLORS.grid },
        ticks: { color: CHART_COLORS.text, font: { size: 11, family: 'monospace' } },
      },
    },
  };
}

/**
 * Render revenue trend chart (bar chart: revenue, net income, FCF).
 * Shows multi-year comparison when prior_metrics are available.
 */
function renderRevenueChart(d) {
  if (_charts.revenue) _charts.revenue.destroy();

  const ctx = document.getElementById('revenue-chart');
  if (!ctx) return;

  const m = d.metrics || {};
  const pm = d.prior_metrics || {};
  const qm = d.qoq_metrics || {};

  // Build multi-period labels + data
  const labels = [];
  const revenue = [];
  const netIncome = [];
  const fcf = [];

  const fy = d.fiscal_year || 'Current';
  const isQuarterly = d.period_type === 'quarterly';

  // Use FMP multi-year history if available (3-5 years of data)
  if (_fmpHistory && _fmpHistory.income && _fmpHistory.income.length >= 2) {
    const incomeData = [..._fmpHistory.income].reverse(); // oldest first
    const cfData = _fmpHistory.cashflow ? [..._fmpHistory.cashflow].reverse() : [];

    for (let i = 0; i < incomeData.length; i++) {
      const row = incomeData[i];
      const cfRow = cfData[i] || {};
      const yr = row.date ? row.date.substring(0, 4) : row.calendarYear || ('Y' + (i + 1));
      labels.push(yr);
      revenue.push(row.revenue ? row.revenue / 1e9 : 0);
      netIncome.push(row.netIncome != null ? row.netIncome / 1e9 : 0);
      fcf.push(cfRow.freeCashFlow != null ? cfRow.freeCashFlow / 1e9 : 0);
    }
  } else {
    // Fallback: Prior vs Current (original behavior)
    if (pm.revenue != null || pm.net_income != null) {
      labels.push(isQuarterly ? 'Prior Yr' : (typeof fy === 'number' ? fy - 1 : 'Prior'));
      revenue.push(pm.revenue ? pm.revenue / 1e9 : 0);
      netIncome.push(pm.net_income ? pm.net_income / 1e9 : 0);
      fcf.push(pm.free_cash_flow ? pm.free_cash_flow / 1e9 : 0);
    }

    // QoQ data (middle, quarterly only)
    if (isQuarterly && (qm.revenue != null || qm.net_income != null)) {
      labels.push('Prior Qtr');
      revenue.push(qm.revenue ? qm.revenue / 1e9 : 0);
      netIncome.push(qm.net_income ? qm.net_income / 1e9 : 0);
      fcf.push(qm.free_cash_flow ? qm.free_cash_flow / 1e9 : 0);
    }

    // Current period (rightmost)
    labels.push(d.quarter_label || fy);
    revenue.push(m.revenue ? m.revenue / 1e9 : 0);
    netIncome.push(m.net_income ? m.net_income / 1e9 : 0);
    fcf.push(m.free_cash_flow ? m.free_cash_flow / 1e9 : 0);
  }

  const defaults = getChartDefaults();

  _charts.revenue = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Revenue ($B)',
          data: revenue,
          backgroundColor: CHART_COLORS.brandBg,
          borderColor: CHART_COLORS.brand,
          borderWidth: 2,
          borderRadius: 6,
          order: 2,
        },
        {
          label: 'Net Income ($B)',
          data: netIncome,
          backgroundColor: CHART_COLORS.successBg,
          borderColor: CHART_COLORS.success,
          borderWidth: 2,
          borderRadius: 6,
          order: 3,
        },
        {
          label: 'FCF ($B)',
          data: fcf,
          backgroundColor: CHART_COLORS.infoBg,
          borderColor: CHART_COLORS.info,
          borderWidth: 2,
          borderRadius: 6,
          order: 4,
        },
      ],
    },
    options: {
      ...defaults,
      layout: { padding: { top: 24 } },
      plugins: {
        ...defaults.plugins,
        legend: {
          display: true,
          labels: { color: CHART_COLORS.text, usePointStyle: true, pointStyle: 'rect', padding: 12, font: { size: 11 } }
        },
        datalabels: {
          display: true,
          anchor: 'end',
          align: 'top',
          offset: 2,
          font: { size: 10, weight: 600, family: "'JetBrains Mono', monospace" },
          color: function(ctx) {
            return ctx.dataset.borderColor || CHART_COLORS.text;
          },
          formatter: function(v) {
            if (v == null || v === 0) return '';
            return '$' + v.toFixed(1) + 'B';
          },
        },
      },
      scales: {
        ...defaults.scales,
        y: {
          ...defaults.scales.y,
          ticks: {
            ...defaults.scales.y.ticks,
            callback: (v) => '$' + v.toFixed(0) + 'B',
          },
        },
      },
    },
  });

  // Show YTD badge if applicable
  const periodInfo = ctx.closest('.chart-card')?.querySelector('.chart-period-info');
  if (!periodInfo && d.is_ytd) {
    const badge = document.createElement('div');
    badge.className = 'chart-period-info';
    badge.innerHTML = '<span class="ytd-badge">YTD</span> ' + (d.quarter_label || '');
    ctx.closest('.chart-card')?.querySelector('.card-header')?.appendChild(badge);
  }
}

/**
 * Render confidence scores grid and validation list.
 */
// Confidence score explanations for hover tooltips
const CONF_EXPLANATIONS = {
  revenue: 'Total revenue/sales. 99% = exact XBRL match. Lower scores mean the value was derived from partial matches or component aggregation.',
  net_income: 'Net income (bottom line). 99% = exact XBRL tag found. Lower = fallback to contains-match or custom extension concepts.',
  gross_profit: 'Revenue minus cost of goods sold. Often derived (revenue conf × COGS conf × 0.9) rather than directly reported.',
  operating_income: 'Income from core operations. May be computed from gross profit minus operating expenses (0.70 confidence).',
  ebitda: 'Earnings before interest, taxes, depreciation & amortization. 0.85 = full calculation, 0.65 = fallback (OI + D&A).',
  total_assets: 'Sum of all assets on the balance sheet. High confidence = directly reported XBRL concept.',
  stockholders_equity: 'Net assets belonging to shareholders. Usually a direct XBRL match.',
  operating_cash_flow: 'Cash generated from operations. Direct XBRL match preferred.',
  free_cash_flow: 'Operating cash flow minus capex. Confidence = min(OCF, capex) since it requires both values.',
  eps_diluted: 'Diluted earnings per share. Direct XBRL match is typical for this metric.',
  cost_of_revenue: 'Direct costs of producing goods/services. High confidence when exact XBRL concept is found.',
  capital_expenditures: 'Money spent on property, plant & equipment. May require sign adjustment.',
  current_assets: 'Assets expected to convert to cash within 1 year. May be aggregated from components (0.70).',
  current_liabilities: 'Obligations due within 1 year. Usually a direct XBRL match.',
  long_term_debt: 'Debt maturing beyond 1 year. Multiple XBRL concepts may be tried.',
};

function renderConfidence(d) {
  const card = document.getElementById('confidence-card');
  const grid = document.getElementById('confidence-grid');
  const valList = document.getElementById('validation-list');
  if (!card || !grid || !valList) return;

  const conf = d.confidence_scores || {};
  const sourced = d.metrics_sourced || {};
  const validation = d.validation || [];

  // Only show if we have confidence data
  const entries = Object.entries(conf).filter(([k, v]) => v > 0);
  if (!entries.length) {
    card.style.display = 'none';
    return;
  }
  card.style.display = 'block';

  // Sort by confidence (lowest first to highlight issues)
  entries.sort((a, b) => a[1] - b[1]);

  // Key metrics to always show
  const keyMetrics = new Set(['revenue', 'net_income', 'gross_profit', 'operating_income',
    'ebitda', 'total_assets', 'stockholders_equity', 'operating_cash_flow', 'free_cash_flow',
    'eps_diluted', 'cost_of_revenue']);

  const displayed = entries.filter(([k]) => keyMetrics.has(k));

  let h = '';
  for (const [metric, score] of displayed) {
    const pct = Math.round(score * 100);
    const level = pct >= 80 ? 'high' : pct >= 50 ? 'medium' : 'low';
    const label = metric.replace(/_/g, ' ');
    const src = sourced[metric] || '';
    const explanation = CONF_EXPLANATIONS[metric] || '';
    const srcTip = src ? '\nXBRL concept: ' + src : '';
    const levelLabel = pct >= 80 ? 'Exact XBRL match' : pct >= 50 ? 'Partial/contains match' : 'Aggregated or fallback';
    const tip = explanation + srcTip + '\nMethod: ' + levelLabel;

    h += '<div class="conf-item" title="' + esc(tip) + '">' +
      '<span class="conf-label">' + esc(label) + '<span class="conf-info">?</span></span>' +
      '<div class="conf-bar-wrap"><div class="conf-bar ' + level + '" style="width:' + pct + '%"></div></div>' +
      '<span class="conf-pct ' + level + '">' + pct + '%</span>' +
      '</div>';
  }
  grid.innerHTML = h;

  // Validation messages
  let vh = '';
  if (!validation.length) {
    vh = '<div class="validation-item success"><i data-lucide="shield-check"></i> All validation checks passed</div>';
  } else {
    for (const v of validation) {
      const cls = v.severity === 'error' ? 'error' : 'warning';
      const icon = v.severity === 'error' ? 'alert-circle' : 'alert-triangle';
      vh += '<div class="validation-item ' + cls + '"><i data-lucide="' + icon + '"></i> ' + esc(v.message) + '</div>';
    }
  }
  valList.innerHTML = vh;
  lucide.createIcons();
}

/**
 * Render segment revenue breakdown (doughnut chart).
 */
function renderSegmentChart(d) {
  // Destroy existing chart
  if (_charts.segment) _charts.segment.destroy();

  const ctx = document.getElementById('segment-chart');
  if (!ctx) return;

  let segments = d.segments?.revenue_segments || [];

  // If XBRL has < 2 segments, try fetching from filing text
  if (segments.length < 2 && _tk) {
    const card = ctx.closest('.chart-card');
    if (card) card.style.display = 'block';
    ctx.style.display = 'none';
    const legendEl = document.getElementById('segment-legend');
    if (legendEl) legendEl.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;padding:24px;color:var(--text-tertiary);font-size:12px;gap:8px"><div class="spinner" style="width:16px;height:16px"></div>Loading segments...</div>';

    fetch(API_BASE + '/api/segments/' + encodeURIComponent(_tk))
      .then(r => r.json())
      .then(j => {
        if (j.segments && j.segments.length >= 2) {
          d.segments = d.segments || {};
          d.segments.revenue_segments = j.segments;
          ctx.style.display = 'block';
          renderSegmentChart(d);
          const sub = document.getElementById('segments-subtitle');
          const srcLabels = { fmp: 'Financial Modeling Prep', filing_text: '10-K filing text', xbrl: 'SEC XBRL' };
          if (sub) sub.textContent = 'Source: ' + (srcLabels[j.source] || j.source);
        } else {
          if (card) card.style.display = 'none';
        }
      })
      .catch(() => {
        if (card) card.style.display = 'none';
      });
    return;
  }

  if (!segments.length) {
    if (ctx.closest('.chart-card')) ctx.closest('.chart-card').style.display = 'none';
    return;
  }

  const card = ctx.closest('.chart-card');
  if (card) card.style.display = 'block';
  ctx.style.display = 'block';

  // Premium color palette — distinct, readable, ordered by contrast
  const PIE_COLORS = [
    '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#06b6d4', '#ec4899', '#14b8a6', '#f97316', '#a855f7',
    '#22d3ee', '#fb923c',
  ];

  const total = segments.reduce((sum, s) => sum + (s.value || 0), 0);

  _charts.segment = new Chart(ctx, {
    type: 'pie',
    data: {
      labels: segments.map((s) => s.segment || s.name),
      datasets: [{
        data: segments.map((s) => s.value || 0),
        backgroundColor: PIE_COLORS.slice(0, segments.length),
        borderWidth: 2,
        borderColor: getComputedStyle(document.documentElement).getPropertyValue('--bg-secondary').trim() || '#111827',
        hoverOffset: 8,
        hoverBorderWidth: 3,
        hoverBorderColor: '#fff',
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: 4 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#f1f5f9',
          titleFont: { size: 13, weight: 600 },
          bodyColor: '#cbd5e1',
          bodyFont: { size: 12 },
          borderColor: '#334155',
          borderWidth: 1,
          padding: 12,
          cornerRadius: 8,
          callbacks: {
            label: function(ctx) {
              const val = ctx.parsed || 0;
              const pct = total > 0 ? (val / total * 100).toFixed(1) : '0';
              return ' ' + fmtN(val) + '  (' + pct + '%)';
            },
          },
        },
        datalabels: {
          display: function(ctx) {
            // Only show label if slice is > 8% of total
            const val = ctx.dataset.data[ctx.dataIndex];
            return total > 0 && (val / total) > 0.08;
          },
          color: '#fff',
          font: { size: 11, weight: 600, family: "'Inter', sans-serif" },
          textShadowColor: 'rgba(0,0,0,0.6)',
          textShadowBlur: 4,
          formatter: function(val, ctx) {
            const pct = total > 0 ? (val / total * 100).toFixed(0) : '0';
            return pct + '%';
          },
        },
      },
    },
  });

  // Build rich legend with values
  const legendEl = document.getElementById('segment-legend');
  if (legendEl) {
    legendEl.innerHTML = segments
      .map((s, i) => {
        const pct = total > 0 ? ((s.value / total) * 100).toFixed(1) : '0';
        return (
          '<div class="seg-item">' +
          '<span class="seg-dot" style="background:' + PIE_COLORS[i % PIE_COLORS.length] + '"></span>' +
          '<span class="seg-name">' + esc(s.segment || s.name) + '</span>' +
          '<span class="seg-val">' + fmtN(s.value) + '</span>' +
          '<span class="seg-pct">' + pct + '%</span>' +
          '</div>'
        );
      })
      .join('');
  }
}

/**
 * Render margin comparison chart (horizontal bars).
 */
function renderMarginChart(d) {
  // Destroy existing chart
  if (_charts.margin) _charts.margin.destroy();
  
  const ctx = document.getElementById('margin-chart');
  if (!ctx) return;
  
  const r = d.ratios || {};
  
  // Collect available margins
  const margins = [];
  if (r.gross_margin != null) {
    margins.push({ label: 'Gross', value: r.gross_margin * 100 });
  }
  if (r.operating_margin != null) {
    margins.push({ label: 'Operating', value: r.operating_margin * 100 });
  }
  if (r.net_margin != null) {
    margins.push({ label: 'Net', value: r.net_margin * 100 });
  }
  if (r.ebitda_margin != null) {
    margins.push({ label: 'EBITDA', value: r.ebitda_margin * 100 });
  }
  if (r.fcf_margin != null) {
    margins.push({ label: 'FCF', value: r.fcf_margin * 100 });
  }
  
  if (!margins.length) return;
  
  const defaults = getChartDefaults();
  
  _charts.margin = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: margins.map((m) => m.label),
      datasets: [
        {
          data: margins.map((m) => m.value),
          backgroundColor: margins.map((_, i) => CHART_COLORS.palette[i]),
          borderRadius: 6,
          barThickness: 'flex',
          maxBarThickness: 30,
        },
      ],
    },
    options: {
      ...defaults,
      indexAxis: 'y',
      scales: {
        x: {
          ...defaults.scales.x,
          grid: { color: CHART_COLORS.grid },
          ticks: {
            ...defaults.scales.x.ticks,
            callback: (v) => v + '%',
          },
        },
        y: {
          ...defaults.scales.y,
          grid: { display: false },
        },
      },
    },
  });
}

// ========================================
// 10. PEER TABLE & PROVENANCE
// ========================================

/**
 * Render peer comparison table.
 */
function renderPeerTable(d) {
  const tbody = document.getElementById('peer-tbody');
  if (!tbody) return;
  
  const peers = d.peers || [];
  if (!peers.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No peer data available</td></tr>';
    return;
  }
  
  let h = '';
  for (const p of peers) {
    h += '<tr>';
    h += '<td class="peer-name">' + esc(p.ticker || p.name) + '</td>';
    h += '<td>' + fmtN(p.revenue) + '</td>';
    h += '<td>' + fmtN(p.net_income) + '</td>';
    h += '<td>' + (p.net_margin != null ? (p.net_margin * 100).toFixed(1) + '%' : '—') + '</td>';
    h += '<td>' + (p.pe_ratio != null ? p.pe_ratio.toFixed(1) + 'x' : '—') + '</td>';
    h += '</tr>';
  }
  
  tbody.innerHTML = h;
}

/**
 * Render provenance and validation status.
 */
function renderProvenance(d) {
  const el = document.getElementById('provenance');
  if (!el) return;
  
  const fi = d.filing_info || {};
  const val = d.validation || [];
  const allPassed = val.every((v) => v.severity !== 'error');
  
  const validationStatus = allPassed
    ? '<span class="prov-badge success"><i data-lucide="shield-check"></i>All validation checks passed</span>'
    : '<span class="prov-badge warning"><i data-lucide="alert-triangle"></i>' +
      val.length +
      ' validation issues</span>';
  
  // Compute average confidence
  const conf = d.confidence_scores || {};
  const confVals = Object.values(conf).filter(v => v > 0);
  const avgConf = confVals.length ? Math.round((confVals.reduce((a, b) => a + b, 0) / confVals.length) * 100) : 0;
  const confLevel = avgConf >= 80 ? 'success' : avgConf >= 50 ? 'warning' : 'error';
  const confBadge = avgConf > 0
    ? '<span class="prov-badge ' + (confLevel === 'error' ? 'warning' : confLevel) + '"><i data-lucide="gauge"></i>Confidence: ' + avgConf + '%</span>'
    : '';

  // Cache status
  const cacheBadge = d._from_cache
    ? '<span class="prov-badge success"><i data-lucide="zap"></i>Cached</span>'
    : '';

  // YTD badge
  const ytdBadge = d.is_ytd
    ? '<span class="prov-badge warning"><i data-lucide="clock"></i>' + esc(d.quarter_label || 'YTD') + '</span>'
    : '';

  // Polygon cross-validation badge
  const src = d._sources || {};
  const polyBadge = src.polygon_validated
    ? '<span class="prov-badge success"><i data-lucide="check-circle"></i>Polygon Verified</span>'
    : src.sec_edgar
      ? '<span class="prov-badge neutral"><i data-lucide="minus-circle"></i>Polygon: N/A</span>'
      : '';

  // Web context indicator
  const webBadge = src.web_context
    ? '<span class="prov-badge warning"><i data-lucide="globe"></i>Web Context Available</span>'
    : '';

  el.innerHTML =
    '<div class="prov-badges">' +
    validationStatus +
    confBadge +
    '<span class="prov-badge neutral"><i data-lucide="database"></i>Source: ' +
    esc(fi.form_type || '10-K') +
    ' filed ' +
    esc(fi.filing_date || '') +
    '</span>' +
    '<span class="prov-badge neutral"><i data-lucide="clock"></i>XBRL 4-pass extraction</span>' +
    '<span class="prov-badge brand"><i data-lucide="layers"></i>Industry: ' +
    esc(d.industry_class || 'standard') +
    '</span>' +
    ytdBadge +
    cacheBadge +
    polyBadge +
    webBadge +
    '</div>';
  
  lucide.createIcons();
}

/**
 * Render web context panel (Perplexity market news) below provenance.
 * Clearly labeled as non-SEC data.
 */
function renderWebContext(d) {
  // Remove existing web context card if present
  const existing = document.getElementById('web-context-card');
  if (existing) existing.remove();

  const src = d._sources || {};
  if (!src.web_context) return;

  const prov = document.getElementById('provenance');
  if (!prov) return;

  const card = document.createElement('div');
  card.id = 'web-context-card';
  card.className = 'web-context-card';

  let citationsHtml = '';
  if (src.web_citations && src.web_citations.length) {
    citationsHtml = '<div class="web-citations"><strong>Sources:</strong> ';
    citationsHtml += src.web_citations.map(function(url, i) {
      const domain = url.replace(/^https?:\/\//, '').split('/')[0];
      return '<a href="' + esc(url) + '" target="_blank" rel="noopener">[' + (i+1) + '] ' + esc(domain) + '</a>';
    }).join(' &middot; ');
    citationsHtml += '</div>';
  }

  const content = miniMarkdown(src.web_context);
  card.innerHTML =
    '<div class="web-context-header">' +
    '<i data-lucide="globe"></i>' +
    '<span class="source-badge source-web"><i data-lucide="alert-triangle"></i>WEB SOURCE</span>' +
    '<span class="web-context-disclaimer">Not from SEC filings</span>' +
    '</div>' +
    '<h4 class="web-context-title">Market Context</h4>' +
    '<div class="web-context-body">' + content + '</div>' +
    citationsHtml;

  // Insert after provenance
  prov.insertAdjacentElement('afterend', card);
  lucide.createIcons();
}

/**
 * Render financial statement preview table.
 */
function renderStatementPreview(d) {
  const tbody = document.getElementById('statement-tbody');
  if (!tbody) return;
  
  const m = d.metrics || {};
  const r = d.ratios || {};
  const pm = d.prior_metrics || {};
  
  // Build income statement preview
  const rows = [
    { label: 'Revenue', current: m.revenue, prior: pm?.revenue },
    { label: 'Cost of Revenue', current: m.cost_of_revenue, prior: pm?.cost_of_revenue },
    { label: 'Gross Profit', current: m.gross_profit, prior: pm?.gross_profit },
    { label: 'Operating Expenses', current: m.operating_expenses, prior: pm?.operating_expenses },
    { label: 'Operating Income', current: m.operating_income, prior: pm?.operating_income },
    { label: 'Interest Expense', current: m.interest_expense, prior: pm?.interest_expense },
    { label: 'Pretax Income', current: m.pretax_income, prior: pm?.pretax_income },
    { label: 'Income Tax', current: m.income_tax, prior: pm?.income_tax },
    { label: 'Net Income', current: m.net_income, prior: pm?.net_income },
  ];
  
  let h = '';
  for (const row of rows) {
    if (row.current == null && row.prior == null) continue;
    const change = calcChange(row.current, row.prior);
    const changeHtml =
      change != null
        ? '<span class="' + (change >= 0 ? 'positive' : 'negative') + '">' +
          (change >= 0 ? '+' : '') +
          change.toFixed(1) +
          '%</span>'
        : '—';
    h +=
      '<tr>' +
      '<td class="stmt-label">' +
      row.label +
      '</td>' +
      '<td class="stmt-value">' +
      fmtN(row.current) +
      '</td>' +
      '<td class="stmt-value">' +
      fmtN(row.prior) +
      '</td>' +
      '<td class="stmt-change">' +
      changeHtml +
      '</td>' +
      '</tr>';
  }
  
  tbody.innerHTML = h;
}

// ========================================
// 11. PERIOD / FORM SELECTORS
// ========================================

/**
 * Fetch available filings for a ticker.
 */
async function fetchAvail(tk) {
  try {
    const r = await fetch(API_BASE + '/api/filings/' + encodeURIComponent(tk));
    if (!r.ok) throw new Error('Filings API error: ' + r.status);
    
    const j = await r.json();
    _avail = j.filings || [];
    
    // Detect FPI status
    const hasDomestic = _avail.some((f) => ['10-K', '10-Q'].includes(f.form_type));
    const hasForeign = _avail.some((f) => ['20-F', '6-K', '40-F'].includes(f.form_type));
    _isFpi = hasForeign && !hasDomestic;
    
    // Populate dropdown
    populatePeriodDropdown();
  } catch (e) {
    console.error('[Fetch Avail Error]', e);
    _avail = [];
  }
}

/**
 * Populate period dropdown based on selected form.
 */
function populatePeriodDropdown() {
  const sel = document.getElementById('sel-period');
  if (!sel) return;
  
  const formSel = document.getElementById('sel-form');
  const formFilter = formSel ? formSel.value : '';
  
  let filtered = _avail;
  if (formFilter) {
    // Map primary forms to alternatives (e.g., 10-K includes 20-F for FPI)
    const altMap = {
      '10-K': ['10-K', '20-F', '40-F'],
      '10-Q': ['10-Q', '6-K'],
      '20-F': ['20-F', '10-K', '40-F'],
      '6-K': ['6-K', '10-Q'],
    };
    const acceptedForms = altMap[formFilter] || [formFilter];
    filtered = _avail.filter((f) => acceptedForms.includes(f.form_type));
  }
  
  let h = '<option value="">Select Period</option>';
  for (const f of filtered) {
    const selected = _curAcc === f.accession ? ' selected' : '';
    // Extract fiscal year from filing date
    // Annual filings (10-K/20-F): fiscal year is usually the year before filing date
    // or the year of the period end. We approximate from filing_date.
    const filingDate = f.filing_date || '';
    const filingYear = filingDate ? parseInt(filingDate.substring(0, 4)) : '';
    const filingMonth = filingDate ? parseInt(filingDate.substring(5, 7)) : 0;
    const isAnnual = ['10-K', '20-F', '40-F'].includes(f.form_type);
    const isQuarterly = ['10-Q', '6-K'].includes(f.form_type);

    let periodLabel = '';
    if (isAnnual && filingYear) {
      // Annual filings are typically filed 60-90 days after fiscal year end
      // If filed Jan-Mar, it's usually for the prior year
      const fy = filingMonth <= 3 ? filingYear - 1 : filingYear;
      periodLabel = 'FY' + fy;
    } else if (isQuarterly && filingYear) {
      // Determine quarter from filing month
      // Q1 filed ~May, Q2 filed ~Aug, Q3 filed ~Nov
      let q = '';
      if (filingMonth >= 1 && filingMonth <= 2) q = 'Q4';      // Q4 of prior year
      else if (filingMonth >= 3 && filingMonth <= 5) q = 'Q1';
      else if (filingMonth >= 6 && filingMonth <= 8) q = 'Q2';
      else if (filingMonth >= 9 && filingMonth <= 11) q = 'Q3';
      else q = 'Q4';
      const qYear = (q === 'Q4' && filingMonth <= 2) ? filingYear - 1 : filingYear;
      periodLabel = q + ' ' + qYear;
    }

    h +=
      '<option value="' +
      esc(f.accession) +
      '|' +
      esc(f.form_type) +
      '"' +
      selected +
      '>' +
      esc(f.form_type) +
      (periodLabel ? ' · ' + periodLabel : '') +
      ' · ' +
      shortDate(f.filing_date) +
      '</option>';
  }
  
  sel.innerHTML = h;
}

/**
 * Handle form selector change: update period dropdown.
 */
function onFormSel() {
  populatePeriodDropdown();
  
  // Optionally auto-load first period
  const sel = document.getElementById('sel-period');
  if (sel && sel.options.length > 1) {
    sel.selectedIndex = 1;
    onPeriodSel();
  }
}

/**
 * Handle period dropdown change: load specific filing.
 */
function onPeriodSel() {
  const sel = document.getElementById('sel-period');
  if (!sel || !sel.value) return;
  
  const parts = sel.value.split('|');
  if (parts.length === 2) {
    loadFiling(parts[0], parts[1]);
  }
}

/**
 * Load a specific filing by accession and form type.
 */
async function loadFiling(acc, ft) {
  if (!_tk) return;
  
  _curAcc = acc;
  
  // Check cache first
  const cached = _dataCache[_tk + '|' + acc];
  if (cached) {
    _curData = cached.data;
    renderDashboard(cached.data);
    return;
  }
  
  // Show loading
  const spinner = document.getElementById('main-spinner') || document.createElement('div');
  if (!spinner.id) {
    spinner.id = 'main-spinner';
    spinner.className = 'loading-spinner';
    spinner.innerHTML = '<div class="spinner"></div><p>Loading filing...</p>';
    const content = document.getElementById('content');
    if (content) content.appendChild(spinner);
  } else {
    spinner.style.display = 'block';
  }
  
  try {
    const r = await fetch(API_BASE + '/api/load-filing', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ticker: _tk,
        accession: acc,
        form_type: ft,
      }),
    });
    
    if (!r.ok) throw new Error('Load filing API error: ' + r.status);
    
    const j = await r.json();
    
    if (spinner) spinner.style.display = 'none';
    
    if (j.data) {
      // Cache the data
      _dataCache[_tk + '|' + acc] = {
        data: j.data,
        summary: j.summary || '',
      };
      
      // Update state and render
      _curData = j.data;
      renderDashboard(j.data);
    }
  } catch (e) {
    if (spinner) spinner.style.display = 'none';
    showError('Error loading filing: ' + e.message);
    console.error('[Load Filing Error]', e);
  }
}

// ========================================
// 12. COMPARISON & FILING TEXT
// ========================================

/**
 * Render comparison results.
 */
function renderComparison(j) {
  const tbody = document.getElementById('comps-tbody');
  const sub = document.getElementById('comps-sub');
  if (!tbody) return;

  const results = j.results || [];
  if (!results.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No comparison data available</td></tr>';
    return;
  }

  if (sub) sub.textContent = results.length + ' companies compared';

  // Populate chips
  const chips = document.getElementById('comps-chips');
  if (chips) {
    chips.innerHTML = results.map(r => {
      const tk = (r.data || {}).ticker_or_cik || '?';
      return '<span class="chip">' + esc(tk) + '</span>';
    }).join('');
  }

  // Build table rows
  let h = '';
  for (const r of results) {
    const d = r.data || {};
    const m = d.metrics || {};
    const name = d.company_name || d.ticker_or_cik || '?';
    const tk = d.ticker_or_cik || '?';

    h += '<tr>';
    h += '<td><strong>' + esc(tk) + '</strong><br><span class="text-muted" style="font-size:11px">' + esc(name) + '</span></td>';
    h += '<td class="right">' + fmtN(m.revenue) + '</td>';
    h += '<td class="right">' + fmtN(m.net_income) + '</td>';
    h += '<td class="right">' + (m.gross_margin != null ? (m.gross_margin * 100).toFixed(1) + '%' : '—') + '</td>';
    h += '<td class="right">' + (m.net_margin != null ? (m.net_margin * 100).toFixed(1) + '%' : '—') + '</td>';
    h += '<td class="right">' + fmtN(m.total_assets) + '</td>';
    h += '<td class="right">' + (m.eps_diluted != null ? '$' + m.eps_diluted.toFixed(2) : '—') + '</td>';
    h += '</tr>';
  }
  tbody.innerHTML = h;

  // Add narrative to chat if available
  if (j.comparison_narrative) {
    addChatMessage('assistant', j.comparison_narrative);
  }
}

/**
 * Render raw filing text.
 */
function renderFilingText(j) {
  const content = document.getElementById('content');
  if (!content) return;
  
  const text = j.data?.text || j.data || '';
  let h = '<div class="filing-text">';
  h += '<pre>' + esc(String(text).substring(0, 5000)) + '...(truncated)</pre>';
  h += '</div>';
  
  content.innerHTML = h;
}

// ========================================
// 13. VIEW RENDERERS (Statements, Comps, Filings, Insights)
// ========================================

/**
 * Render full financial statement by type (income, balance, cashflow).
 * Prefers the richer statement arrays from backend if available,
 * falls back to top-level metrics.
 */
function renderFullStatement(type) {
  if (!_curData) return;

  // Update toggle buttons
  document.querySelectorAll('#stmt-type-toggle .toggle-btn').forEach((b) => {
    b.classList.toggle('active', b.dataset.stmt === type);
  });

  const title = document.getElementById('full-stmt-title');
  const sub = document.getElementById('full-stmt-sub');
  const tbody = document.getElementById('full-stmt-tbody');
  if (!tbody) return;

  if (sub) {
    sub.textContent = (_curData.company_name || _tk || '') + ' · ' + (_curData.filing_info?.form_type || '') + ' ' + (_curData.filing_info?.filing_date || '');
  }

  // Try to use the richer statement arrays from backend
  const stmtKey = type === 'income' ? 'income_statement' : type === 'balance' ? 'balance_sheet' : 'cash_flow_statement';
  const stmtData = _curData[stmtKey];

  if (stmtData && stmtData.length > 0) {
    // Use backend statement array: [{label, value, prior_value, concept}, ...]
    if (title) title.textContent = type === 'income' ? 'Income Statement' : type === 'balance' ? 'Balance Sheet' : 'Cash Flow Statement';

    let h = '';
    for (const row of stmtData) {
      if (row.value == null && row.prior_value == null) continue;
      const change = calcChange(row.value, row.prior_value);
      const changeHtml = change != null
        ? '<span class="' + (change >= 0 ? 'positive' : 'negative') + '">' + (change >= 0 ? '+' : '') + change.toFixed(1) + '%</span>'
        : '—';
      // Bold totals / key lines
      const isBold = /^(total|net income|revenue|gross profit|operating income|free cash flow|stockholders)/i.test(row.label);
      const boldStyle = isBold ? ' style="font-weight:600;color:var(--text-primary)"' : '';
      h += '<tr>' +
        '<td class="stmt-label"' + boldStyle + '>' + esc(row.label) + '</td>' +
        '<td class="stmt-value">' + fmtN(row.value) + '</td>' +
        '<td class="stmt-value">' + fmtN(row.prior_value) + '</td>' +
        '<td class="stmt-change">' + changeHtml + '</td>' +
        '</tr>';
    }
    if (!h) h = '<tr><td colspan="4" class="text-center text-muted">No data in statement</td></tr>';
    tbody.innerHTML = h;
    return;
  }

  // Fallback: build from top-level metrics
  const m = _curData.metrics || {};
  const pm = _curData.prior_metrics || {};

  let rows = [];

  if (type === 'income') {
    if (title) title.textContent = 'Income Statement';
    rows = [
      { label: 'Revenue', current: m.revenue, prior: pm?.revenue, bold: true },
      { label: 'Cost of Revenue', current: m.cost_of_revenue, prior: pm?.cost_of_revenue },
      { label: 'Gross Profit', current: m.gross_profit, prior: pm?.gross_profit, bold: true },
      { label: 'Research & Development', current: m.rd_expense, prior: pm?.rd_expense },
      { label: 'SG&A Expenses', current: m.sga_expense, prior: pm?.sga_expense },
      { label: 'Operating Expenses', current: m.operating_expenses, prior: pm?.operating_expenses },
      { label: 'Operating Income', current: m.operating_income, prior: pm?.operating_income, bold: true },
      { label: 'Interest Expense', current: m.interest_expense, prior: pm?.interest_expense },
      { label: 'Other Income/Expense', current: m.other_income, prior: pm?.other_income },
      { label: 'Pretax Income', current: m.pretax_income, prior: pm?.pretax_income },
      { label: 'Income Tax', current: m.income_tax, prior: pm?.income_tax },
      { label: 'Net Income', current: m.net_income, prior: pm?.net_income, bold: true },
      { label: 'EPS (Diluted)', current: m.eps_diluted, prior: pm?.eps_diluted, fmt: 'eps' },
      { label: 'Shares Outstanding', current: m.shares_outstanding, prior: pm?.shares_outstanding, fmt: 'shares' },
    ];
  } else if (type === 'balance') {
    if (title) title.textContent = 'Balance Sheet';
    rows = [
      { label: 'Cash & Equivalents', current: m.cash, prior: pm?.cash },
      { label: 'Short-term Investments', current: m.short_term_investments, prior: pm?.short_term_investments },
      { label: 'Accounts Receivable', current: m.accounts_receivable, prior: pm?.accounts_receivable },
      { label: 'Inventory', current: m.inventory, prior: pm?.inventory },
      { label: 'Total Current Assets', current: m.current_assets, prior: pm?.current_assets, bold: true },
      { label: 'PP&E (Net)', current: m.ppe_net, prior: pm?.ppe_net },
      { label: 'Goodwill', current: m.goodwill, prior: pm?.goodwill },
      { label: 'Total Assets', current: m.total_assets, prior: pm?.total_assets, bold: true },
      { label: 'Accounts Payable', current: m.accounts_payable, prior: pm?.accounts_payable },
      { label: 'Short-term Debt', current: m.short_term_debt, prior: pm?.short_term_debt },
      { label: 'Total Current Liabilities', current: m.current_liabilities, prior: pm?.current_liabilities, bold: true },
      { label: 'Long-term Debt', current: m.long_term_debt, prior: pm?.long_term_debt },
      { label: 'Total Liabilities', current: m.total_liabilities, prior: pm?.total_liabilities, bold: true },
      { label: "Stockholders' Equity", current: m.stockholders_equity, prior: pm?.stockholders_equity, bold: true },
    ];
  } else if (type === 'cashflow') {
    if (title) title.textContent = 'Cash Flow Statement';
    rows = [
      { label: 'Net Income', current: m.net_income, prior: pm?.net_income },
      { label: 'Depreciation & Amortization', current: m.depreciation, prior: pm?.depreciation },
      { label: 'Cash from Operations', current: m.operating_cash_flow, prior: pm?.operating_cash_flow, bold: true },
      { label: 'Capital Expenditures', current: m.capex, prior: pm?.capex },
      { label: 'Cash from Investing', current: m.investing_cash_flow, prior: pm?.investing_cash_flow, bold: true },
      { label: 'Dividends Paid', current: m.dividends_paid, prior: pm?.dividends_paid },
      { label: 'Share Repurchases', current: m.share_repurchases, prior: pm?.share_repurchases },
      { label: 'Cash from Financing', current: m.financing_cash_flow, prior: pm?.financing_cash_flow, bold: true },
      { label: 'Free Cash Flow', current: m.free_cash_flow, prior: pm?.free_cash_flow, bold: true },
    ];
  }

  let h = '';
  for (const row of rows) {
    if (row.current == null && row.prior == null) continue;
    const change = calcChange(row.current, row.prior);
    const fmtCur = row.fmt === 'eps' ? (row.current != null ? '$' + row.current.toFixed(2) : '—') :
                   row.fmt === 'shares' ? (row.current != null ? (row.current / 1e6).toFixed(0) + 'M' : '—') :
                   fmtN(row.current);
    const fmtPr = row.fmt === 'eps' ? (row.prior != null ? '$' + row.prior.toFixed(2) : '—') :
                  row.fmt === 'shares' ? (row.prior != null ? (row.prior / 1e6).toFixed(0) + 'M' : '—') :
                  fmtN(row.prior);
    const changeHtml = change != null
      ? '<span class="' + (change >= 0 ? 'positive' : 'negative') + '">' + (change >= 0 ? '+' : '') + change.toFixed(1) + '%</span>'
      : '—';
    const boldStyle = row.bold ? ' style="font-weight:600;color:var(--text-primary)"' : '';
    h += '<tr>' +
      '<td class="stmt-label"' + boldStyle + '>' + row.label + '</td>' +
      '<td class="stmt-value">' + fmtCur + '</td>' +
      '<td class="stmt-value">' + fmtPr + '</td>' +
      '<td class="stmt-change">' + changeHtml + '</td>' +
      '</tr>';
  }

  if (!h) h = '<tr><td colspan="4" class="text-center text-muted">No data available for this statement</td></tr>';
  tbody.innerHTML = h;
}

/**
 * Comps view state: list of tickers to compare.
 */
let _compsTickers = [];

/**
 * Load comps view — auto-add current ticker + peers.
 */
async function loadCompsView() {
  if (_tk && !_compsTickers.includes(_tk)) {
    _compsTickers = [_tk];
  }
  // Auto-load suggested peers if we only have the target ticker
  if (_tk && _compsTickers.length <= 1) {
    try {
      const r = await fetch(API_BASE + '/api/peers/' + encodeURIComponent(_tk));
      if (r.ok) {
        const j = await r.json();
        const peers = j.peers || [];
        for (const p of peers) {
          if (!_compsTickers.includes(p)) _compsTickers.push(p);
        }
        // Show sector info in subtitle
        const sub = document.getElementById('comps-sub');
        if (sub && j.sector && j.sector !== 'unknown') {
          const sectorLabel = j.sector.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
          sub.textContent = sectorLabel + ' sector · Ranked by revenue proximity';
        }
      }
    } catch (e) {
      console.warn('[Comps] Failed to fetch peers:', e);
    }
  }
  renderCompsChips();
  if (_compsTickers.length > 0) fetchComps();
}

/**
 * Add a ticker to the comps list.
 */
function addCompTicker() {
  const inp = document.getElementById('comps-input');
  if (!inp) return;
  const tk = inp.value.trim().toUpperCase();
  if (!tk || _compsTickers.includes(tk)) { inp.value = ''; return; }
  _compsTickers.push(tk);
  inp.value = '';
  renderCompsChips();
  fetchComps();
}

/**
 * Remove a ticker from comps.
 */
function removeCompTicker(tk) {
  _compsTickers = _compsTickers.filter((t) => t !== tk);
  renderCompsChips();
  if (_compsTickers.length > 0) fetchComps();
  else document.getElementById('comps-tbody').innerHTML = '<tr><td colspan="7" class="text-center text-muted">Add tickers above to compare</td></tr>';
}

/**
 * Render comps ticker chips.
 */
function renderCompsChips() {
  const el = document.getElementById('comps-chips');
  if (!el) return;
  el.innerHTML = _compsTickers.map((t) =>
    '<span class="chip" style="cursor:default">' + esc(t) +
    ' <span style="cursor:pointer;margin-left:4px;opacity:0.6" onclick="removeCompTicker(\'' + esc(t) + '\')">&times;</span></span>'
  ).join('');
}

/**
 * Fetch comps data from API.
 */
async function fetchComps() {
  const tbody = document.getElementById('comps-tbody');
  if (!tbody) return;
  tbody.innerHTML = '<tr><td colspan="7" class="text-center"><div class="spinner" style="margin:8px auto"></div></td></tr>';

  try {
    const r = await fetch(API_BASE + '/api/comps', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tickers: _compsTickers }),
    });
    if (!r.ok) throw new Error('API error: ' + r.status);
    const j = await r.json();

    const rawResults = j.results || j.comparisons || [];
    if (!rawResults.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No comparison data returned</td></tr>';
      return;
    }

    let h = '';
    for (const r of rawResults) {
      // Backend wraps each result in {data: {...}, summary: "..."}
      const d = r.data || r;
      const m = d.metrics || {};
      const ra = d.ratios || {};
      const tk = d.ticker_or_cik || d.ticker || '';
      const name = d.company_name || '';
      h += '<tr>' +
        '<td><strong>' + esc(tk) + '</strong>' + (name ? '<br><span class="text-muted" style="font-size:11px">' + esc(name) + '</span>' : '') + '</td>' +
        '<td class="right">' + fmtN(m.revenue) + '</td>' +
        '<td class="right">' + fmtN(m.net_income) + '</td>' +
        '<td class="right">' + (m.gross_margin != null ? (m.gross_margin * 100).toFixed(1) + '%' : ra.gross_margin != null ? (ra.gross_margin * 100).toFixed(1) + '%' : '—') + '</td>' +
        '<td class="right">' + (m.net_margin != null ? (m.net_margin * 100).toFixed(1) + '%' : ra.net_margin != null ? (ra.net_margin * 100).toFixed(1) + '%' : '—') + '</td>' +
        '<td class="right">' + fmtN(m.total_assets) + '</td>' +
        '<td class="right">' + (m.eps_diluted != null ? '$' + m.eps_diluted.toFixed(2) : '—') + '</td>' +
        '</tr>';
    }
    tbody.innerHTML = h;
  } catch (e) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">Error: ' + esc(e.message) + '</td></tr>';
    console.error('[Comps Error]', e);
  }
}

// ========================================
// 10-K EXPLORER (Filing sections + AI Q&A)
// ========================================

/** Currently loaded explorer section text (for AI context) */
let _explorerSectionText = '';
let _explorerSectionName = '';

/**
 * Load the filings overview (All Filings tab).
 */
async function loadFilingsView() {
  // Default to overview tab
  loadExplorerSection('overview');
}

/**
 * Switch explorer section tab.
 */
function loadExplorerSection(section) {
  // Update tab active state
  document.querySelectorAll('.explorer-tab').forEach((t) => {
    t.classList.toggle('active', t.dataset.section === section);
  });

  const overview = document.getElementById('explorer-overview');
  const sectionPanel = document.getElementById('explorer-section');

  if (section === 'overview') {
    if (overview) overview.style.display = 'block';
    if (sectionPanel) sectionPanel.style.display = 'none';
    if (_tk) loadFilingsTable();
  } else {
    if (overview) overview.style.display = 'none';
    if (sectionPanel) sectionPanel.style.display = 'block';
    loadSectionContent(section);
  }

  lucide.createIcons();
}

/**
 * Load the filings table for overview.
 */
async function loadFilingsTable() {
  if (!_tk) return;
  const tbody = document.getElementById('filings-tbody');
  const sub = document.getElementById('filings-sub');
  if (sub) sub.textContent = _tk + ' — Recent SEC filings';
  if (!tbody) return;

  tbody.innerHTML = '<tr><td colspan="4" class="text-center"><div class="spinner" style="margin:8px auto"></div></td></tr>';

  try {
    const r = await fetch(API_BASE + '/api/filings/' + encodeURIComponent(_tk));
    if (!r.ok) throw new Error('API error: ' + r.status);
    const j = await r.json();
    const filings = j.filings || [];

    if (!filings.length) {
      tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No filings found</td></tr>';
      return;
    }

    let h = '';
    for (const f of filings) {
      h += '<tr>' +
        '<td><span class="ticker-badge" style="font-size:11px">' + esc(f.form_type) + '</span></td>' +
        '<td>' + esc(f.filing_date || '') + '</td>' +
        '<td style="color:var(--text-secondary)">' + esc(f.description || '') + '</td>' +
        '<td class="right"><button class="text-btn" onclick="viewFilingInExplorer(\'' + esc(f.accession) + '\',\'' + esc(f.form_type) + '\')">View</button></td>' +
        '</tr>';
    }
    tbody.innerHTML = h;
  } catch (e) {
    tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">Error: ' + esc(e.message) + '</td></tr>';
  }
}

/**
 * View a filing inline from the filings table.
 */
async function viewFilingInExplorer(accession, formType) {
  const overview = document.getElementById('explorer-overview');
  const sectionPanel = document.getElementById('explorer-section');
  if (overview) overview.style.display = 'none';
  if (sectionPanel) sectionPanel.style.display = 'block';

  // Clear tab active states
  document.querySelectorAll('.explorer-tab').forEach((t) => t.classList.remove('active'));

  const title = document.getElementById('explorer-section-title');
  const sub = document.getElementById('explorer-section-sub');
  const body = document.getElementById('explorer-section-body');
  if (title) title.textContent = formType + ' Filing';
  if (sub) sub.textContent = 'Loading...';
  if (body) body.innerHTML = '<div class="spinner" style="margin:20px auto"></div>';

  // Clear Q&A
  const qaMessages = document.getElementById('explorer-qa-messages');
  if (qaMessages) qaMessages.innerHTML = '';

  try {
    const r = await fetch(API_BASE + '/api/filing-text/' + encodeURIComponent(_tk) + '/full?accession=' + encodeURIComponent(accession) + '&form_type=' + encodeURIComponent(formType || (_isFpi ? '20-F' : '10-K')));
    if (!r.ok) throw new Error('API error: ' + r.status);
    const j = await r.json();

    if (sub) sub.textContent = (j.form_type || formType) + ' · Filed ' + (j.filing_date || '') + ' · ' + ((j.text_length || 0) / 1000).toFixed(0) + 'K chars';
    const text = j.text || 'No content available.';
    _explorerSectionText = text;
    _explorerSectionName = formType + ' filing';
    if (body) body.textContent = text;
  } catch (e) {
    if (body) body.innerHTML = '<p class="text-muted">Error: ' + esc(e.message) + '</p>';
  }
}

/** Section label map */
const SECTION_LABELS = {
  business: { title: 'Business Overview', item: 'Item 1', icon: 'building-2' },
  risk_factors: { title: 'Risk Factors', item: 'Item 1A', icon: 'alert-triangle' },
  mda: { title: "Management's Discussion & Analysis", item: 'Item 7', icon: 'bar-chart-3' },
  financial_statements: { title: 'Financial Statements & Notes', item: 'Item 8', icon: 'calculator' },
  executive_compensation: { title: 'Executive Compensation', item: 'Item 11', icon: 'users' },
  controls: { title: 'Controls & Procedures', item: 'Item 9A', icon: 'shield-check' },
};

/**
 * Load a specific 10-K section (business, risk_factors, mda, etc.)
 */
async function loadSectionContent(section) {
  if (!_tk) {
    const body = document.getElementById('explorer-section-body');
    if (body) body.innerHTML = '<p class="text-muted">Search for a company first</p>';
    return;
  }

  const info = SECTION_LABELS[section] || { title: section, item: '', icon: 'file-text' };
  const title = document.getElementById('explorer-section-title');
  const sub = document.getElementById('explorer-section-sub');
  const body = document.getElementById('explorer-section-body');

  if (title) title.textContent = info.title;
  if (sub) sub.textContent = _tk + ' · ' + info.item + ' · Loading...';
  if (body) body.innerHTML = '<div style="display:flex;align-items:center;gap:12px;padding:24px 0"><div class="spinner"></div><span class="text-muted">Extracting ' + info.title.toLowerCase() + ' from SEC filing...</span></div>';

  // Clear previous Q&A
  const qaMessages = document.getElementById('explorer-qa-messages');
  if (qaMessages) qaMessages.innerHTML = '';

  try {
    const accession = _curAcc || '';
    const formType = document.getElementById('sel-form')?.value || (_isFpi ? '20-F' : '10-K');
    const params = new URLSearchParams();
    if (accession) params.set('accession', accession);
    params.set('form_type', formType);
    let url = API_BASE + '/api/filing-text/' + encodeURIComponent(_tk) + '/' + encodeURIComponent(section);
    url += '?' + params.toString();

    const r = await fetch(url);
    if (!r.ok) throw new Error('API error: ' + r.status);
    const j = await r.json();

    if (j.error) {
      if (body) body.innerHTML = '<p class="text-muted">Error: ' + esc(j.error) + '</p>';
      return;
    }

    const text = j.text || 'No content found for this section.';
    const chars = text.length;
    if (sub) sub.textContent = _tk + ' · ' + (j.form_type || info.item) + ' · Filed ' + (j.filing_date || '') + ' · ' + (chars / 1000).toFixed(0) + 'K chars';

    _explorerSectionText = text;
    _explorerSectionName = info.title;

    if (body) body.textContent = text;
  } catch (e) {
    if (body) body.innerHTML = '<p class="text-muted">Error loading section: ' + esc(e.message) + '</p>';
  }
}

/**
 * Copy explorer section text to clipboard.
 */
function copyExplorerText() {
  if (_explorerSectionText) {
    navigator.clipboard.writeText(_explorerSectionText).then(() => {
      showToast('Section text copied to clipboard');
    });
  }
}

/**
 * Ask AI a question about the currently loaded explorer section.
 */
async function askExplorerAI() {
  const inp = document.getElementById('explorer-qa-inp');
  const msg = inp?.value?.trim();
  if (!msg || !_explorerSectionText) return;
  inp.value = '';

  const messagesEl = document.getElementById('explorer-qa-messages');
  if (!messagesEl) return;

  // Add user question
  messagesEl.innerHTML += '<div class="msg msg-user"><div class="msg-bubble">' + esc(msg) + '</div></div>';

  // Add loading
  messagesEl.innerHTML += '<div class="msg msg-assistant" id="explorer-qa-loading"><div class="msg-bubble"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div></div>';
  messagesEl.scrollTop = messagesEl.scrollHeight;

  try {
    // Send the section text as context, trimmed to ~30K chars
    const contextText = _explorerSectionText.substring(0, 30000);
    const body = {
      message: msg + '\n\nContext — ' + _explorerSectionName + ' section:\n' + contextText,
      ticker: _tk || '',
      context: _curData || {},
      history: [],
    };

    const r = await fetch(API_BASE + '/api/chatbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!r.ok) throw new Error('API error: ' + r.status);
    const j = await r.json();

    // Remove loading
    const loading = document.getElementById('explorer-qa-loading');
    if (loading) loading.remove();

    // Add AI response
    messagesEl.innerHTML += '<div class="msg msg-assistant"><div class="msg-bubble">' + md(j.answer || 'No response.') + '</div></div>';
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } catch (e) {
    const loading = document.getElementById('explorer-qa-loading');
    if (loading) loading.remove();
    messagesEl.innerHTML += '<div class="msg msg-assistant"><div class="msg-bubble">Error: ' + esc(e.message) + '</div></div>';
  }
}

// ========================================
// GEOGRAPHIC MAP (Leaflet)
// ========================================

/** Leaflet map instance */
let _geoMap = null;
let _geoLayers = [];

/** Map geographic segment labels to precise lat/lng coordinates */
/**
 * Sub-regions for broad geographic segments. Used for drill-down markers
 * on the map. Each entry has estimated % share of the parent region.
 */
const GEO_SUB_REGIONS = {
  'americas': [
    { lat: 40.7, lng: -74.0, label: 'New York', pct: 0.18 },
    { lat: 37.8, lng: -122.4, label: 'San Francisco', pct: 0.12 },
    { lat: 34.1, lng: -118.2, label: 'Los Angeles', pct: 0.10 },
    { lat: 41.9, lng: -87.6, label: 'Chicago', pct: 0.08 },
    { lat: 29.8, lng: -95.4, label: 'Houston', pct: 0.06 },
    { lat: 33.7, lng: -84.4, label: 'Atlanta', pct: 0.05 },
    { lat: 25.8, lng: -80.2, label: 'Miami', pct: 0.04 },
    { lat: 47.6, lng: -122.3, label: 'Seattle', pct: 0.04 },
    { lat: 43.7, lng: -79.4, label: 'Toronto', pct: 0.05 },
    { lat: 19.4, lng: -99.1, label: 'Mexico City', pct: 0.04 },
    { lat: -23.5, lng: -46.6, label: 'São Paulo', pct: 0.06 },
    { lat: 42.4, lng: -71.1, label: 'Boston', pct: 0.05 },
    { lat: 39.1, lng: -77.2, label: 'D.C. Metro', pct: 0.05 },
    { lat: 32.8, lng: -96.8, label: 'Dallas', pct: 0.04 },
    { lat: 38.9, lng: -104.8, label: 'Denver', pct: 0.04 },
  ],
  'united states': [
    { lat: 40.7, lng: -74.0, label: 'New York', pct: 0.20 },
    { lat: 37.8, lng: -122.4, label: 'San Francisco', pct: 0.14 },
    { lat: 34.1, lng: -118.2, label: 'Los Angeles', pct: 0.11 },
    { lat: 41.9, lng: -87.6, label: 'Chicago', pct: 0.09 },
    { lat: 29.8, lng: -95.4, label: 'Houston', pct: 0.07 },
    { lat: 33.7, lng: -84.4, label: 'Atlanta', pct: 0.06 },
    { lat: 47.6, lng: -122.3, label: 'Seattle', pct: 0.05 },
    { lat: 25.8, lng: -80.2, label: 'Miami', pct: 0.05 },
    { lat: 42.4, lng: -71.1, label: 'Boston', pct: 0.06 },
    { lat: 39.1, lng: -77.2, label: 'D.C. Metro', pct: 0.05 },
    { lat: 32.8, lng: -96.8, label: 'Dallas', pct: 0.06 },
    { lat: 38.9, lng: -104.8, label: 'Denver', pct: 0.04 },
    { lat: 36.2, lng: -115.2, label: 'Las Vegas', pct: 0.02 },
  ],
  'europe': [
    { lat: 51.5, lng: -0.1, label: 'London', pct: 0.22 },
    { lat: 52.5, lng: 13.4, label: 'Berlin', pct: 0.10 },
    { lat: 48.9, lng: 2.3, label: 'Paris', pct: 0.12 },
    { lat: 52.4, lng: 4.9, label: 'Amsterdam', pct: 0.06 },
    { lat: 48.2, lng: 16.4, label: 'Vienna', pct: 0.03 },
    { lat: 55.7, lng: 12.6, label: 'Copenhagen', pct: 0.04 },
    { lat: 59.3, lng: 18.1, label: 'Stockholm', pct: 0.05 },
    { lat: 53.3, lng: -6.3, label: 'Dublin', pct: 0.06 },
    { lat: 47.4, lng: 8.5, label: 'Zurich', pct: 0.05 },
    { lat: 41.9, lng: 12.5, label: 'Rome', pct: 0.04 },
    { lat: 40.4, lng: -3.7, label: 'Madrid', pct: 0.04 },
    { lat: 50.8, lng: 4.4, label: 'Brussels', pct: 0.03 },
    { lat: 48.1, lng: 11.6, label: 'Munich', pct: 0.06 },
    { lat: 52.2, lng: 21.0, label: 'Warsaw', pct: 0.03 },
    { lat: 60.2, lng: 24.9, label: 'Helsinki', pct: 0.03 },
    { lat: 28.6, lng: 77.2, label: 'New Delhi', pct: 0.04 },
  ],
  'greater china': [
    { lat: 39.9, lng: 116.4, label: 'Beijing', pct: 0.20 },
    { lat: 31.2, lng: 121.5, label: 'Shanghai', pct: 0.25 },
    { lat: 22.5, lng: 114.1, label: 'Shenzhen', pct: 0.15 },
    { lat: 23.1, lng: 113.3, label: 'Guangzhou', pct: 0.10 },
    { lat: 22.3, lng: 114.2, label: 'Hong Kong', pct: 0.15 },
    { lat: 25.0, lng: 121.5, label: 'Taipei', pct: 0.08 },
    { lat: 30.6, lng: 104.1, label: 'Chengdu', pct: 0.07 },
  ],
  'japan': [
    { lat: 35.7, lng: 139.7, label: 'Tokyo', pct: 0.45 },
    { lat: 34.7, lng: 135.5, label: 'Osaka', pct: 0.20 },
    { lat: 35.2, lng: 136.9, label: 'Nagoya', pct: 0.10 },
    { lat: 33.6, lng: 130.4, label: 'Fukuoka', pct: 0.08 },
    { lat: 43.1, lng: 141.3, label: 'Sapporo', pct: 0.05 },
    { lat: 35.0, lng: 135.8, label: 'Kyoto', pct: 0.05 },
    { lat: 34.4, lng: 132.5, label: 'Hiroshima', pct: 0.04 },
    { lat: 38.3, lng: 140.9, label: 'Sendai', pct: 0.03 },
  ],
  'rest of asia': [
    { lat: -33.9, lng: 151.2, label: 'Sydney', pct: 0.15 },
    { lat: 1.4, lng: 103.8, label: 'Singapore', pct: 0.15 },
    { lat: 37.6, lng: 127.0, label: 'Seoul', pct: 0.18 },
    { lat: 13.8, lng: 100.5, label: 'Bangkok', pct: 0.08 },
    { lat: -37.8, lng: 145.0, label: 'Melbourne', pct: 0.08 },
    { lat: 3.1, lng: 101.7, label: 'Kuala Lumpur', pct: 0.06 },
    { lat: 21.0, lng: 105.8, label: 'Hanoi', pct: 0.06 },
    { lat: -6.2, lng: 106.8, label: 'Jakarta', pct: 0.06 },
    { lat: 14.6, lng: 121.0, label: 'Manila', pct: 0.05 },
    { lat: 19.1, lng: 72.9, label: 'Mumbai', pct: 0.07 },
    { lat: -36.8, lng: 174.8, label: 'Auckland', pct: 0.03 },
    { lat: 12.9, lng: 77.6, label: 'Bangalore', pct: 0.03 },
  ],
  'rest of asia pacific': [
    { lat: -33.9, lng: 151.2, label: 'Sydney', pct: 0.15 },
    { lat: 1.4, lng: 103.8, label: 'Singapore', pct: 0.15 },
    { lat: 37.6, lng: 127.0, label: 'Seoul', pct: 0.18 },
    { lat: 13.8, lng: 100.5, label: 'Bangkok', pct: 0.08 },
    { lat: -37.8, lng: 145.0, label: 'Melbourne', pct: 0.08 },
    { lat: 3.1, lng: 101.7, label: 'Kuala Lumpur', pct: 0.06 },
    { lat: 21.0, lng: 105.8, label: 'Hanoi', pct: 0.06 },
    { lat: -6.2, lng: 106.8, label: 'Jakarta', pct: 0.06 },
    { lat: 14.6, lng: 121.0, label: 'Manila', pct: 0.05 },
    { lat: 19.1, lng: 72.9, label: 'Mumbai', pct: 0.07 },
    { lat: -36.8, lng: 174.8, label: 'Auckland', pct: 0.03 },
    { lat: 12.9, lng: 77.6, label: 'Bangalore', pct: 0.03 },
  ],
  'asia pacific': [
    { lat: 35.7, lng: 139.7, label: 'Tokyo', pct: 0.18 },
    { lat: 31.2, lng: 121.5, label: 'Shanghai', pct: 0.15 },
    { lat: 1.4, lng: 103.8, label: 'Singapore', pct: 0.10 },
    { lat: 37.6, lng: 127.0, label: 'Seoul', pct: 0.12 },
    { lat: -33.9, lng: 151.2, label: 'Sydney', pct: 0.10 },
    { lat: 22.3, lng: 114.2, label: 'Hong Kong', pct: 0.08 },
    { lat: 19.1, lng: 72.9, label: 'Mumbai', pct: 0.07 },
    { lat: 13.8, lng: 100.5, label: 'Bangkok', pct: 0.05 },
    { lat: 3.1, lng: 101.7, label: 'Kuala Lumpur', pct: 0.05 },
    { lat: -6.2, lng: 106.8, label: 'Jakarta', pct: 0.05 },
    { lat: 25.0, lng: 121.5, label: 'Taipei', pct: 0.05 },
  ],
  'international': [
    { lat: 51.5, lng: -0.1, label: 'London', pct: 0.15 },
    { lat: 48.9, lng: 2.3, label: 'Paris', pct: 0.08 },
    { lat: 35.7, lng: 139.7, label: 'Tokyo', pct: 0.10 },
    { lat: 31.2, lng: 121.5, label: 'Shanghai', pct: 0.10 },
    { lat: 52.5, lng: 13.4, label: 'Berlin', pct: 0.06 },
    { lat: -33.9, lng: 151.2, label: 'Sydney', pct: 0.05 },
    { lat: 37.6, lng: 127.0, label: 'Seoul', pct: 0.06 },
    { lat: 1.4, lng: 103.8, label: 'Singapore', pct: 0.05 },
    { lat: 43.7, lng: -79.4, label: 'Toronto', pct: 0.05 },
    { lat: 19.1, lng: 72.9, label: 'Mumbai', pct: 0.05 },
    { lat: -23.5, lng: -46.6, label: 'São Paulo', pct: 0.05 },
    { lat: 19.4, lng: -99.1, label: 'Mexico City', pct: 0.03 },
  ],
};

const GEO_REGIONS = {
  // ── Americas ──
  'americas': { lat: 20, lng: -90, label: 'Americas' },
  'north america': { lat: 40, lng: -100, label: 'North America' },
  'united states': { lat: 39, lng: -98, label: 'United States' },
  'us': { lat: 39, lng: -98, label: 'United States' },
  'u.s.': { lat: 39, lng: -98, label: 'United States' },
  'u.s': { lat: 39, lng: -98, label: 'United States' },
  'domestic': { lat: 39, lng: -98, label: 'United States' },
  'canada': { lat: 56, lng: -106, label: 'Canada' },
  'latin america': { lat: -10, lng: -60, label: 'Latin America' },
  'south america': { lat: -15, lng: -60, label: 'South America' },
  'central america': { lat: 14, lng: -87, label: 'Central America' },
  'brazil': { lat: -14, lng: -51, label: 'Brazil' },
  'mexico': { lat: 23, lng: -102, label: 'Mexico' },
  'argentina': { lat: -34, lng: -64, label: 'Argentina' },
  'colombia': { lat: 4, lng: -72, label: 'Colombia' },
  'chile': { lat: -33, lng: -71, label: 'Chile' },
  'peru': { lat: -12, lng: -77, label: 'Peru' },
  'caribbean': { lat: 18, lng: -72, label: 'Caribbean' },
  // ── Europe ──
  'europe': { lat: 50, lng: 10, label: 'Europe' },
  'emea': { lat: 40, lng: 20, label: 'EMEA' },
  'european union': { lat: 50, lng: 10, label: 'European Union' },
  'western europe': { lat: 48, lng: 5, label: 'Western Europe' },
  'eastern europe': { lat: 50, lng: 25, label: 'Eastern Europe' },
  'uk': { lat: 54, lng: -2, label: 'United Kingdom' },
  'united kingdom': { lat: 54, lng: -2, label: 'United Kingdom' },
  'great britain': { lat: 54, lng: -2, label: 'United Kingdom' },
  'germany': { lat: 51, lng: 10, label: 'Germany' },
  'france': { lat: 46, lng: 2, label: 'France' },
  'netherlands': { lat: 52, lng: 5, label: 'Netherlands' },
  'ireland': { lat: 53, lng: -8, label: 'Ireland' },
  'switzerland': { lat: 47, lng: 8, label: 'Switzerland' },
  'italy': { lat: 42, lng: 12, label: 'Italy' },
  'spain': { lat: 40, lng: -4, label: 'Spain' },
  'sweden': { lat: 62, lng: 15, label: 'Sweden' },
  'norway': { lat: 62, lng: 10, label: 'Norway' },
  'denmark': { lat: 56, lng: 10, label: 'Denmark' },
  'finland': { lat: 64, lng: 26, label: 'Finland' },
  'belgium': { lat: 51, lng: 4, label: 'Belgium' },
  'austria': { lat: 48, lng: 14, label: 'Austria' },
  'poland': { lat: 52, lng: 20, label: 'Poland' },
  'czech republic': { lat: 50, lng: 15, label: 'Czech Republic' },
  'portugal': { lat: 39, lng: -8, label: 'Portugal' },
  'greece': { lat: 39, lng: 22, label: 'Greece' },
  'russia': { lat: 56, lng: 38, label: 'Russia' },
  'luxembourg': { lat: 50, lng: 6, label: 'Luxembourg' },
  'nordics': { lat: 60, lng: 15, label: 'Nordics' },
  // ── Asia Pacific ──
  'asia pacific': { lat: 20, lng: 110, label: 'Asia Pacific' },
  'asia': { lat: 30, lng: 105, label: 'Asia' },
  'apac': { lat: 20, lng: 110, label: 'Asia Pacific' },
  'greater china': { lat: 35, lng: 105, label: 'Greater China' },
  'china': { lat: 35, lng: 105, label: 'China' },
  'mainland china': { lat: 35, lng: 105, label: 'China' },
  'hong kong': { lat: 22, lng: 114, label: 'Hong Kong' },
  'taiwan': { lat: 24, lng: 121, label: 'Taiwan' },
  'japan': { lat: 36, lng: 138, label: 'Japan' },
  'india': { lat: 20, lng: 77, label: 'India' },
  'korea': { lat: 36, lng: 128, label: 'South Korea' },
  'south korea': { lat: 36, lng: 128, label: 'South Korea' },
  'australia': { lat: -25, lng: 134, label: 'Australia' },
  'new zealand': { lat: -41, lng: 174, label: 'New Zealand' },
  'singapore': { lat: 1, lng: 104, label: 'Singapore' },
  'malaysia': { lat: 4, lng: 102, label: 'Malaysia' },
  'thailand': { lat: 15, lng: 101, label: 'Thailand' },
  'indonesia': { lat: -2, lng: 118, label: 'Indonesia' },
  'vietnam': { lat: 16, lng: 108, label: 'Vietnam' },
  'philippines': { lat: 13, lng: 122, label: 'Philippines' },
  'southeast asia': { lat: 5, lng: 110, label: 'Southeast Asia' },
  'rest of asia': { lat: 15, lng: 100, label: 'Rest of Asia' },
  'rest of asia pacific': { lat: 10, lng: 105, label: 'Rest of Asia Pacific' },
  // ── Middle East & Africa ──
  'middle east': { lat: 25, lng: 45, label: 'Middle East' },
  'middle east & africa': { lat: 20, lng: 40, label: 'Middle East & Africa' },
  'middle east and africa': { lat: 20, lng: 40, label: 'Middle East & Africa' },
  'africa': { lat: 5, lng: 25, label: 'Africa' },
  'south africa': { lat: -30, lng: 25, label: 'South Africa' },
  'israel': { lat: 31, lng: 35, label: 'Israel' },
  'saudi arabia': { lat: 24, lng: 45, label: 'Saudi Arabia' },
  'uae': { lat: 24, lng: 54, label: 'UAE' },
  'united arab emirates': { lat: 24, lng: 54, label: 'UAE' },
  'turkey': { lat: 39, lng: 35, label: 'Turkey' },
  'egypt': { lat: 27, lng: 30, label: 'Egypt' },
  'nigeria': { lat: 10, lng: 8, label: 'Nigeria' },
  'kenya': { lat: -1, lng: 37, label: 'Kenya' },
  // ── Generic ──
  'rest of world': { lat: 0, lng: 0, label: 'Rest of World' },
  'other': { lat: 0, lng: 30, label: 'Other' },
  'international': { lat: 20, lng: 30, label: 'International' },
  'other countries': { lat: 0, lng: 30, label: 'Other Countries' },
  'outside united states': { lat: 20, lng: 30, label: 'International' },
  'foreign': { lat: 20, lng: 30, label: 'International' },
  'all other': { lat: 0, lng: 30, label: 'Other' },
};

/**
 * Render geographic revenue map using Leaflet.
 */
function renderGeoMap(d) {
  const container = document.getElementById('geo-map');
  const legendEl = document.getElementById('geo-legend');
  if (!container) return;

  let geoSegs = d.segments?.geographic_segments || [];
  const geoSub = document.getElementById('geo-subtitle');

  // If no XBRL geo segments, try fetching from filing text parser
  if (!geoSegs.length && _tk) {
    // Show loading state
    container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-tertiary);font-size:12px;gap:8px"><div class="spinner" style="width:16px;height:16px"></div>Loading geographic data...</div>';

    fetch(API_BASE + '/api/geo-revenue/' + encodeURIComponent(_tk))
      .then(r => r.json())
      .then(j => {
        if (j.geographic_segments && j.geographic_segments.length > 0) {
          // Re-render with real data
          d._geoFromApi = j.geographic_segments;
          d._geoSource = j.source;
          _renderGeoMapWithData(container, legendEl, geoSub, j.geographic_segments, j.source);
        } else {
          container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-tertiary);font-size:12px">No geographic revenue data found in filing</div>';
          if (legendEl) legendEl.innerHTML = '';
          if (geoSub) geoSub.textContent = 'No geographic breakdown available';
        }
      })
      .catch(() => {
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-tertiary);font-size:12px">Geographic data unavailable</div>';
      });
    return;
  }

  const source = geoSegs.length > 0 ? 'xbrl' : 'none';
  _renderGeoMapWithData(container, legendEl, geoSub, geoSegs, source);
}

/**
 * Internal: render the map with resolved geo segments.
 */
function _renderGeoMapWithData(container, legendEl, geoSub, geoSegs, source) {
  if (geoSub) {
    const labels = {
      'fmp': 'Revenue by region (Financial Modeling Prep)',
      'xbrl': 'Revenue by region (XBRL)',
      'filing_text': 'Revenue by region (from 10-K text)',
      'none': 'No geographic breakdown available',
    };
    geoSub.textContent = labels[source] || labels.xbrl;
  }

  if (!geoSegs.length) {
    container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-tertiary);font-size:12px">No geographic data available</div>';
    if (legendEl) legendEl.innerHTML = '';
    return;
  }

  // Initialize map if needed
  if (!_geoMap) {
    _geoMap = L.map('geo-map', {
      center: [25, 10],
      zoom: 1,
      minZoom: 1,
      maxZoom: 6,
      zoomControl: true,
      attributionControl: false,
      worldCopyJump: true,
    });

    // Dark tile layer with labels
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      subdomains: 'abcd',
      maxZoom: 19,
    }).addTo(_geoMap);
  }

  // Clear old layers
  _geoLayers.forEach((l) => _geoMap.removeLayer(l));
  _geoLayers = [];

  // Find max value for scaling
  const maxVal = Math.max(...geoSegs.map((s) => s.value || 0));
  const total = geoSegs.reduce((sum, s) => sum + (s.value || 0), 0);

  // Place one circle per region — sized by revenue share
  geoSegs.forEach((seg, i) => {
    const rawName = (seg.segment || seg.name || '').trim();
    const name = rawName.toLowerCase();
    let coords = GEO_REGIONS[name];
    if (!coords) {
      const match = Object.keys(GEO_REGIONS).find((k) => name.includes(k) || k.includes(name));
      if (match) coords = GEO_REGIONS[match];
    }
    // Fallback: "Other" / "Rest of World" → place in Atlantic so it's not on Africa
    if (!coords) {
      coords = { lat: 15, lng: -30, label: rawName };
    }

    const pct = total > 0 ? (seg.value / total * 100) : 0;
    const color = CHART_COLORS.palette[i % CHART_COLORS.palette.length];
    const radius = Math.max(14, Math.sqrt(seg.value / maxVal) * 45);

    // Glow ring
    const glow = L.circleMarker([coords.lat, coords.lng], {
      radius: radius + 6,
      fillColor: color, fillOpacity: 0.12,
      color: color, weight: 0, opacity: 0,
    }).addTo(_geoMap);
    _geoLayers.push(glow);

    // Main circle
    const circle = L.circleMarker([coords.lat, coords.lng], {
      radius: radius,
      fillColor: color, fillOpacity: 0.6,
      color: color, weight: 2, opacity: 0.9,
    }).addTo(_geoMap);

    circle.bindTooltip(
      '<div class="geo-tooltip">' +
      '<div class="region-name">' + esc(coords.label || rawName) + '</div>' +
      '<div class="region-value">' + fmtN(seg.value) + '</div>' +
      '<div class="region-pct">' + pct.toFixed(1) + '% of total</div>' +
      '</div>',
      { className: 'geo-tooltip-wrapper', direction: 'top' }
    );

    circle.on('mouseover', function() { this.setStyle({ fillOpacity: 0.85, weight: 3 }); });
    circle.on('mouseout', function() { this.setStyle({ fillOpacity: 0.6, weight: 2 }); });

    _geoLayers.push(circle);
  });

  // Render legend
  if (legendEl) {
    legendEl.innerHTML = geoSegs.map((s, i) => {
      const pct = total > 0 ? (s.value / total * 100).toFixed(1) : '0';
      return '<div class="seg-item">' +
        '<span class="seg-dot" style="background:' + CHART_COLORS.palette[i % CHART_COLORS.palette.length] + '"></span>' +
        '<span class="seg-name">' + esc(s.segment || s.name) + '</span>' +
        '<span class="seg-pct">' + fmtN(s.value) + ' (' + pct + '%)</span>' +
        '</div>';
    }).join('');
  }

  // Fit bounds to show all markers
  if (_geoLayers.length > 0) {
    const group = L.featureGroup(_geoLayers);
    _geoMap.fitBounds(group.getBounds().pad(0.3));
  }
}

/**
 * Generate AI insights for the currently loaded company.
 */
async function generateInsights() {
  if (!_tk || !_curData) {
    showError('Load a company first');
    return;
  }

  const el = document.getElementById('insights-content');
  if (!el) return;
  el.innerHTML = '<div style="display:flex;align-items:center;gap:12px;padding:16px 0"><div class="spinner"></div><span class="text-muted">Generating AI insights...</span></div>';

  try {
    const body = {
      message: 'Give me a comprehensive executive summary and key insights for ' + _tk + '. Cover: financial performance, margins, growth trends, strengths, risks, and outlook.',
      ticker: _tk,
      context: _curData,
      history: [],
    };

    const r = await fetch(API_BASE + '/api/chatbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!r.ok) throw new Error('API error: ' + r.status);
    const j = await r.json();

    const rendered = md(j.answer || 'No insights generated.');
    el.innerHTML = '<div class="insights-body">' + rendered + '</div>';
    // Style tables inside insights
    el.querySelectorAll('table').forEach(t => {
      t.classList.add('data-table');
      const wrap = document.createElement('div');
      wrap.className = 'table-wrap';
      t.parentNode.insertBefore(wrap, t);
      wrap.appendChild(t);
    });
  } catch (e) {
    el.innerHTML = '<p class="text-muted">Error: ' + esc(e.message) + '</p>';
    console.error('[Insights Error]', e);
  }
}

// ========================================
// 14. UTILITIES & HELPERS
// ========================================

/**
 * Format a number as currency with K/M/B/T suffix.
 */
function fmtN(n) {
  if (n == null || isNaN(n) || !isFinite(n)) return '—';
  const abs = Math.abs(n);
  const sign = n < 0 ? '-' : '';
  if (abs >= 1e12) return sign + '$' + (abs / 1e12).toFixed(1) + 'T';
  if (abs >= 1e9) return sign + '$' + (abs / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6) return sign + '$' + (abs / 1e6).toFixed(1) + 'M';
  if (abs >= 1e3) return sign + '$' + (abs / 1e3).toFixed(1) + 'K';
  return sign + '$' + abs.toFixed(0);
}

/**
 * Calculate percentage change between two values.
 */
function calcChange(cur, prev) {
  if (cur == null || prev == null || prev === 0) return null;
  return ((cur - prev) / Math.abs(prev)) * 100;
}

/**
 * Format a date string (YYYY-MM-DD format) as MM/DD/YYYY or MM/DD.
 */
function shortDate(d) {
  if (!d) return '';
  const parts = d.slice(0, 10).split('-');
  if (parts.length === 3) {
    return parts[1] + '/' + parts[2];
  }
  return d.slice(0, 10);
}

/**
 * HTML escape a string (prevent XSS).
 */
function esc(s) {
  if (s == null) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

/**
 * Simple markdown to HTML converter.
 * Supports **bold**, *italic*, [link](url), and line breaks.
 */
function md(s) {
  if (!s) return '';

  // Use marked.js for full markdown rendering (headers, tables, lists, code)
  if (typeof marked !== 'undefined' && marked.parse) {
    try {
      return marked.parse(s, { breaks: true, gfm: true });
    } catch (e) {
      console.warn('[md] marked.parse error, falling back:', e);
    }
  }

  // Fallback: basic markdown
  let h = esc(s);
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
  h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  h = h.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
  return h;
}

/**
 * Build a summary message for the chat when data is loaded.
 */
function buildLoadedSummary(d) {
  const m = d.metrics || {};
  const fi = d.filing_info || {};
  
  let msg = '**' + (d.company_name || _tk) + '** loaded\n\n';
  
  if (fi.form_type) {
    msg += '- Filing: **' + fi.form_type + '** filed ' + (fi.filing_date || '') + '\n';
  }
  if (m.revenue != null) {
    msg += '- Revenue: **' + fmtN(m.revenue) + '**\n';
  }
  if (m.net_income != null) {
    msg += '- Net Income: **' + fmtN(m.net_income) + '**\n';
  }
  
  msg += '\n_Ask me anything about this data_';
  
  return msg;
}

/**
 * Show an error message to the user.
 */
function showError(msg) {
  const toast = document.createElement('div');
  toast.className = 'toast toast-error';
  toast.innerHTML = '<i data-lucide="alert-circle"></i><span>' + esc(msg) + '</span>';
  document.body.appendChild(toast);
  lucide.createIcons();
  setTimeout(() => { toast.remove(); }, 5000);
}

/**
 * Show a neutral toast message.
 */
function showToast(msg) {
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = '<i data-lucide="check"></i><span>' + esc(msg) + '</span>';
  document.body.appendChild(toast);
  lucide.createIcons();
  setTimeout(() => { toast.remove(); }, 3000);
}

/**
 * Navigate to the full statement view.
 */
function showFullStatement() {
  switchView('statements');
}

/**
 * Set up keyboard shortcuts:
 * - "/" to focus search
 * - "Esc" to close chat panel
 */
function initKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    // "/" focuses search (but not if in an input)
    if (e.key === '/' && !e.target.closest('input,textarea')) {
      e.preventDefault();
      document.getElementById('search-input')?.focus();
    }
    
    // "Esc" closes chat panel
    if (e.key === 'Escape') {
      const panel = document.getElementById('chat-panel');
      if (panel && panel.classList.contains('open')) {
        toggleChat();
      }
    }
  });
}

/**
 * Export current dashboard data as JSON.
 */
function exportData() {
  if (!_curData) {
    showError('No data to export');
    return;
  }
  
  const data = {
    ticker: _tk,
    company_name: _curData.company_name,
    filing_info: _curData.filing_info,
    metrics: _curData.metrics,
    ratios: _curData.ratios,
    peers: _curData.peers,
    export_date: new Date().toISOString(),
  };
  
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = (_tk || 'fineas') + '-' + new Date().toISOString().slice(0, 10) + '.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ========================================
// RESEARCH FOLDERS (localStorage persistence)
// ========================================

/**
 * Save a research folder entry when a company is loaded.
 */
function saveResearch(ticker, companyName) {
  if (!ticker) return;
  const key = 'fineas-research';
  let items = [];
  try { items = JSON.parse(localStorage.getItem(key)) || []; } catch(e) { items = []; }

  const tk = ticker.toUpperCase();
  const existing = items.find(i => i.ticker === tk);

  if (existing) {
    // Update name and bump to top
    existing.name = companyName || tk;
    existing.updated = new Date().toISOString();
    items = items.filter(i => i.ticker !== tk);
    items.unshift(existing);
  } else {
    // New entry
    items.unshift({
      ticker: tk,
      name: companyName || tk,
      date: new Date().toISOString().slice(0, 10),
      updated: new Date().toISOString(),
      queries: [],
    });
  }

  // Keep max 20
  if (items.length > 20) items = items.slice(0, 20);

  localStorage.setItem(key, JSON.stringify(items));
  loadResearchFolders();
}

/**
 * Add a research query to a company's research entry.
 */
function addResearchQuery(ticker, question) {
  if (!ticker || !question) return;
  const key = 'fineas-research';
  let items = [];
  try { items = JSON.parse(localStorage.getItem(key)) || []; } catch(e) { items = []; }

  const tk = ticker.toUpperCase();
  const entry = items.find(i => i.ticker === tk);
  if (!entry) return; // Company not in research yet

  // Initialize queries array if missing (legacy entries)
  if (!entry.queries) entry.queries = [];

  // Avoid duplicate consecutive questions
  const last = entry.queries[entry.queries.length - 1];
  if (last && last.question === question) return;

  entry.queries.push({
    question: question,
    timestamp: new Date().toISOString(),
  });

  // Keep max 50 queries per company
  if (entry.queries.length > 50) entry.queries = entry.queries.slice(-50);

  entry.updated = new Date().toISOString();

  // Move to top (most recently updated)
  items = items.filter(i => i.ticker !== tk);
  items.unshift(entry);

  localStorage.setItem(key, JSON.stringify(items));
  loadResearchFolders();
}

/**
 * Format a date string nicely (e.g. "Mar 18").
 */
function fmtResearchDate(dateStr) {
  if (!dateStr) return '';
  try {
    const d = new Date(dateStr);
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    return months[d.getMonth()] + ' ' + d.getDate();
  } catch(e) { return dateStr; }
}

/**
 * Toggle expand/collapse of a research item's queries.
 */
function toggleResearchExpand(ticker, event) {
  event.stopPropagation();
  const item = document.querySelector('.research-item[data-ticker="' + ticker + '"]');
  const queries = document.querySelector('.research-queries[data-ticker="' + ticker + '"]');
  if (!item || !queries) return;
  item.classList.toggle('expanded');
  queries.classList.toggle('expanded');
}

/**
 * Re-ask a research query from history (via data-question attribute).
 */
function initResearchQueryClicks() {
  document.getElementById('research-folders')?.addEventListener('click', (e) => {
    const queryEl = e.target.closest('.research-query');
    if (!queryEl) return;
    e.stopPropagation();
    const question = queryEl.getAttribute('data-question');
    if (!question) return;
    const inp = document.getElementById('chat-inp');
    if (inp) inp.value = question;
    // Open chat if not open
    const panel = document.getElementById('chat-panel');
    if (panel && !panel.classList.contains('open')) toggleChat();
    sendCb();
  });
}

/**
 * Load and render research folder items in sidebar.
 */
function loadResearchFolders() {
  const container = document.getElementById('research-folders');
  if (!container) return;

  let items = [];
  try { items = JSON.parse(localStorage.getItem('fineas-research')) || []; } catch(e) { items = []; }

  if (!items.length) {
    container.innerHTML = '<div style="padding:4px 12px;font-size:11px;color:var(--text-muted)">No research yet</div>';
    return;
  }

  // Sort by most recently updated
  items.sort((a, b) => {
    const da = a.updated || a.date || '';
    const db = b.updated || b.date || '';
    return db.localeCompare(da);
  });

  container.innerHTML = items.map(item => {
    const queries = item.queries || [];
    const qCount = queries.length;
    const dateLabel = fmtResearchDate(item.date);
    const qLabel = qCount ? qCount + (qCount === 1 ? ' question' : ' questions') : '';

    let html = `
    <div class="research-item" data-ticker="${esc(item.ticker)}" onclick="q('${esc(item.ticker)}')" title="${esc(item.name)}">
      <svg class="research-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
      </svg>
      <div class="research-info">
        <div class="research-ticker">${esc(item.ticker)} <span style="font-weight:400;color:var(--text-muted);font-family:var(--font-ui);font-size:9px">${esc(dateLabel)}</span></div>
        <div class="research-name">${esc(item.name)}</div>
      </div>`;

    if (qCount > 0) {
      html += `<span class="query-count" onclick="toggleResearchExpand('${esc(item.ticker)}', event)">${esc(qLabel)}</span>`;
    }

    html += `
      <button class="research-delete" onclick="event.stopPropagation(); deleteResearch('${esc(item.ticker)}')" title="Remove">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>`;

    // Expandable query list
    if (qCount > 0) {
      html += `<div class="research-queries" data-ticker="${esc(item.ticker)}">`;
      // Show most recent queries first, max 10
      const recentQueries = queries.slice(-10).reverse();
      recentQueries.forEach(q => {
        html += `<div class="research-query" data-question="${esc(q.question)}" title="${esc(q.question)}">${esc(q.question)}</div>`;
      });
      html += `</div>`;
    }

    return html;
  }).join('');
}

/**
 * Remove a research folder entry.
 */
function deleteResearch(ticker) {
  const key = 'fineas-research';
  let items = [];
  try { items = JSON.parse(localStorage.getItem(key)) || []; } catch(e) { items = []; }
  items = items.filter(i => i.ticker !== ticker.toUpperCase());
  localStorage.setItem(key, JSON.stringify(items));
  loadResearchFolders();
}

/**
 * Refresh current data by re-querying the API.
 */
function refreshData() {
  if (_tk) {
    send(_tk);
  } else {
    showError('No ticker selected');
  }
}

// ============================================================
// End of app.js
// ============================================================
