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

/** Current main view ('dashboard', 'compare', 'filing') */
let _activeView = 'dashboard';

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
  
  // Log initialization complete
  console.log('[Fineas.ai] Application initialized');
});

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
      const r = await fetch('/api/search?q=' + encodeURIComponent(val));
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
  
  // Chat toggle button
  const chatToggle = document.getElementById('chat-toggle');
  if (chatToggle) {
    chatToggle.addEventListener('click', toggleChat);
  }
  
  // Theme toggle
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', toggleTheme);
  }
}

/**
 * Switch to a different view (dashboard, compare, filing, etc).
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
  
  // Hide all view panels, show active one
  document.querySelectorAll('.view-panel').forEach((p) => p.style.display = 'none');
  const panel = document.getElementById('panel-' + view);
  if (panel) panel.style.display = 'block';
  
  // For dashboard, make sure dashboard content is visible
  if (view === 'dashboard') {
    const dashboard = document.getElementById('dashboard');
    if (dashboard) dashboard.style.display = 'block';
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
    const r = await fetch('/api/chat', {
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
    
    // Update ticker
    if (tk) _tk = tk.toUpperCase();
    
    // Store current data
    _curData = d;
    if (d.filing_info) {
      _curAcc = d.filing_info.accession_number || null;
    }
    
    // Update company pill in header
    updateCompanyPill(d);
    
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
    // Handle peer/sector comparison
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
  
  // Focus on chat panel open
  const chatToggle = document.getElementById('chat-toggle');
  if (chatToggle) {
    chatToggle.addEventListener('click', () => {
      setTimeout(() => {
        if (inp && document.getElementById('chat-panel')?.classList.contains('open')) {
          inp.focus();
        }
      }, 100);
    });
  }
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
  
  // Show typing indicator
  addChatLoading();
  
  try {
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
    
    const r = await fetch('/api/chatbot', {
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
  
  // 1. Render KPI cards
  renderKPIs(m, r, pm);
  
  // 2. Render charts with slight delay to allow DOM rendering
  setTimeout(() => {
    renderRevenueChart(d);
    renderSegmentChart(d);
    renderMarginChart(d);
  }, 100);
  
  // 3. Render peer comparison table
  renderPeerTable(d);
  
  // 4. Render provenance/validation bar
  renderProvenance(d);
  
  // 5. Render financial statement preview
  renderStatementPreview(d);
}

/**
 * Render KPI cards in a grid.
 */
function renderKPIs(m, r, pm) {
  const grid = document.getElementById('kpi-grid');
  if (!grid) return;
  
  // Define KPIs: label, value, prior value, icon, color
  const kpis = [
    {
      label: 'Revenue',
      value: m.revenue,
      prior: pm?.revenue,
      icon: 'dollar-sign',
      color: 'brand',
    },
    {
      label: 'Net Income',
      value: m.net_income,
      prior: pm?.net_income,
      icon: 'trending-up',
      color: 'emerald',
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
      label: 'Total Assets',
      value: m.total_assets,
      prior: pm?.total_assets,
      icon: 'building-2',
      color: 'amber',
    },
    {
      label: 'EPS',
      value: m.eps_diluted,
      prior: pm?.eps_diluted,
      fmt: 'eps',
      icon: 'users',
      color: 'rose',
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
          : fmtN(kpi.value);
    
    h += buildKpiCard(kpi.label, fmtVal, change, kpi.icon, kpi.color, i);
  });
  
  grid.innerHTML = h;
  lucide.createIcons();
}

/**
 * Build a single KPI card HTML.
 */
function buildKpiCard(label, value, change, icon, color, index) {
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
  
  return (
    '<div class="kpi-card animate-in" style="animation-delay:' +
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
    '</div>'
  );
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
 */
function renderRevenueChart(d) {
  // Destroy existing chart
  if (_charts.revenue) _charts.revenue.destroy();
  
  const ctx = document.getElementById('revenue-chart');
  if (!ctx) return;
  
  const m = d.metrics || {};
  
  // For single-period display, show current year metrics
  const labels = [d.fiscal_year_end || d.fiscal_year || 'Current'];
  const revenue = [m.revenue ? m.revenue / 1e9 : 0];
  const netIncome = [m.net_income ? m.net_income / 1e9 : 0];
  const fcf = [m.free_cash_flow ? m.free_cash_flow / 1e9 : 0];
  
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
}

/**
 * Render segment revenue breakdown (doughnut chart).
 */
function renderSegmentChart(d) {
  // Destroy existing chart
  if (_charts.segment) _charts.segment.destroy();
  
  const ctx = document.getElementById('segment-chart');
  if (!ctx) return;
  
  const segments = d.segments?.revenue_segments || [];
  if (!segments.length) {
    if (ctx.parentElement) ctx.parentElement.style.display = 'none';
    return;
  }
  
  if (ctx.parentElement) ctx.parentElement.style.display = 'block';
  
  _charts.segment = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: segments.map((s) => s.segment || s.name),
      datasets: [
        {
          data: segments.map((s) => s.value || 0),
          backgroundColor: CHART_COLORS.palette,
          borderWidth: 3,
          borderColor: 'var(--bg-secondary)',
          hoverOffset: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '62%',
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1e293b',
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          borderColor: '#334155',
          borderWidth: 1,
        },
      },
    },
  });
  
  // Build custom legend
  const legendEl = document.getElementById('segment-legend');
  if (legendEl) {
    const total = segments.reduce((sum, s) => sum + (s.value || 0), 0);
    legendEl.innerHTML = segments
      .map((s, i) => {
        const pct = total > 0 ? ((s.value / total) * 100).toFixed(1) : '0';
        return (
          '<div class="seg-item">' +
          '<span class="seg-dot" style="background:' +
          CHART_COLORS.palette[i % CHART_COLORS.palette.length] +
          '"></span>' +
          '<span class="seg-name">' +
          esc(s.segment || s.name) +
          '</span>' +
          '<span class="seg-pct">' +
          pct +
          '%</span>' +
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
  
  el.innerHTML =
    '<div class="prov-badges">' +
    validationStatus +
    '<span class="prov-badge neutral"><i data-lucide="database"></i>Source: ' +
    esc(fi.form_type || '10-K') +
    ' filed ' +
    esc(fi.filing_date || '') +
    '</span>' +
    '<span class="prov-badge neutral"><i data-lucide="clock"></i>XBRL 4-pass extraction</span>' +
    '<span class="prov-badge brand"><i data-lucide="layers"></i>Industry: ' +
    esc(d.industry_class || 'standard') +
    '</span>' +
    '</div>';
  
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
    const r = await fetch('/api/filings/' + encodeURIComponent(tk));
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
    h +=
      '<option value="' +
      esc(f.accession) +
      '|' +
      esc(f.form_type) +
      '"' +
      selected +
      '>' +
      esc(f.form_type) +
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
    const r = await fetch('/api/load-filing', {
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
  const content = document.getElementById('content');
  if (!content) return;
  
  const comps = j.data?.comparisons || [];
  if (!comps.length) {
    content.innerHTML = '<p>No comparison data available.</p>';
    return;
  }
  
  let h = '<div class="comparison-grid">';
  for (const c of comps) {
    h +=
      '<div class="comp-card">' +
      '<div class="comp-ticker">' +
      esc(c.ticker) +
      '</div>' +
      '<div class="comp-metric">Revenue: ' +
      fmtN(c.revenue) +
      '</div>' +
      '<div class="comp-metric">Margin: ' +
      (c.net_margin != null ? (c.net_margin * 100).toFixed(1) + '%' : '—') +
      '</div>' +
      '</div>';
  }
  h += '</div>';
  
  content.innerHTML = h;
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
// 13. UTILITIES & HELPERS
// ========================================

/**
 * Format a number as currency with K/M/B/T suffix.
 */
function fmtN(n) {
  if (n == null) return '—';
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
  
  // Escape HTML first
  let h = esc(s);
  
  // Convert markdown
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
  h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  
  // Line breaks
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
  // Show toast notification
  const toast = document.createElement('div');
  toast.className = 'toast toast-error';
  toast.innerHTML = '<i data-lucide="alert-circle"></i><span>' + esc(msg) + '</span>';
  document.body.appendChild(toast);
  
  lucide.createIcons();
  
  // Auto-remove after 5 seconds
  setTimeout(() => {
    toast.remove();
  }, 5000);
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
