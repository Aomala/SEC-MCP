import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadarAxis,
  PieChart, Pie, Cell, ComposedChart
} from "recharts";
import {
  Search, TrendingUp, BarChart3, FileText, MessageSquare,
  ChevronDown, ChevronRight, ArrowUpRight, ArrowDownRight,
  Building2, DollarSign, Activity, PieChart as PieChartIcon,
  Layers, GitCompare, AlertTriangle, Download, Settings,
  Moon, Sun, Bell, Filter, RefreshCw, ExternalLink,
  Zap, Shield, Globe, BookOpen, Users, Clock, Star,
  ChevronLeft, X, Plus, Minus, Eye, Code, Database
} from "lucide-react";

// ============================================================
// COLOR PALETTE & THEME
// ============================================================
const COLORS = {
  brand: "#6366f1",        // indigo-500 — primary accent
  brandLight: "#818cf8",   // indigo-400
  brandDark: "#4f46e5",    // indigo-600
  success: "#10b981",      // emerald-500
  danger: "#ef4444",       // red-500
  warning: "#f59e0b",      // amber-500
  info: "#3b82f6",         // blue-500
  chart: ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"],
};

// ============================================================
// MOCK DATA — Simulates what the backend would return
// ============================================================
const MOCK_FINANCIALS = {
  ticker: "AAPL",
  company: "Apple Inc.",
  sic: "3571",
  industry: "standard",
  exchange: "NASDAQ",
  marketCap: 3420000000000,
  price: 227.48,
  priceChange: 2.34,
  priceChangePct: 1.04,
  metrics: {
    revenue: 391035000000,
    gross_profit: 180683000000,
    operating_income: 123216000000,
    net_income: 93736000000,
    ebitda: 134660000000,
    total_assets: 364980000000,
    total_liabilities: 308030000000,
    total_equity: 56950000000,
    cash: 29965000000,
    free_cash_flow: 111443000000,
    eps_diluted: 6.08,
    shares_outstanding: 15408000000,
  },
  ratios: {
    gross_margin: 46.2,
    operating_margin: 31.5,
    net_margin: 24.0,
    roa: 25.7,
    roe: 164.6,
    current_ratio: 0.99,
    debt_to_equity: 5.41,
    fcf_margin: 28.5,
  },
  segments: [
    { name: "iPhone", value: 200583000000, pct: 51.3 },
    { name: "Services", value: 96169000000, pct: 24.6 },
    { name: "Mac", value: 29984000000, pct: 7.7 },
    { name: "iPad", value: 28300000000, pct: 7.2 },
    { name: "Wearables", value: 36999000000, pct: 9.5 },
  ],
  historical: [
    { year: "FY2020", revenue: 274515, netIncome: 57411, fcf: 73365, margin: 20.9 },
    { year: "FY2021", revenue: 365817, netIncome: 94680, fcf: 92953, margin: 25.9 },
    { year: "FY2022", revenue: 394328, netIncome: 99803, fcf: 111443, margin: 25.3 },
    { year: "FY2023", revenue: 383285, netIncome: 96995, fcf: 110543, margin: 25.3 },
    { year: "FY2024", revenue: 391035, netIncome: 93736, fcf: 111443, margin: 24.0 },
  ],
  peers: [
    { ticker: "MSFT", name: "Microsoft", revenue: 245122, margin: 36.4, pe: 34.2 },
    { ticker: "GOOG", name: "Alphabet", revenue: 307394, margin: 27.6, pe: 22.8 },
    { ticker: "AMZN", name: "Amazon", revenue: 574785, margin: 8.0, pe: 42.1 },
    { ticker: "META", name: "Meta", revenue: 134902, margin: 35.6, pe: 24.5 },
  ],
  validation: [
    { rule: "balance_sheet_equation", severity: "info", message: "Assets = L + E within 0.2%" },
  ],
};

const MOCK_CHAT = [
  { role: "user", text: "Analyze Apple's financials for FY2024" },
  {
    role: "assistant",
    text: `**Apple Inc. (AAPL) — FY2024 Executive Summary**

Revenue came in at **$391B**, up 2% YoY after a rare decline in FY2023. Services hit a record **$96.2B** (24.6% of revenue), now the highest-margin segment. Net income was **$93.7B** with a 24% net margin — world-class profitability.

**Key Strengths:**
- Free cash flow of $111.4B (28.5% FCF margin)
- Services growing at 2x hardware rate
- ROE of 164.6% reflects capital-light model

**Watch Items:**
- iPhone still 51% of revenue — concentration risk
- Debt-to-equity at 5.41x (deliberate leverage strategy)
- China revenue declining amid geopolitical tensions`,
    sources: ["10-K FY2024 (000032019324000123)", "XBRL Facts API"],
  },
];

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

// Format large numbers into human-readable strings (e.g., 391B, 96.2M)
const fmt = (n, decimals = 1) => {
  if (n === null || n === undefined) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e12) return `$${(n / 1e12).toFixed(decimals)}T`;
  if (abs >= 1e9) return `$${(n / 1e9).toFixed(decimals)}B`;
  if (abs >= 1e6) return `$${(n / 1e6).toFixed(decimals)}M`;
  if (abs >= 1e3) return `$${(n / 1e3).toFixed(decimals)}K`;
  return `$${n.toFixed(decimals)}`;
};

// Format percentage values with + prefix for positives
const fmtPct = (n) => {
  if (n === null || n === undefined) return "—";
  const prefix = n > 0 ? "+" : "";
  return `${prefix}${n.toFixed(1)}%`;
};

// ============================================================
// SUB-COMPONENTS
// ============================================================

// --- Metric Card: Shows a single KPI with label, value, and optional change ---
const MetricCard = ({ label, value, change, icon: Icon, color = "indigo" }) => {
  const isPositive = change > 0; // Determine arrow direction and color
  const colorMap = {
    indigo: "bg-indigo-50 text-indigo-600 border-indigo-100",
    emerald: "bg-emerald-50 text-emerald-600 border-emerald-100",
    amber: "bg-amber-50 text-amber-600 border-amber-100",
    blue: "bg-blue-50 text-blue-600 border-blue-100",
    violet: "bg-violet-50 text-violet-600 border-violet-100",
    red: "bg-red-50 text-red-600 border-red-100",
  };

  return (
    <div className="bg-white rounded-xl border border-gray-100 p-4 hover:shadow-md transition-all duration-200 cursor-pointer group">
      {/* Top row: icon badge + change indicator */}
      <div className="flex items-center justify-between mb-3">
        <div className={`p-2 rounded-lg ${colorMap[color]} border`}>
          <Icon size={16} />
        </div>
        {change !== undefined && (
          <span className={`text-xs font-medium flex items-center gap-0.5 ${isPositive ? "text-emerald-600" : "text-red-500"}`}>
            {isPositive ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
            {fmtPct(change)}
          </span>
        )}
      </div>
      {/* Metric value */}
      <div className="text-2xl font-bold text-gray-900 mb-1 group-hover:text-indigo-600 transition-colors">{value}</div>
      {/* Label */}
      <div className="text-xs text-gray-500 font-medium uppercase tracking-wider">{label}</div>
    </div>
  );
};

// --- Section Header: Title bar with optional action buttons ---
const SectionHeader = ({ title, subtitle, action }) => (
  <div className="flex items-center justify-between mb-4">
    <div>
      <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
      {subtitle && <p className="text-sm text-gray-500 mt-0.5">{subtitle}</p>}
    </div>
    {action}
  </div>
);

// --- Pill/Tab Selector: Horizontal tab row for switching views ---
const TabPills = ({ tabs, active, onChange }) => (
  <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
    {tabs.map((tab) => (
      <button
        key={tab.id}
        onClick={() => onChange(tab.id)}
        className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${
          active === tab.id
            ? "bg-white text-indigo-600 shadow-sm" // Active state: white bg + brand color
            : "text-gray-500 hover:text-gray-700"   // Inactive state: gray text
        }`}
      >
        {tab.label}
      </button>
    ))}
  </div>
);

// --- Badge: Small colored label for tags/status ---
const Badge = ({ children, color = "gray" }) => {
  const colors = {
    gray: "bg-gray-100 text-gray-600",
    green: "bg-emerald-50 text-emerald-700",
    red: "bg-red-50 text-red-700",
    blue: "bg-blue-50 text-blue-700",
    amber: "bg-amber-50 text-amber-700",
    indigo: "bg-indigo-50 text-indigo-700",
  };
  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${colors[color]}`}>
      {children}
    </span>
  );
};

// ============================================================
// MAIN DASHBOARD COMPONENT
// ============================================================
export default function FineasDashboard() {
  // --- STATE ---
  const [darkMode, setDarkMode] = useState(false);               // Theme toggle
  const [searchQuery, setSearchQuery] = useState("");             // Search bar input
  const [activeView, setActiveView] = useState("dashboard");      // Main content view
  const [activePeriod, setActivePeriod] = useState("annual");     // Annual vs quarterly toggle
  const [chatOpen, setChatOpen] = useState(true);                 // AI chat sidebar visibility
  const [chatInput, setChatInput] = useState("");                 // Current chat message draft
  const [chatMessages, setChatMessages] = useState(MOCK_CHAT);   // Chat history
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);// Left nav collapse state
  const [selectedCompany, setSelectedCompany] = useState(MOCK_FINANCIALS); // Active company data
  const chatEndRef = useRef(null);                                // Ref for auto-scroll to bottom

  // Auto-scroll chat to latest message when new messages arrive
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  const data = selectedCompany; // Alias for cleaner JSX

  // --- VIEW DEFINITIONS: Tabs in the main content area ---
  const views = [
    { id: "dashboard", label: "Dashboard", icon: BarChart3 },
    { id: "statements", label: "Statements", icon: FileText },
    { id: "comps", label: "Comparables", icon: GitCompare },
    { id: "filings", label: "Filings", icon: BookOpen },
    { id: "insights", label: "AI Insights", icon: Zap },
  ];

  // --- CHAT SUBMIT HANDLER ---
  const handleChatSubmit = (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return; // Ignore empty messages
    // Add user message to history
    setChatMessages((prev) => [...prev, { role: "user", text: chatInput }]);
    // Simulate AI response after brief delay
    setTimeout(() => {
      setChatMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Analyzing your question against the loaded financial data. Processing XBRL facts and filing sections...",
        },
      ]);
    }, 800);
    setChatInput(""); // Clear input
  };

  // ============================================================
  // RENDER
  // ============================================================
  return (
    <div className={`min-h-screen flex ${darkMode ? "bg-gray-950 text-white" : "bg-gray-50 text-gray-900"}`}>

      {/* ============================================================ */}
      {/* LEFT SIDEBAR — Navigation + Quick Actions                     */}
      {/* ============================================================ */}
      <aside
        className={`${sidebarCollapsed ? "w-16" : "w-64"} bg-white border-r border-gray-200 flex flex-col transition-all duration-300 shrink-0`}
      >
        {/* Brand Logo */}
        <div className="p-4 border-b border-gray-100">
          <div className="flex items-center gap-3">
            {/* Animated gradient logo mark */}
            <div className="w-9 h-9 bg-gradient-to-br from-indigo-500 to-violet-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-200">
              <TrendingUp size={18} className="text-white" />
            </div>
            {/* Brand name — hidden when sidebar collapsed */}
            {!sidebarCollapsed && (
              <div>
                <span className="text-lg font-bold bg-gradient-to-r from-indigo-600 to-violet-600 bg-clip-text text-transparent">
                  Fineas
                </span>
                <span className="text-xs text-gray-400 block -mt-0.5">.ai</span>
              </div>
            )}
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 p-3 space-y-1">
          {/* Each nav item: icon + label, active state highlights in indigo */}
          {[
            { icon: BarChart3, label: "Dashboard", id: "dashboard" },
            { icon: Search, label: "Screener", id: "screener" },
            { icon: GitCompare, label: "Compare", id: "comps" },
            { icon: BookOpen, label: "Filings", id: "filings" },
            { icon: Bell, label: "Alerts", id: "alerts" },
            { icon: Star, label: "Watchlist", id: "watchlist" },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveView(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                activeView === item.id
                  ? "bg-indigo-50 text-indigo-600"   // Active: indigo highlight
                  : "text-gray-600 hover:bg-gray-50"  // Inactive: subtle hover
              }`}
            >
              <item.icon size={18} />
              {!sidebarCollapsed && item.label}
            </button>
          ))}
        </nav>

        {/* Sidebar Collapse Toggle */}
        <div className="p-3 border-t border-gray-100">
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm text-gray-500 hover:bg-gray-50"
          >
            {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
            {!sidebarCollapsed && "Collapse"}
          </button>
        </div>
      </aside>

      {/* ============================================================ */}
      {/* MAIN CONTENT AREA                                            */}
      {/* ============================================================ */}
      <main className="flex-1 flex flex-col overflow-hidden">

        {/* --- TOP BAR: Search + Company Header + Actions --- */}
        <header className="bg-white border-b border-gray-200 px-6 py-3">
          <div className="flex items-center gap-4">

            {/* Search Bar — Always visible, auto-complete ready */}
            <div className="relative flex-1 max-w-md">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search companies, tickers, or ask a question..."
                className="w-full pl-9 pr-4 py-2 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
              />
              {/* Keyboard shortcut hint */}
              <kbd className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded border border-gray-200">
                /
              </kbd>
            </div>

            {/* Company Quick Info — Ticker badge + price */}
            <div className="flex items-center gap-3 px-4 py-1.5 bg-gray-50 rounded-xl border border-gray-100">
              <div className="flex items-center gap-2">
                <Badge color="indigo">{data.ticker}</Badge>
                <span className="font-semibold text-sm">{data.company}</span>
              </div>
              <div className="h-4 w-px bg-gray-200" /> {/* Vertical divider */}
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-sm">${data.price}</span>
                <span className={`text-xs font-medium flex items-center ${data.priceChangePct > 0 ? "text-emerald-600" : "text-red-500"}`}>
                  {data.priceChangePct > 0 ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                  {fmtPct(data.priceChangePct)}
                </span>
              </div>
            </div>

            {/* Action Buttons — Theme toggle, notifications, export, chat toggle */}
            <div className="flex items-center gap-2">
              <button onClick={() => setDarkMode(!darkMode)} className="p-2 rounded-lg hover:bg-gray-100 text-gray-500 transition-colors">
                {darkMode ? <Sun size={18} /> : <Moon size={18} />}
              </button>
              <button className="p-2 rounded-lg hover:bg-gray-100 text-gray-500 relative transition-colors">
                <Bell size={18} />
                {/* Notification dot */}
                <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full" />
              </button>
              <button className="p-2 rounded-lg hover:bg-gray-100 text-gray-500 transition-colors">
                <Download size={18} />
              </button>
              <button
                onClick={() => setChatOpen(!chatOpen)}
                className={`p-2 rounded-lg transition-colors ${chatOpen ? "bg-indigo-50 text-indigo-600" : "hover:bg-gray-100 text-gray-500"}`}
              >
                <MessageSquare size={18} />
              </button>
            </div>
          </div>

          {/* View Tabs — Directly below the top bar */}
          <div className="flex items-center gap-4 mt-3">
            {/* Main view selector tabs */}
            <div className="flex gap-1">
              {views.map((v) => (
                <button
                  key={v.id}
                  onClick={() => setActiveView(v.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg transition-all ${
                    activeView === v.id
                      ? "bg-indigo-600 text-white shadow-sm"
                      : "text-gray-500 hover:bg-gray-100"
                  }`}
                >
                  <v.icon size={14} />
                  {v.label}
                </button>
              ))}
            </div>

            <div className="flex-1" /> {/* Spacer */}

            {/* Period toggle: Annual vs Quarterly */}
            <TabPills
              tabs={[
                { id: "annual", label: "Annual" },
                { id: "quarterly", label: "Quarterly" },
              ]}
              active={activePeriod}
              onChange={setActivePeriod}
            />

            {/* Form type selector dropdown */}
            <select className="text-sm bg-gray-50 border border-gray-200 rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-indigo-500">
              <option>10-K (Annual)</option>
              <option>10-Q (Quarterly)</option>
              <option>20-F (FPI Annual)</option>
              <option>8-K (Current)</option>
            </select>
          </div>
        </header>

        {/* --- SCROLLABLE CONTENT AREA --- */}
        <div className="flex-1 flex overflow-hidden">

          {/* ============================================================ */}
          {/* DASHBOARD CONTENT (center area, scrollable)                  */}
          {/* ============================================================ */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">

            {/* ROW 1: Key Metric Cards — 6 KPIs at a glance */}
            <div className="grid grid-cols-6 gap-4">
              <MetricCard label="Revenue" value={fmt(data.metrics.revenue)} change={2.0} icon={DollarSign} color="indigo" />
              <MetricCard label="Net Income" value={fmt(data.metrics.net_income)} change={-3.4} icon={TrendingUp} color="emerald" />
              <MetricCard label="FCF" value={fmt(data.metrics.free_cash_flow)} change={0.8} icon={Activity} color="blue" />
              <MetricCard label="Gross Margin" value={`${data.ratios.gross_margin}%`} icon={PieChartIcon} color="violet" />
              <MetricCard label="Market Cap" value={fmt(data.marketCap)} change={1.04} icon={Globe} color="amber" />
              <MetricCard label="EPS" value={`$${data.metrics.eps_diluted}`} change={-2.1} icon={Users} color="red" />
            </div>

            {/* ROW 2: Revenue Trend Chart + Revenue Segments Pie */}
            <div className="grid grid-cols-3 gap-6">
              {/* Revenue & Profitability Trend (2/3 width) */}
              <div className="col-span-2 bg-white rounded-xl border border-gray-100 p-5">
                <SectionHeader
                  title="Revenue & Profitability Trend"
                  subtitle="5-year annual data from SEC XBRL"
                  action={
                    <div className="flex gap-2">
                      {/* Color-coded legend pills */}
                      <Badge color="indigo">Revenue</Badge>
                      <Badge color="green">Net Income</Badge>
                      <Badge color="blue">FCF</Badge>
                    </div>
                  }
                />
                {/* Recharts composed chart: bars for revenue, lines for income/FCF */}
                <ResponsiveContainer width="100%" height={280}>
                  <ComposedChart data={data.historical}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="year" tick={{ fontSize: 12, fill: "#94a3b8" }} />
                    <YAxis tick={{ fontSize: 12, fill: "#94a3b8" }} tickFormatter={(v) => `$${v / 1000}T`} />
                    <Tooltip
                      contentStyle={{
                        borderRadius: "12px",
                        border: "1px solid #e2e8f0",
                        boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                      }}
                      formatter={(v) => [`$${(v / 1000).toFixed(1)}B`, undefined]}
                    />
                    <Bar dataKey="revenue" fill="#6366f1" radius={[4, 4, 0, 0]} opacity={0.8} name="Revenue" />
                    <Line type="monotone" dataKey="netIncome" stroke="#10b981" strokeWidth={2.5} dot={{ r: 4 }} name="Net Income" />
                    <Line type="monotone" dataKey="fcf" stroke="#3b82f6" strokeWidth={2.5} dot={{ r: 4 }} strokeDasharray="5 5" name="FCF" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              {/* Revenue Segments Donut (1/3 width) */}
              <div className="bg-white rounded-xl border border-gray-100 p-5">
                <SectionHeader title="Revenue Segments" subtitle="FY2024 breakdown" />
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={data.segments}
                      cx="50%"
                      cy="50%"
                      innerRadius={55}
                      outerRadius={85}
                      paddingAngle={3}
                      dataKey="value"
                    >
                      {/* Each slice gets a unique color from the palette */}
                      {data.segments.map((_, i) => (
                        <Cell key={i} fill={COLORS.chart[i % COLORS.chart.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(v) => fmt(v)} />
                  </PieChart>
                </ResponsiveContainer>
                {/* Segment legend below the chart */}
                <div className="space-y-2 mt-2">
                  {data.segments.map((seg, i) => (
                    <div key={seg.name} className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS.chart[i] }} />
                        <span className="text-gray-600">{seg.name}</span>
                      </div>
                      <span className="font-medium text-gray-900">{seg.pct}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* ROW 3: Margin Trend + Peer Comparison Table */}
            <div className="grid grid-cols-2 gap-6">
              {/* Margin Trend Area Chart */}
              <div className="bg-white rounded-xl border border-gray-100 p-5">
                <SectionHeader title="Margin Trend" subtitle="Net margin over time" />
                <ResponsiveContainer width="100%" height={220}>
                  <AreaChart data={data.historical}>
                    <defs>
                      {/* Gradient fill for the area */}
                      <linearGradient id="marginGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="year" tick={{ fontSize: 12, fill: "#94a3b8" }} />
                    <YAxis tick={{ fontSize: 12, fill: "#94a3b8" }} tickFormatter={(v) => `${v}%`} domain={[15, 30]} />
                    <Tooltip formatter={(v) => `${v}%`} />
                    <Area type="monotone" dataKey="margin" stroke="#6366f1" strokeWidth={2.5} fill="url(#marginGrad)" dot={{ r: 4, fill: "#6366f1" }} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Peer Comparison Table */}
              <div className="bg-white rounded-xl border border-gray-100 p-5">
                <SectionHeader
                  title="Peer Comparison"
                  subtitle="Key metrics vs industry peers"
                  action={
                    <button className="text-xs text-indigo-600 font-medium hover:underline flex items-center gap-1">
                      <Plus size={12} /> Add Peer
                    </button>
                  }
                />
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-gray-500 uppercase tracking-wider border-b border-gray-100">
                      <th className="pb-3 font-medium">Company</th>
                      <th className="pb-3 font-medium text-right">Revenue</th>
                      <th className="pb-3 font-medium text-right">Net Margin</th>
                      <th className="pb-3 font-medium text-right">P/E</th>
                    </tr>
                  </thead>
                  <tbody>
                    {/* Current company row — highlighted */}
                    <tr className="border-b border-gray-50 bg-indigo-50/50">
                      <td className="py-2.5 font-semibold text-indigo-600">AAPL</td>
                      <td className="py-2.5 text-right font-medium">$391.0B</td>
                      <td className="py-2.5 text-right">24.0%</td>
                      <td className="py-2.5 text-right">36.5x</td>
                    </tr>
                    {/* Peer rows */}
                    {data.peers.map((peer) => (
                      <tr key={peer.ticker} className="border-b border-gray-50 hover:bg-gray-50 cursor-pointer transition-colors">
                        <td className="py-2.5">
                          <span className="font-medium text-gray-900">{peer.ticker}</span>
                          <span className="text-gray-400 ml-1.5 text-xs">{peer.name}</span>
                        </td>
                        <td className="py-2.5 text-right font-medium">${(peer.revenue / 1000).toFixed(1)}B</td>
                        <td className="py-2.5 text-right">{peer.margin}%</td>
                        <td className="py-2.5 text-right">{peer.pe}x</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* ROW 4: Validation Warnings + Data Provenance */}
            <div className="bg-white rounded-xl border border-gray-100 p-5">
              <SectionHeader title="Data Quality & Provenance" subtitle="XBRL extraction validation results" />
              <div className="flex items-center gap-3">
                {/* Green checkmark badge for passing validation */}
                <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-50 rounded-lg border border-emerald-100">
                  <Shield size={14} className="text-emerald-600" />
                  <span className="text-sm font-medium text-emerald-700">All validation checks passed</span>
                </div>
                {/* Filing source badge */}
                <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-50 rounded-lg border border-gray-100">
                  <Database size={14} className="text-gray-500" />
                  <span className="text-sm text-gray-600">Source: 10-K FY2024 XBRL</span>
                </div>
                {/* Extraction timestamp */}
                <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-50 rounded-lg border border-gray-100">
                  <Clock size={14} className="text-gray-500" />
                  <span className="text-sm text-gray-600">Extracted: 2 min ago</span>
                </div>
              </div>
            </div>
          </div>

          {/* ============================================================ */}
          {/* RIGHT SIDEBAR — AI Chat Panel (toggleable)                   */}
          {/* ============================================================ */}
          {chatOpen && (
            <aside className="w-96 bg-white border-l border-gray-200 flex flex-col shrink-0">
              {/* Chat Header */}
              <div className="p-4 border-b border-gray-100 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 bg-gradient-to-br from-indigo-500 to-violet-600 rounded-lg flex items-center justify-center">
                    <Zap size={14} className="text-white" />
                  </div>
                  <div>
                    <span className="font-semibold text-sm">AI Analyst</span>
                    <span className="text-xs text-gray-400 block">Powered by Claude</span>
                  </div>
                </div>
                <button onClick={() => setChatOpen(false)} className="p-1 rounded hover:bg-gray-100 text-gray-400">
                  <X size={16} />
                </button>
              </div>

              {/* Chat Messages — scrollable area */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {chatMessages.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                        msg.role === "user"
                          ? "bg-indigo-600 text-white rounded-br-md"    // User bubble: indigo
                          : "bg-gray-100 text-gray-800 rounded-bl-md"   // Assistant bubble: gray
                      }`}
                    >
                      {/* Render text with basic markdown bold support */}
                      {msg.text.split("\n").map((line, j) => (
                        <p key={j} className={j > 0 ? "mt-2" : ""}>
                          {line.split(/\*\*(.*?)\*\*/).map((part, k) =>
                            k % 2 === 1 ? <strong key={k}>{part}</strong> : part
                          )}
                        </p>
                      ))}
                      {/* Source citations below the message */}
                      {msg.sources && (
                        <div className="mt-2 pt-2 border-t border-gray-200/50">
                          <p className="text-xs text-gray-500 font-medium mb-1">Sources:</p>
                          {msg.sources.map((s, j) => (
                            <p key={j} className="text-xs text-indigo-400 flex items-center gap-1">
                              <ExternalLink size={10} /> {s}
                            </p>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} /> {/* Invisible element at end for scroll-to */}
              </div>

              {/* Quick Action Chips — Contextual one-click queries */}
              <div className="px-4 py-2 flex flex-wrap gap-1.5">
                {["Executive Summary", "Key Risks", "Revenue Trend", "Compare Peers", "Filing Diff"].map(
                  (chip) => (
                    <button
                      key={chip}
                      className="px-2.5 py-1 text-xs bg-gray-50 border border-gray-200 rounded-full text-gray-600 hover:bg-indigo-50 hover:text-indigo-600 hover:border-indigo-200 transition-colors"
                    >
                      {chip}
                    </button>
                  )
                )}
              </div>

              {/* Chat Input Form */}
              <form onSubmit={handleChatSubmit} className="p-3 border-t border-gray-100">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Ask about financials..."
                    className="flex-1 px-3 py-2 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                  <button
                    type="submit"
                    className="px-4 py-2 bg-indigo-600 text-white rounded-xl text-sm font-medium hover:bg-indigo-700 transition-colors shadow-sm"
                  >
                    Send
                  </button>
                </div>
              </form>
            </aside>
          )}
        </div>
      </main>
    </div>
  );
}