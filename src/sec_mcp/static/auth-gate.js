/* Shared Fineas account access. Load synchronously before any app fetch wrapper.
 * Enforcement lives on the API; this file only supplies a session and account UI.
 * Never accept an API/auth origin or a credential from the page's query string.
 */
(function () {
  'use strict';
  const nativeFetch = window.fetch.bind(window);
  const host = window.location.hostname;
  const sameOrigin = host.endsWith('.railway.app') || host === 'localhost' ||
    host === '127.0.0.1' || host === '[::1]';
  const apiOrigin = sameOrigin ? window.location.origin : 'https://sec-mcp-production.up.railway.app';
  const apiBase = sameOrigin ? '' : apiOrigin;
  const sdkUrl = 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.89.0/dist/umd/supabase.js';
  // Email links use authorization codes + a browser-held verifier, never hash tokens.
  const callbackUrl = new URL(window.location.href);
  const hadCallbackCode = callbackUrl.searchParams.has('code');
  const hash = new URLSearchParams(callbackUrl.hash.slice(1));
  if (hash.has('access_token') || hash.has('refresh_token')) {
    callbackUrl.hash = '';
    window.history.replaceState(window.history.state, '', callbackUrl.href);
  }
  const domReady = document.readyState === 'loading'
    ? new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve, { once: true }))
    : Promise.resolve();
  let config, clientPromise, session, rejectedToken, pendingLogin, accountBar, dialog;
  let remaining = null, dailyLimit = null, cooldown = null, notice = '';
  let waitingDataRequests = 0;

  function apiTarget(input) {
    const raw = input instanceof Request ? input.url : String(input);
    let url;
    try { url = new URL(raw, window.location.href); } catch (_) { return null; }
    if (url.username || url.password || !/^\/(api|v1)(\/|$)/.test(url.pathname)) return null;
    // Relative API paths on the static deployment must reach the real backend.
    if (url.origin === window.location.origin) url = new URL(url.pathname + url.search, apiOrigin);
    return url.origin === apiOrigin ? url : null;
  }

  function decodeClaims(token) {
    try {
      const part = token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
      return JSON.parse(atob(part));
    } catch (_) { return null; }
  }

  function isProjectSession(value) {
    const claims = value && decodeClaims(value.access_token || '');
    return !!(claims && claims.iss === config.supabaseUrl + '/auth/v1' &&
      claims.role === 'authenticated' && typeof claims.sub === 'string' &&
      claims.exp > Date.now() / 1000 && value.access_token !== rejectedToken);
  }

  function failResponse(status, code, message, headers) {
    return new Response(JSON.stringify({ code, error: message, detail: message }), {
      status, headers: Object.assign({ 'Content-Type': 'application/json' }, headers || {})
    });
  }

  const ready = nativeFetch(apiOrigin + '/auth/config', {
    cache: 'no-store', credentials: 'omit', redirect: 'error', signal: AbortSignal.timeout(10000)
  }).then(async response => {
    if (!response.ok) throw new Error('Account configuration unavailable');
    const value = await response.json();
    if (value.enabled === false) { config = value; return value; }
    if (value.enabled !== true) throw new Error('Invalid account configuration');
    const project = new URL(value.supabaseUrl);
    const local = ['localhost', '127.0.0.1', '[::1]'].includes(project.hostname);
    const key = value.supabaseAnonKey;
    const publicKey = typeof key === 'string' && (key.startsWith('sb_publishable_') ||
      decodeClaims(key)?.role === 'anon');
    if ((!local && project.protocol !== 'https:') || project.username || project.password ||
      project.pathname !== '/' || project.search || project.hash || !publicKey) {
      throw new Error('Invalid public account configuration');
    }
    config = Object.assign({}, value, { supabaseUrl: project.origin });
    dailyLimit = Number(config.dailyLimit) || 250;
    await domReady;
    mount();
    // Restore the account without making a data request or spending its allowance.
    getClient().then(async client => {
      const result = await client.auth.getSession();
      if (isProjectSession(result.data.session)) session = result.data.session;
      if (hadCallbackCode) {
        const clean = new URL(window.location.href);
        clean.searchParams.delete('code');
        window.history.replaceState(window.history.state, '', clean.href);
        if (!session) notice = 'This email link could not be completed. Request a new link and open it in this browser.';
      }
      renderAccount();
    }).catch(() => { notice = 'Sign-in is unavailable. Reload to try again.'; renderAccount(); });
    return config;
  }).catch(async () => {
    config = { enabled: true, unavailable: true };
    await domReady;
    mount();
    notice = 'Account access is temporarily unavailable. Reload to try again.';
    renderAccount();
    return config;
  });

  function loadSdk() {
    if (window.supabase?.createClient) return Promise.resolve(window.supabase);
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = sdkUrl;
      script.crossOrigin = 'anonymous';
      const timer = setTimeout(() => reject(new Error('Sign-in library unavailable')), 10000);
      script.onload = () => {
        clearTimeout(timer);
        if (window.supabase?.createClient) resolve(window.supabase);
        else reject(new Error('Sign-in library unavailable'));
      };
      script.onerror = () => { clearTimeout(timer); reject(new Error('Sign-in library unavailable')); };
      document.head.appendChild(script);
    });
  }

  function getClient() {
    if (!clientPromise) clientPromise = loadSdk().then(sdk => {
      const client = sdk.createClient(config.supabaseUrl, config.supabaseAnonKey, {
        auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true, flowType: 'pkce' },
        global: { fetch: nativeFetch }
      });
      client.auth.onAuthStateChange((event, value) => {
        if (event === 'SIGNED_OUT') {
          session = null; rejectedToken = null; remaining = null; cooldown = null;
          notice = 'Signed out. Sign in to continue exploring.';
        } else if (isProjectSession(value)) {
          if (session?.user?.id !== value.user?.id) { remaining = null; cooldown = null; }
          session = value;
          const shouldReload = pendingLogin && waitingDataRequests === 0;
          if (pendingLogin) { pendingLogin.resolve(value); pendingLogin = null; }
          if (dialog?.open) dialog.close();
          if (!cooldown) notice = '';
          // A sign-in from the account button restores requests previously canceled.
          // Initial page fetches still waiting for login resume without a reload.
          if (shouldReload) window.location.reload();
        }
        renderAccount();
      });
      return client;
    });
    return clientPromise;
  }

  async function requireSession(signal, dataRequest) {
    if (signal?.aborted) throw signal.reason || new DOMException('Aborted', 'AbortError');
    const client = await getClient();
    const result = await client.auth.getSession();
    if (isProjectSession(result.data.session)) { session = result.data.session; return session; }
    if (!pendingLogin) {
      let resolve, reject;
      const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
      pendingLogin = { promise, resolve, reject };
      showLogin();
    }
    // One dialog serves concurrent requests. Cancelling one request does not cancel others.
    const login = pendingLogin.promise;
    if (dataRequest) waitingDataRequests++;
    try {
      if (!signal) return await login;
      return await new Promise((resolve, reject) => {
        const abort = () => reject(signal.reason || new DOMException('Aborted', 'AbortError'));
        signal.addEventListener('abort', abort, { once: true });
        login.then(resolve, reject).finally(() => signal.removeEventListener('abort', abort));
        if (signal.aborted) abort();
      });
    } finally { if (dataRequest) waitingDataRequests--; }
  }

  function renderAccount() {
    if (!accountBar) return;
    const identity = accountBar.querySelector('[data-account-identity]');
    identity.textContent = session?.user?.email || 'Fineas account';
    accountBar.querySelector('[data-account-allowance]').textContent = remaining === null
      ? (dailyLimit ? dailyLimit + ' free requests/day' : '')
      : remaining + ' of ' + dailyLimit + ' requests left today';
    accountBar.querySelector('[data-account-notice]').textContent = notice;
    const signIn = accountBar.querySelector('[data-account-signin]');
    signIn.hidden = !!session;
    signIn.disabled = !!config.unavailable;
    accountBar.querySelector('[data-account-signout]').hidden = !session;
  }

  function mount() {
    if (accountBar) return;
    accountBar = document.createElement('aside');
    accountBar.className = 'fineas-account-bar';
    accountBar.setAttribute('aria-label', 'Fineas account and usage');
    accountBar.innerHTML = '<div class="fineas-account-summary"><strong data-account-identity></strong>' +
      '<span data-account-allowance></span><span data-account-notice role="status" aria-live="polite"></span></div>' +
      '<button type="button" data-account-signin>Sign in</button>' +
      '<button type="button" data-account-signout hidden>Sign out</button>';
    document.body.appendChild(accountBar);
    dialog = document.createElement('dialog');
    dialog.className = 'fineas-account-dialog';
    dialog.setAttribute('aria-labelledby', 'fineas-login-title');
    dialog.setAttribute('aria-describedby', 'fineas-login-description');
    dialog.innerHTML = '<button type="button" class="fineas-account-close" aria-label="Close sign-in">×</button>' +
      '<h2 id="fineas-login-title">Sign in to Fineas</h2>' +
      '<p id="fineas-login-description">Use your Fineas account to explore financial data. Your free allowance is shared across this site.</p>' +
      '<form><label for="fineas-login-email">Email</label>' +
      '<input id="fineas-login-email" name="email" type="email" autocomplete="username" required>' +
      '<label for="fineas-login-password">Password</label>' +
      '<input id="fineas-login-password" name="password" type="password" autocomplete="current-password" required>' +
      '<p class="fineas-login-status" role="status" aria-live="polite"></p>' +
      '<button type="submit">Sign in with password</button>' +
      '<button type="button" class="fineas-email-link">Email me a sign-in link</button></form>' +
      '<p class="fineas-account-links"><a href="https://fineas.ai/sign-up" target="_blank" rel="noopener noreferrer">Create a Fineas account</a>' +
      '<a href="https://fineas.ai/sign-in" target="_blank" rel="noopener noreferrer">Open Fineas sign-in</a></p>' +
      '<p class="fineas-account-help">Usually use Google or Apple, or forgot your password? Request an email link for the email on your Fineas account. Open it in this browser. Signing in on another site does not automatically sign you in here.</p>';
    document.body.appendChild(dialog);
    accountBar.querySelector('[data-account-signin]').addEventListener('click', () => {
      requireSession().catch(() => {});
    });
    accountBar.querySelector('[data-account-signout]').addEventListener('click', async event => {
      const button = event.currentTarget;
      button.disabled = true;
      try {
        const client = await getClient();
        const result = await client.auth.signOut({ scope: 'local' });
        if (result.error) throw result.error;
        // Clear page-level financial caches and rendered results for the next account.
        window.location.reload();
      } catch (_) { notice = 'Sign-out failed. Please try again.'; renderAccount(); }
      finally { button.disabled = false; }
    });
    const cancel = () => {
      if (pendingLogin) {
        pendingLogin.reject(new Error('Sign in to access financial data.'));
        pendingLogin = null;
      }
      dialog.querySelector('[name=password]').value = '';
    };
    dialog.addEventListener('close', cancel);
    dialog.addEventListener('cancel', cancel);
    dialog.querySelector('.fineas-account-close').addEventListener('click', () => dialog.close());
    dialog.querySelector('.fineas-email-link').addEventListener('click', async event => {
      const form = dialog.querySelector('form');
      if (!form.elements.email.reportValidity()) return;
      const button = event.currentTarget;
      const submit = form.querySelector('[type=submit]');
      const status = form.querySelector('[role=status]');
      button.disabled = true; submit.disabled = true;
      status.textContent = 'Requesting an email link…';
      try {
        const client = await getClient();
        // Do not create accounts here or disclose whether an email is registered.
        await client.auth.signInWithOtp({
          email: form.elements.email.value.trim(),
          options: {
            shouldCreateUser: false,
            emailRedirectTo: window.location.origin + window.location.pathname
          }
        });
        status.textContent = 'If this email has a Fineas account, check its inbox for a sign-in link. Open it in this browser.';
      } catch (_) {
        status.textContent = 'Email sign-in is temporarily unavailable. Please try again later.';
      } finally {
        form.elements.password.value = '';
        button.disabled = false; submit.disabled = false;
      }
    });
    dialog.querySelector('form').addEventListener('submit', async event => {
      event.preventDefault();
      const form = event.currentTarget;
      const button = form.querySelector('[type=submit]');
      const status = form.querySelector('[role=status]');
      button.disabled = true;
      status.textContent = 'Signing in…';
      try {
        const client = await getClient();
        rejectedToken = null;
        const result = await client.auth.signInWithPassword({
          email: form.elements.email.value.trim(), password: form.elements.password.value
        });
        if (result.error || !isProjectSession(result.data.session)) throw new Error('Sign-in failed');
        session = result.data.session;
        const shouldReload = pendingLogin && waitingDataRequests === 0;
        if (pendingLogin) { pendingLogin.resolve(session); pendingLogin = null; }
        status.textContent = '';
        notice = '';
        renderAccount();
        dialog.close();
        if (shouldReload) window.location.reload();
      } catch (_) {
        status.textContent = 'Unable to sign in. Check your email and password, or request an email sign-in link.';
      } finally { form.elements.password.value = ''; button.disabled = false; }
    });
    renderAccount();
  }

  function showLogin() {
    if (dialog && !dialog.open) {
      dialog.querySelector('[role=status]').textContent = '';
      dialog.showModal();
      dialog.querySelector('[name=email]').focus();
    }
  }

  function showExportNotice() {
    notice = 'Bulk downloads and exports require a separate data license. Contact Fineas for access.';
    renderAccount();
  }

  function observeResponse(response, usedToken) {
    const numberHeader = name => {
      const value = response.headers.get(name);
      return value !== null && /^\d+$/.test(value) ? Number(value) : null;
    };
    const left = numberHeader('X-RateLimit-Remaining');
    if (left !== null) remaining = left;
    dailyLimit = numberHeader('X-RateLimit-Limit') || dailyLimit;
    if (response.status === 401) {
      rejectedToken = usedToken;
      session = null;
      notice = 'Your session has expired. Sign in again, then retry your request.';
    } else if (response.status === 429) {
      const retry = numberHeader('Retry-After') || 60;
      const reset = numberHeader('X-RateLimit-Reset');
      const daily = remaining === 0;
      cooldown = {
        until: daily && reset ? reset * 1000 : Date.now() + retry * 1000,
        code: daily ? 'quota_exceeded' : 'rate_limited'
      };
      notice = daily
        ? 'Your daily allowance is used. Requests resume at ' + new Date(cooldown.until).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) + '.'
        : 'You are exploring quickly. Wait a moment before your next request.';
    } else if (response.status === 403) {
      notice = 'This request is not included in your account access. Bulk exports require a separate license.';
    } else if (response.status >= 500) {
      notice = 'Data access is temporarily unavailable. Please try again shortly.';
    } else if (response.ok && !cooldown) {
      notice = '';
    }
    renderAccount();
  }

  window.fetch = async function (input, init) {
    const target = apiTarget(input);
    if (!target) return nativeFetch(input, init);
    // Construct before awaiting so the caller's headers, body, method and signal are preserved.
    const original = input instanceof Request ? input : new URL(String(input), window.location.href);
    const request = new Request(original, init);
    const cfg = await ready;
    if (request.signal.aborted) throw request.signal.reason || new DOMException('Aborted', 'AbortError');
    const routed = new Request(target.href, request);
    if (!cfg.enabled || target.pathname === '/v1/plans') return nativeFetch(routed);
    if (cfg.unavailable) return failResponse(503, 'auth_unavailable', notice);
    let current;
    try { current = await requireSession(request.signal, true); }
    catch (error) {
      if (request.signal.aborted) throw error;
      return failResponse(401, 'account_required', 'Sign in to your Fineas account to access financial data.');
    }
    if (/^\/api\/export(?:\/|$)|^\/v1\/(?:bulk|export)(?:\/|$)/.test(target.pathname)) {
      showExportNotice();
      return failResponse(403, 'bulk_restricted', notice);
    }
    if (cooldown && cooldown.until > Date.now() && !['/v1/me', '/v1/usage'].includes(target.pathname)) {
      return failResponse(429, cooldown.code, notice, {
        'Retry-After': String(Math.ceil((cooldown.until - Date.now()) / 1000))
      });
    }
    if (cooldown && cooldown.until <= Date.now()) cooldown = null;
    const headers = new Headers(routed.headers);
    headers.set('Authorization', 'Bearer ' + current.access_token);
    // Authenticated data calls must never redirect credentials to another endpoint.
    const response = await nativeFetch(new Request(routed, {
      headers, redirect: 'error', credentials: 'omit', cache: 'no-store'
    }));
    observeResponse(response, current.access_token);
    return response;
  };

  // Download anchors cannot carry a bearer token; stop navigation before any request.
  document.addEventListener('click', async event => {
    const anchor = event.target.closest?.('a[href]');
    if (!anchor) return;
    const target = apiTarget(anchor.href);
    if (target?.pathname === '/api/tickers/bulk') return;
    if (!target || !/\/(export|bulk)(\/|$)/.test(target.pathname)) return;
    event.preventDefault();
    const cfg = await ready;
    if (cfg.enabled) showExportNotice();
    else window.location.assign(target.href);
  }, true);

  window.FineasAccount = Object.freeze({
    apiBase, ready,
    signIn: async () => { await ready; if (config.enabled && !config.unavailable) return requireSession(); },
    allowExport: async () => { await ready; if (config.enabled) { showExportNotice(); return false; } return true; }
  });
})();
