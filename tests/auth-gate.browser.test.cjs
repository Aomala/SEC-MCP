const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
// Run: NODE_PATH=/path/to/node_modules node --test tests/auth-gate.browser.test.cjs
const { JSDOM } = require('jsdom');
const root = require('node:path').join(__dirname, '../src/sec_mcp/static/');
const code = fs.readFileSync(root + 'auth-gate.js', 'utf8');
const project = 'https://test-project.supabase.co';
const api = 'https://sec-mcp-production.up.railway.app';
const token = (claims) => ['header', Buffer.from(JSON.stringify(claims)).toString('base64url'), 'synthetic-signature'].join('.');
const anon = token({ role: 'anon' });
const access = (issuer = project) => token({ iss: issuer + '/auth/v1', role: 'authenticated', sub: 'test-user', exp: Math.floor(Date.now()/1000)+3600 });
const session = (issuer) => ({ access_token: access(issuer), user: { id:'test-user', email:'person@example.test' } });
const tick = () => new Promise(resolve => setTimeout(resolve, 5));
async function setup(options = {}) {
  const dom = new JSDOM('<!DOCTYPE html><html><head></head><body></body></html>', { url: options.url || 'https://fineasmcp.vercel.app/explorer?api=https://attacker.example', runScripts:'outside-only' });
  const w = dom.window;
  w.Request=Request; w.Response=Response; w.Headers=Headers; w.AbortSignal=AbortSignal;
  w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};
  w.HTMLDialogElement.prototype.close=function(){this.open=false;this.dispatchEvent(new w.Event('close'));};
  const calls=[];
  let current=options.session || null, change, logins=0, otpCalls=[];
  w.supabase={createClient(url,key,opts){
    assert.equal(url,project); assert.equal(key,anon); assert.equal(opts.auth.detectSessionInUrl,true); assert.equal(opts.auth.flowType,'pkce');
    return { auth: {
      onAuthStateChange(cb){change=cb;},
      async getSession(){return {data:{session:current}};},
      async signInWithPassword(input){logins++; assert.equal(input.email,'person@example.test'); current=session();change('SIGNED_IN',current);return {data:{session:current}};},
      async signInWithOtp(input){otpCalls.push(input);return options.otpError ? {error:{message:'user not found'}} : {data:{}};},
      async signOut(){current=null;change('SIGNED_OUT',null);return {error:null};}
    }};
  }};
  w.fetch = async (input,init) => {
    const r = input instanceof Request ? input : new Request(input,init);
    calls.push(r);
    if(new URL(r.url).pathname==='/auth/config') return new Response(JSON.stringify(options.config || {enabled:true,supabaseUrl:project,supabaseAnonKey:anon,dailyLimit:250,minuteLimit:30}),{status:options.configStatus || 200,headers:{'Content-Type':'application/json'}});
    return options.respond ? options.respond(r) : new Response('{}',{headers:{'Content-Type':'application/json','X-RateLimit-Remaining':'249','X-RateLimit-Limit':'250'}});
  };
  w.eval(code);
  await w.FineasAccount.ready;
  await tick();
  return {w,dom,calls,otpCalls,get logins(){return logins;}};
}

test('JWT goes only to the trusted exact API origin; external fetches remain untouched',async()=>{
  const ctx=await setup({session:session()}); const {w,calls,dom}=ctx;
  await w.fetch('/api/entity/AAPL');
  assert.equal(calls.at(-1).url,api+'/api/entity/AAPL');
  assert.equal(calls.at(-1).headers.get('Authorization'),'Bearer '+access());
  assert.equal(calls.at(-1).redirect,'error');
  for(const url of ['https://attacker.example/api/entity/AAPL',api+'.attacker.example/api/entity/AAPL',api+'/public/logo.png','https://world.example/maps.json']){
    await w.fetch(url);assert.equal(calls.at(-1).url,url);assert.equal(calls.at(-1).headers.get('Authorization'),null);
  }
  assert.equal(w.FineasAccount.apiBase,api); // malicious ?api= ignored
  assert.equal(calls.filter(r=>r.url.endsWith('/auth/config')).length,1);
  assert.match(w.document.querySelector('[data-account-allowance]').textContent,/249 of 250/);
  dom.window.close();
});

test('preserves Request body, Headers, init override, method and cancellation',async()=>{
  const {w,calls,dom}=await setup({session:session()});
  const controller=new AbortController();
  const request=new Request(api+'/api/comps',{method:'POST',body:'{"tickers":["AAPL"]}',headers:new Headers({'Content-Type':'application/json','X-Original':'kept'}),signal:controller.signal});
  await w.fetch(request);
  const sent=calls.at(-1);assert.equal(sent.method,'POST');assert.equal(sent.headers.get('X-Original'),'kept');assert.equal(await sent.text(),'{"tickers":["AAPL"]}');
  controller.abort();assert.equal(sent.signal.aborted,true);
  await w.fetch(new Request(api+'/api/search',{headers:{'X-Old':'ignored'}}),{headers:new Headers({'X-Init':'kept'})});
  assert.equal(calls.at(-1).headers.get('X-Init'),'kept');assert.equal(calls.at(-1).headers.get('X-Old'),null);
  const aborted=new AbortController();aborted.abort();const before=calls.length;
  await assert.rejects(w.fetch('/api/search',{signal:aborted.signal}));assert.equal(calls.length,before);
  dom.window.close();
});

test('concurrent unauthenticated requests share one dialog and resume once after login',async()=>{
  const ctx=await setup();const {w,calls,dom}=ctx;
  const first=w.fetch('/api/entity/AAPL'),second=w.fetch('/api/entity/MSFT');
  await tick();assert.equal(calls.length,1);assert.equal(w.document.querySelectorAll('dialog').length,1);assert.equal(w.document.querySelector('dialog').open,true);
  const form=w.document.querySelector('dialog form');form.elements.email.value='person@example.test';form.elements.password.value='test-only-password';
  form.dispatchEvent(new w.Event('submit',{bubbles:true,cancelable:true}));
  assert.equal((await first).status,200);assert.equal((await second).status,200);assert.equal(ctx.logins,1);assert.equal(calls.length,3);
  assert.equal(form.elements.password.value,'');assert.equal(w.document.querySelector('dialog').open,false);
  assert(calls.slice(1).every(r=>r.headers.has('Authorization')));
  dom.window.close();
});

test('canceling sign-in returns a friendly 401 without a data request',async()=>{
  const {w,calls,dom}=await setup();const pending=w.fetch('/api/search');await tick();w.document.querySelector('dialog').close();
  assert.equal((await pending).status,401);assert.equal(calls.length,1);dom.window.close();
});

test('one aborted login waiter does not prevent another from signing in',async()=>{
  const ctx=await setup();const {w,calls,dom}=ctx;const controller=new AbortController();
  const canceled=w.fetch('/api/entity/AAPL',{signal:controller.signal});
  const success=w.fetch('/api/entity/MSFT');await tick();controller.abort();await assert.rejects(canceled);
  const form=w.document.querySelector('form');form.elements.email.value='person@example.test';form.elements.password.value='test';form.dispatchEvent(new w.Event('submit',{cancelable:true}));
  assert.equal((await success).status,200);assert.equal(calls.length,2);dom.window.close();
});

test('wrong-project and service-role sessions are never attached',async()=>{
  const {w,calls,dom}=await setup({session:session('https://other-project.supabase.co')});
  const pending=w.fetch('/api/entity/AAPL');await tick();assert.equal(calls.length,1);w.document.querySelector('dialog').close();assert.equal((await pending).status,401);dom.window.close();
});

test('disabled mode preserves data access and leaves no account UI',async()=>{
  const {w,calls,dom}=await setup({config:{enabled:false}});
  await w.fetch('/api/entity/AAPL');assert.equal(calls.at(-1).headers.get('Authorization'),null);assert.equal(w.document.querySelector('dialog'),null);assert.equal(await w.FineasAccount.allowExport(),true);dom.window.close();
});

test('configuration failure fails closed and never forwards data requests',async()=>{
  for(const options of [{configStatus:503},{config:{enabled:true,supabaseUrl:project,supabaseAnonKey:token({role:'service_role'})}}]){
    const {w,calls,dom}=await setup(options);assert.equal((await w.fetch('/api/entity/AAPL')).status,503);assert.equal(calls.length,1);dom.window.close();
  }
});

test('quota pauses repeated calls locally while allowing authenticated account status',async()=>{
  const {w,calls,dom}=await setup({session:session(),respond:r=>new URL(r.url).pathname==='/v1/me'?new Response('{}'):new Response('{"code":"quota_exceeded"}',{status:429,headers:{'X-RateLimit-Remaining':'0','X-RateLimit-Limit':'250','X-RateLimit-Reset':String(Math.floor(Date.now()/1000)+120),'Retry-After':'120'}})});
  assert.equal((await w.fetch('/api/entity/AAPL')).status,429);const count=calls.length;
  assert.equal((await w.fetch('/api/entity/MSFT')).status,429);assert.equal(calls.length,count);
  assert.equal((await w.fetch('/v1/me')).status,200);assert.equal(calls.length,count+1);
  assert.equal((await w.fetch('/api/entity/NVDA')).status,429);assert.equal(calls.length,count+1);
  assert.match(w.document.querySelector('[data-account-allowance]').textContent,/0 of 250/);dom.window.close();
});

test('exports never navigate or expose a JWT in a URL',async()=>{
  const {w,calls,dom}=await setup({session:session()});
  assert.equal((await w.fetch('/api/export/AAPL/csv')).status,403);assert.equal(await w.FineasAccount.allowExport(),false);
  const anchor=w.document.createElement('a');anchor.href=api+'/api/export/AAPL/csv';w.document.body.append(anchor);
  const event=new w.MouseEvent('click',{bubbles:true,cancelable:true});anchor.dispatchEvent(event);assert.equal(event.defaultPrevented,true);await tick();
  assert.equal(calls.length,1);assert(calls.every(r=>!r.url.includes('access_token')&&!r.url.includes('Bearer')));dom.window.close();
});

test('legacy demo fetch wrapper composes without losing caller headers or auth',async()=>{
  const {w,calls,dom}=await setup({session:session()});
  const legacy=fs.readFileSync(root+'app.js','utf8').split('/** Current main view')[0];
  w.eval(legacy);
  await w.fetch(new Request(api+'/api/comps',{method:'POST',body:'{}',headers:new Headers({'X-Preserved':'yes'})}));
  assert.equal(calls.at(-1).headers.get('X-Preserved'),'yes');assert(calls.at(-1).headers.has('X-Demo-Id'));assert(calls.at(-1).headers.has('Authorization'));
  await w.fetch(api+'.attacker.example/api/data');assert.equal(calls.at(-1).headers.get('X-Demo-Id'),null);assert.equal(calls.at(-1).headers.get('Authorization'),null);dom.window.close();
});

test('401 responses are not retried and require a fresh sign-in on the next request',async()=>{
  const {w,calls,dom}=await setup({session:session(),respond:()=>new Response('{"code":"invalid_session"}',{status:401})});
  assert.equal((await w.fetch('/api/entity/AAPL')).status,401);assert.equal(calls.length,2);
  const next=w.fetch('/api/entity/MSFT');await tick();assert.equal(calls.length,2);assert.equal(w.document.querySelector('dialog').open,true);
  w.document.querySelector('dialog').close();assert.equal((await next).status,401);assert.equal(calls.length,2);dom.window.close();
});

test('public plans need no login and authenticated requests explicitly avoid browser caches',async()=>{
  const guest=await setup();await guest.w.fetch('/v1/plans');assert.equal(guest.calls.at(-1).headers.get('Authorization'),null);assert.equal(guest.w.document.querySelector('dialog').open,false);guest.dom.window.close();
  const signed=await setup({session:session()});await signed.w.fetch('/api/entity/AAPL');assert.equal(signed.calls.at(-1).cache,'no-store');signed.dom.window.close();
});

test('standalone sign-in reloads after canceled initial requests',async()=>{
  const {w,calls,dom}=await setup();let reloads=0;
  const locationImpl=Object.getOwnPropertySymbols(w.location).map(key=>w.location[key]).find(value=>value&&typeof value.reload==='function');
  locationImpl.reload=()=>{reloads++;};
  const canceled=w.fetch('/api/entity/AAPL');await tick();w.document.querySelector('dialog').close();await canceled;
  w.document.querySelector('[data-account-signin]').click();await tick();
  const form=w.document.querySelector('form');form.elements.email.value='person@example.test';form.elements.password.value='test';form.dispatchEvent(new w.Event('submit',{cancelable:true}));await tick();
  assert.equal(reloads,1);assert.equal(calls.length,1);dom.window.close();
});

test('service-role and expired local sessions never reach data endpoints',async()=>{
  for(const claims of [{iss:project+'/auth/v1',role:'service_role',sub:'service',exp:Math.floor(Date.now()/1000)+3600},{iss:project+'/auth/v1',role:'authenticated',sub:'test-user',exp:1}]){
    const {w,calls,dom}=await setup({session:{access_token:token(claims),user:{id:'test-user',email:'person@example.test'}}});
    const pending=w.fetch('/api/search');await tick();assert.equal(calls.length,1);w.document.querySelector('dialog').close();await pending;dom.window.close();
  }
});

test('email sign-in uses existing accounts, PKCE, safe same-page return and generic messaging',async()=>{
  const messages=[];
  for(const otpError of [false,true]){
    const {w,dom,otpCalls}=await setup({otpError,url:'https://fineasmcp.vercel.app/explorer?api=https://attacker.example&ticker=AAPL#section'});
    w.document.querySelector('[data-account-signin]').click();await tick();
    w.document.querySelector('[name=email]').value='person@example.test';
    w.document.querySelector('.fineas-email-link').click();await tick();
    assert.equal(otpCalls.length,1);assert.equal(otpCalls[0].options.shouldCreateUser,false);
    assert.equal(otpCalls[0].options.emailRedirectTo,'https://fineasmcp.vercel.app/explorer');
    assert.equal(otpCalls[0].email,'person@example.test');
    messages.push(w.document.querySelector('.fineas-login-status').textContent);
    assert(!w.document.querySelector('dialog').textContent.includes('reset your password'));
    w.document.querySelector('dialog').close();dom.window.close();
  }
  assert.equal(messages[0],messages[1]);assert.match(messages[0],/If this email has a Fineas account/);
});

test('implicit hash tokens are discarded; callback codes are scrubbed after session initialization',async()=>{
  const hash=await setup({url:'https://fineasmcp.vercel.app/explorer#access_token=should-not-be-used&refresh_token=never-use'});
  assert.equal(hash.w.location.hash,'');assert.equal(hash.w.document.querySelector('[data-account-identity]').textContent,'Fineas account');hash.dom.window.close();
  const code=await setup({url:'https://fineasmcp.vercel.app/explorer?code=single-use-code',session:session()});
  assert.equal(code.w.location.search,'');assert.match(code.w.document.querySelector('[data-account-identity]').textContent,/person@example.test/);code.dom.window.close();
});
