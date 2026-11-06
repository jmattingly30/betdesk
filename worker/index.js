// Bet Desk Worker: sign-in by invite link, per-person storage, guest pick sheet, Honcho personalization.
//
// Who sees what
//   owner (J): every file in public/ exactly as before, plus /api/db (bets sync) and /api/admin (guest list).
//   guest:     only /guest/* (the simple desk), icons, and /api/* scoped to their own id.
//   nobody:    a "this desk is private" page. Raw model files, J's page and J's bets are never served.
// The person is always taken from the signed cookie, never from anything the browser sends, so one
// person's request can only ever read or write that person's Desk.
export { Desk } from './desk.js';
import { loadPicks, denverDate } from './picks.js';
import { record, profile, honchoOn } from './honcho.js';

const COOKIE = 'bd';
const DAY = 86400e3;
const SESSION_DAYS = 400; // browsers cap cookies at 400 days
const COLLS = { owner: ['bets', 'settings', 'lines', 'meta'], guest: ['bets', 'settings'] };
const ID_RE = /^[A-Za-z0-9_.:-]{1,100}$/;
const GUEST_FILES = /^\/(guest\/.*|icon-\d+\.png|favicon\.ico)$/;
const enc = new TextEncoder();

// ---------- small helpers ----------
const b64u = buf => btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
const randomToken = () => b64u(crypto.getRandomValues(new Uint8Array(32)));
const sha256 = async s => b64u(await crypto.subtle.digest('SHA-256', enc.encode(s)));
async function hmac(secret, msg) {
  const key = await crypto.subtle.importKey('raw', enc.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  return b64u(await crypto.subtle.sign('HMAC', key, enc.encode(msg)));
}
function same(a, b) { // constant-time string compare
  if (typeof a !== 'string' || typeof b !== 'string' || a.length !== b.length) return false;
  let x = 0;
  for (let i = 0; i < a.length; i++) x |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return x === 0;
}
const SEC = {
  'x-content-type-options': 'nosniff',
  'referrer-policy': 'no-referrer',
  'x-frame-options': 'DENY',
};
function withHeaders(res, extra = {}) {
  const r = new Response(res.body, res);
  for (const [k, v] of Object.entries({ ...SEC, 'cache-control': 'private, no-cache', ...extra })) r.headers.set(k, v);
  return r;
}
const json = (data, status = 200) => withHeaders(new Response(JSON.stringify(data), { status, headers: { 'content-type': 'application/json' } }), { 'cache-control': 'no-store' });
const err = (status, message) => json({ error: message }, status);
const desk = (env, name) => env.DESK.get(env.DESK.idFromName(name));
const dir = env => desk(env, 'directory');
const userDesk = (env, who) => desk(env, `u:${who.id}`);

// ---------- sessions ----------
const ownerVer = async env => 'o' + (await sha256(env.OWNER_TOKEN)).slice(0, 10); // new OWNER_TOKEN signs J's old cookies out

async function issue(env, id, ver) {
  const exp = Date.now() + SESSION_DAYS * DAY;
  const body = `${id}.${ver}.${exp}`;
  const value = `${body}.${await hmac(env.SESSION_SECRET, body)}`;
  return `${COOKIE}=${value}; Path=/; Max-Age=${SESSION_DAYS * 86400}; HttpOnly; Secure; SameSite=Lax`;
}

async function whoIs(req, env) {
  const m = (req.headers.get('cookie') || '').match(new RegExp(`(?:^|;\\s*)${COOKIE}=([^;]+)`));
  if (!m) return null;
  const parts = m[1].split('.');
  if (parts.length !== 4) return null;
  const [id, ver, exp, sig] = parts;
  if (!same(sig, await hmac(env.SESSION_SECRET, `${id}.${ver}.${exp}`)) || Number(exp) < Date.now()) return null;
  if (id === 'owner') return ver === (await ownerVer(env)) ? { id: 'owner', role: 'owner', name: 'J' } : null;
  const u = await dir(env).user(id);
  if (!u || !u.active || String(u.ver) !== ver) return null;
  return { id: u.id, role: 'guest', name: u.name, level: u.level || null };
}

// ---------- pages ----------
const PAGE = (title, body) => withHeaders(new Response(`<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>${title}</title><meta name="robots" content="noindex">
<style>:root{color-scheme:light dark;--bg:#f6f5f2;--fg:#16181d;--mute:#5d6470}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#eceef2;--mute:#9aa1ad}}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--fg);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:360px;padding:24px 16px;text-align:center}h1{font-size:22px;margin:0 0 8px}p{color:var(--mute);margin:0}</style></head>
<body><main>${body}</main></body></html>`, { status: 401, headers: { 'content-type': 'text/html; charset=utf-8' } }), { 'cache-control': 'no-store' });
const lockedPage = () => PAGE('Bet Desk', '<h1>This Bet Desk is private</h1><p>Open the invite link you were sent to get in. If it stopped working, ask for a new one.</p>');

// ---------- guest activity -> Honcho ----------
const fmtBet = b => `${b.pick || '?'} (${b.sport || ''} ${b.market || ''}, ${b.label || 'own pick'}, ${b.units || 1} unit${(b.units || 1) == 1 ? '' : 's'} at ${b.price ?? '?'})`;
function describe(prev, next) {
  if (!next) return prev ? [`I deleted my bet on ${fmtBet(prev)}.`] : [];
  if (!prev) return [`I placed a bet: ${fmtBet(next)}.`];
  if (prev.status !== next.status && next.status && next.status !== 'open') return [`My bet on ${fmtBet(next)} ${next.status === 'push' ? 'pushed' : next.status === 'won' ? 'won' : 'lost'}.`];
  return [];
}

async function refreshProfile(env, d, who) {
  try {
    const p = await profile(env, d, who.id);
    if (p) await d.setKey('profile', p);
  } catch (e) { console.log('honcho profile', e.message); }
}
async function tell(env, ctx, d, who, lines) {
  if (who.role !== 'guest' || !honchoOn(env) || !lines.length) return;
  ctx.waitUntil((async () => {
    try {
      await record(env, d, who.id, lines);
      const n = ((await d.getKey('events')) || 0) + lines.length;
      await d.setKey('events', n);
      if (n % 6 === 0) await refreshProfile(env, d, who); // learn as they go, not only once a day
    } catch (e) { console.log('honcho record', e.message); }
  })());
}

// Level the guest sees: their own choice wins; otherwise grow with use, and Honcho may raise it.
function autoLevel(bets, prof) {
  const n = bets.length, settled = bets.filter(b => b.data.status && b.data.status !== 'open').length;
  const byUse = n >= 30 && settled >= 15 ? 3 : n >= 8 ? 2 : 1;
  return Math.max(byUse, Math.min(prof?.level || 1, byUse + 1));
}

// ---------- API ----------
async function api(req, env, ctx, who, url) {
  const path = url.pathname.slice(5).split('/').filter(Boolean).map(decodeURIComponent); // after /api/
  const method = req.method;
  if (method !== 'GET' && req.headers.get('x-desk') !== '1') return err(403, 'missing header'); // blocks cross-site form posts
  const body = async () => {
    const t = await req.text();
    if (t.length > 512 * 1024) throw new Error('too large');
    return t ? JSON.parse(t) : {};
  };
  const d = userDesk(env, who);

  if (path[0] === 'me' && method === 'GET') {
    const out = { id: who.id, role: who.role, name: who.name, honcho: honchoOn(env) && who.role === 'guest', today: denverDate() };
    if (who.role === 'guest') {
      const [bets, prof] = await Promise.all([d.list('bets'), d.getKey('profile')]);
      out.profile = prof;
      out.level = who.level || autoLevel(bets, prof);
      out.level_auto = !who.level;
      if (honchoOn(env) && (!prof || Date.now() - Date.parse(prof.at) > 20 * 3600e3)) ctx.waitUntil(refreshProfile(env, d, who));
      ctx.waitUntil(dir(env).touch(who.id, new Date().toISOString()));
    }
    return json(out);
  }

  if (path[0] === 'picks' && method === 'GET') {
    if (who.role !== 'guest' && who.role !== 'owner') return err(403, 'no');
    return json({ picks: await loadPicks(env, url.origin), today: denverDate() });
  }

  if (path[0] === 'level' && method === 'POST' && who.role === 'guest') {
    const { level } = await body();
    const lv = level === null ? null : Math.min(3, Math.max(1, Number(level) || 1));
    await dir(env).updateUser(who.id, { level: lv });
    tell(env, ctx, d, who, [lv ? `I switched my desk to detail level ${lv} of 3.` : 'I let the desk pick my detail level again.']);
    return json({ ok: true, level: lv });
  }

  if (path[0] === 'note' && method === 'POST' && who.role === 'guest') { // "tell the desk what you like" box + hide-pick taps
    const { text } = await body();
    const t = String(text || '').trim().slice(0, 600);
    if (!t) return err(400, 'empty');
    tell(env, ctx, d, who, [t]);
    return json({ ok: true });
  }

  if (path[0] === 'db') {
    const [, coll, id] = path;
    if (!COLLS[who.role].includes(coll) || (id && !ID_RE.test(id))) return err(404, 'not found');
    if (method === 'GET' && !id) {
      if (url.searchParams.get('count') === '1') return json({ count: await d.count(coll) });
      return json({ docs: await d.list(coll) });
    }
    if (method === 'POST' && !id) { // add with a server id (owner's checked-lines log)
      const nid = Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
      await d.put(coll, nid, await body());
      return json({ id: nid });
    }
    if (method === 'POST' && id === '_bulk') { // one-time copy of bets saved on a phone
      const { docs } = await body();
      if (!Array.isArray(docs) || docs.length > 5000) return err(400, 'bad bulk');
      const clean = docs.filter(x => x && ID_RE.test(String(x.id)) && x.data && typeof x.data === 'object');
      await d.putMany(coll, clean);
      return json({ ok: true, n: clean.length });
    }
    if (method === 'PUT' && id) {
      const data = await body();
      if (!data || typeof data !== 'object') return err(400, 'bad doc');
      if (coll === 'bets' && who.role === 'guest') {
        const prev = await d.getKey(`d:bets:${id}`);
        await d.put(coll, id, data);
        tell(env, ctx, d, who, describe(prev, data));
      } else await d.put(coll, id, data);
      return json({ ok: true });
    }
    if (method === 'DELETE' && id) {
      const prev = coll === 'bets' && who.role === 'guest' ? await d.getKey(`d:bets:${id}`) : null;
      await d.del(coll, id);
      tell(env, ctx, d, who, describe(prev, null));
      return json({ ok: true });
    }
    return err(405, 'method');
  }

  if (path[0] === 'admin' && who.role === 'owner') {
    const D = dir(env);
    if (path[1] === 'guests' && method === 'GET') {
      const users = await D.users();
      const counts = await Promise.all(users.map(u => desk(env, `u:${u.id}`).count('bets').catch(() => 0)));
      return json({ guests: users.map((u, i) => ({ id: u.id, name: u.name, active: u.active, created: u.created, last_seen: u.last_seen || null, level: u.level || null, bets: counts[i] })) });
    }
    if (path[1] === 'guests' && method === 'POST' && !path[2]) {
      const name = String((await body()).name || '').trim().slice(0, 40);
      if (!name) return err(400, 'name needed');
      const token = randomToken();
      const u = await D.addUser({ id: 'g' + randomToken().slice(0, 12).replace(/[-_]/g, 'x'), name, ver: 1, active: true, created: new Date().toISOString(), level: null }, await sha256(token));
      return json({ guest: u, link: `${url.origin}/join/${token}` });
    }
    const gid = path[2];
    if (path[1] === 'guests' && gid && method === 'POST' && path[3] === 'link') {
      const token = randomToken();
      const u = await D.rotateToken(gid, await sha256(token));
      return u ? json({ link: `${url.origin}/join/${token}` }) : err(404, 'no such guest');
    }
    if (path[1] === 'guests' && gid && method === 'POST' && path[3] === 'active') {
      const { active } = await body();
      const u = await D.updateUser(gid, { active: !!active });
      return u ? json({ ok: true, active: u.active }) : err(404, 'no such guest');
    }
  }
  return err(404, 'not found');
}

// ---------- entry ----------
export default {
  async fetch(req, env, ctx) {
    const url = new URL(req.url);
    // Not set up yet (no secrets): serve the site exactly as before so a merge never locks J out.
    if (!env.SESSION_SECRET || !env.OWNER_TOKEN) {
      if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/join/') || url.pathname.startsWith('/guest')) return err(404, 'sign-in is not set up yet');
      return env.ASSETS.fetch(req);
    }

    if (url.pathname.startsWith('/join/')) {
      const token = url.pathname.slice(6);
      let cookie = null;
      if (same(token, env.OWNER_TOKEN)) cookie = await issue(env, 'owner', await ownerVer(env));
      else if (/^[A-Za-z0-9_-]{20,80}$/.test(token)) {
        const u = await dir(env).userByToken(await sha256(token));
        if (u && u.active) cookie = await issue(env, u.id, String(u.ver));
      }
      if (!cookie) return PAGE('Bet Desk', '<h1>That link has expired</h1><p>Ask for a new invite link.</p>');
      return withHeaders(new Response(null, { status: 302, headers: { location: '/', 'set-cookie': cookie } }), { 'cache-control': 'no-store' });
    }
    if (url.pathname === '/logout') {
      return withHeaders(new Response(null, { status: 302, headers: { location: '/', 'set-cookie': `${COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax` } }));
    }

    const who = await whoIs(req, env);
    if (url.pathname.startsWith('/api/')) {
      if (!who) return err(401, 'signed out');
      try { return await api(req, env, ctx, who, url); }
      catch (e) { console.log('api', e.message); return err(400, 'request failed'); }
    }
    if (!who) return lockedPage();

    if (who.role === 'owner') return withHeaders(await env.ASSETS.fetch(req));
    // guest: their page and icons only
    if (url.pathname === '/' || url.pathname === '/index.html') return withHeaders(await env.ASSETS.fetch(new Request(`${url.origin}/guest/`, req)));
    if (GUEST_FILES.test(url.pathname)) return withHeaders(await env.ASSETS.fetch(req));
    return withHeaders(new Response('Not found', { status: 404 }));
  },
};
