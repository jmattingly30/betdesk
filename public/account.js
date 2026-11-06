// J's desk on the shared site: keeps bets in J's own account on the server (synced across devices) by
// giving the page the same small db interface it used inside Claude, and adds the "People" card
// (invite links for guests) to Settings. If the server has no sign-in (setup mode) this does nothing
// and the page keeps saving on the device as before.
(() => {
  const H = { 'content-type': 'application/json', 'x-desk': '1' };
  const call = async (method, path, body) => {
    const r = await fetch('/api/' + path, { method, headers: H, credentials: 'same-origin', body: body === undefined ? undefined : JSON.stringify(body) });
    if (!r.ok) throw Object.assign(new Error('save failed'), { code: r.status === 401 ? 'signed out' : 'server ' + r.status });
    return r.json();
  };
  const LS = 'betdesk.v1', MOVED = 'betdesk.moved_to_account';

  // ---- db shim: collection(name).onSnapshot / doc(id).set|delete / add, doc('settings/main') ----
  function makeDb() {
    const cache = {}, subs = {}, docSubs = {};
    const snap = c => { const docs = [...(cache[c] || new Map())].map(([id, data]) => ({ id, data: () => data })); return { docs, size: docs.length }; };
    const emit = c => {
      (subs[c] || []).forEach(f => f(snap(c)));
      (docSubs[c] || []).forEach(([id, f]) => { const v = (cache[c] || new Map()).get(id); f({ exists: !!v, data: () => v }); });
    };
    const load = async c => {
      if (c === 'lines') { // only the count is shown; counts what the phone logged before the move too
        const [{ count }, base] = await Promise.all([call('GET', 'db/lines?count=1'), call('GET', 'db/meta').then(r => r.docs.find(d => d.id === 'lines_base'))]);
        cache.lines = new Map(Array.from({ length: count + ((base && base.data.n) || 0) }, (_, i) => [String(i), {}]));
      } else cache[c] = new Map((await call('GET', 'db/' + c)).docs.map(d => [d.id, d.data]));
      emit(c);
    };
    const doc = (c, id) => ({
      async set(data) { await call('PUT', `db/${c}/${encodeURIComponent(id)}`, data); (cache[c] = cache[c] || new Map()).set(id, data); emit(c); },
      async delete() { await call('DELETE', `db/${c}/${encodeURIComponent(id)}`); cache[c] && cache[c].delete(id); emit(c); },
      onSnapshot(f, e) { (docSubs[c] = docSubs[c] || []).push([id, f]); load(c).catch(err => e && e(err)); return () => {}; },
    });
    const db = {
      collection: c => ({
        doc: id => doc(c, id),
        async add(data) { const { id } = await call('POST', 'db/' + c, data); (cache[c] = cache[c] || new Map()).set(id, data); emit(c); return { id }; },
        onSnapshot(f, e) { (subs[c] = subs[c] || []).push(f); load(c).catch(err => e && e(err)); return () => {}; },
      }),
      doc: path => { const [c, id] = path.split('/'); return doc(c, id); },
    };
    // Coming back to the app (other device may have saved): reload bets.
    document.addEventListener('visibilitychange', () => { if (!document.hidden && subs.bets) load('bets').catch(() => {}); });
    return db;
  }

  // One time per device: copy the bets, settings and line count saved on this phone into the account.
  async function moveLocal() {
    let l = null; try { if (localStorage.getItem(MOVED)) return; l = JSON.parse(localStorage.getItem(LS) || 'null'); } catch (e) { return; }
    if (!l) { try { localStorage.setItem(MOVED, new Date().toISOString()); } catch (e) {} return; }
    const bets = (l.bets || []).filter(b => b && b.id).map(b => ({ id: String(b.id), data: b }));
    // Ask first: this device may hold someone else's old bets.
    if ((bets.length || l.lines) && !confirm(`This device has ${bets.length} bet${bets.length === 1 ? '' : 's'} saved on it from before you signed in. Copy them into your account?`)) {
      try { localStorage.setItem(MOVED, 'declined ' + new Date().toISOString()); } catch (e) {}
      return;
    }
    if (bets.length) await call('POST', 'db/bets/_bulk', { docs: bets }); // same id = same bet, so a second device cannot double them
    const srv = await call('GET', 'db/settings');
    if (l.settings && !srv.docs.length) await call('PUT', 'db/settings/main', l.settings);
    if (l.lines) {
      const meta = await call('GET', 'db/meta'), base = meta.docs.find(d => d.id === 'lines_base');
      await call('PUT', 'db/meta/lines_base', { n: ((base && base.data.n) || 0) + Number(l.lines || 0) });
    }
    try { localStorage.setItem(MOVED, new Date().toISOString()); } catch (e) {}
  }

  let me = null;
  const ready = fetch('/api/me', { credentials: 'same-origin' }).then(r => (r.ok ? r.json() : null)).catch(() => null);
  window.claude = {
    async use(name) {
      if (name !== 'db') return null;
      me = await ready;
      if (!me || me.role !== 'owner') return null;
      try { await moveLocal(); } catch (e) { console.warn('moving phone bets failed, will retry next open', e); }
      return makeDb();
    },
  };

  // ---- People card (Settings): invite links for guests ----
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const ago = t => { if (!t) return 'never'; const h = (Date.now() - Date.parse(t)) / 36e5; return h < 1 ? 'just now' : h < 48 ? Math.round(h) + ' h ago' : Math.round(h / 24) + ' days ago'; };
  async function people() {
    const anchor = document.getElementById('backupCard');
    if (!anchor || document.getElementById('peopleCard')) return;
    const card = document.createElement('div');
    card.className = 'card'; card.id = 'peopleCard';
    anchor.parentNode.insertBefore(card, anchor);
    const draw = async (msg = '') => {
      let g = [];
      try { g = (await call('GET', 'admin/guests')).guests; } catch (e) { msg = 'Could not load the guest list'; }
      card.innerHTML = `<h3>People</h3>
        <p class="small muted">Each person gets their own desk with their own bets and a simpler pick sheet that grows with them. Only people with a link can get in. Your desk and bets stay yours.</p>
        ${g.map(u => `<div class="row" style="justify-content:space-between;align-items:center;gap:8px;margin:6px 0">
          <span><b>${esc(u.name)}</b> <span class="small muted">${u.bets} bets · seen ${ago(u.last_seen)}${u.active ? '' : ' · <b>off</b>'}</span></span>
          <span class="row" style="gap:6px"><button class="btn ghost" type="button" data-link="${esc(u.id)}">New link</button>
          <button class="btn ghost" type="button" data-signout="${esc(u.id)}">Sign out everywhere</button>
          <button class="btn ghost" type="button" data-active="${esc(u.id)}" data-on="${u.active ? 0 : 1}">${u.active ? 'Turn off' : 'Turn on'}</button></span></div>`).join('') || '<p class="small muted">No one yet.</p>'}
        <div class="row" style="gap:6px;margin-top:8px"><input id="newGuest" placeholder="Name" maxlength="40" style="flex:1"><button class="btn" type="button" id="addGuest">Make invite link</button></div>
        <div id="guestMsg" class="small" style="margin-top:8px;word-break:break-all">${msg}</div>
        <p class="small muted" style="margin-top:8px"><a href="/guest/" target="_blank" rel="noopener">See what guests see</a> · Each link works once. Guests' bets and notes go to Honcho to tailor their desk; yours never do.</p>`;
    };
    const showLink = (name, link) => draw(`Send this to ${esc(name)}. It works once, on one device (make another for a second device):<br><b>${esc(link)}</b> <button class="btn ghost" type="button" data-copy="${esc(link)}">Copy</button>`);
    card.addEventListener('click', async e => {
      const t = e.target.closest('button'); if (!t) return;
      try {
        if (t.id === 'addGuest') {
          const name = document.getElementById('newGuest').value.trim(); if (!name) return;
          const r = await call('POST', 'admin/guests', { name }); showLink(name, r.link);
        } else if (t.dataset.link) {
          const r = await call('POST', `admin/guests/${t.dataset.link}/link`, {}); showLink(t.closest('.row').querySelector('b').textContent, r.link);
        } else if (t.dataset.signout) {
          if (!confirm('Sign this person out on every device? They will need a new link to get back in.')) return;
          await call('POST', `admin/guests/${t.dataset.signout}/signout`, {}); draw('Signed out everywhere. Make a new link when they should get back in.');
        } else if (t.dataset.active) {
          await call('POST', `admin/guests/${t.dataset.active}/active`, { active: t.dataset.on === '1' }); draw();
        } else if (t.dataset.copy) {
          await navigator.clipboard.writeText(t.dataset.copy); t.textContent = 'Copied';
        }
      } catch (err) { draw('That did not work, try again'); }
    });
    draw();
  }
  ready.then(m => { if (m && m.role === 'owner') (document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', people) : people()); });
})();
