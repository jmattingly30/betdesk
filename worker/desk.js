// One Durable Object instance per person ("u:<id>") holds that person's bets and settings, so two
// people's data never share storage. One more instance ("directory") holds the guest list and
// the hashes of their invite tokens (never the tokens themselves).
import { DurableObject } from 'cloudflare:workers';

const MAX_DOC = 64 * 1024;

export class Desk extends DurableObject {
  // ----- per-person documents (collections: bets, settings, lines, meta) -----
  async list(coll) {
    const m = await this.ctx.storage.list({ prefix: `d:${coll}:` });
    return [...m].map(([k, data]) => ({ id: k.slice(coll.length + 3), data }));
  }
  async count(coll) {
    // Counting via keys only would still load values; lines are small, so list is fine.
    return (await this.ctx.storage.list({ prefix: `d:${coll}:` })).size;
  }
  async put(coll, id, data) {
    if (JSON.stringify(data).length > MAX_DOC) throw new Error('document too large');
    await this.ctx.storage.put(`d:${coll}:${id}`, data);
  }
  async putMany(coll, docs) {
    for (let i = 0; i < docs.length; i += 100) {
      const chunk = {};
      for (const d of docs.slice(i, i + 100)) {
        if (JSON.stringify(d.data).length > MAX_DOC) throw new Error('document too large');
        chunk[`d:${coll}:${d.id}`] = d.data;
      }
      await this.ctx.storage.put(chunk);
    }
  }
  async del(coll, id) { await this.ctx.storage.delete(`d:${coll}:${id}`); }
  async getKey(k) { return (await this.ctx.storage.get(k)) ?? null; }
  async setKey(k, v) { await this.ctx.storage.put(k, v); }

  // ----- directory instance only -----
  async users() { return [...(await this.ctx.storage.list({ prefix: 'user:' })).values()]; }
  async user(id) { return (await this.ctx.storage.get(`user:${id}`)) ?? null; }
  async addUser(u, hash) {
    await this.ctx.storage.put({ [`user:${u.id}`]: u, [`tok:${hash}`]: u.id });
    return u;
  }
  async updateUser(id, patch) {
    const u = await this.user(id);
    if (!u) return null;
    const next = { ...u, ...patch, id: u.id };
    await this.ctx.storage.put(`user:${id}`, next);
    return next;
  }
  // Invite links work once: signing in deletes the token hash, so a forwarded or reused link does nothing.
  async consumeToken(hash) {
    const id = await this.ctx.storage.get(`tok:${hash}`);
    if (!id) return null;
    await this.ctx.storage.delete(`tok:${hash}`);
    return this.user(id);
  }
  // New link (another device, or a lost link): replaces any unused link; devices already signed in stay in.
  async newToken(id, hash) {
    const u = await this.user(id);
    if (!u) return null;
    await this.dropTokens(id);
    await this.ctx.storage.put(`tok:${hash}`, id);
    return u;
  }
  // Sign out everywhere: bump ver so every cookie for this person stops working, and drop unused links.
  async signOutAll(id) {
    const u = await this.user(id);
    if (!u) return null;
    await this.dropTokens(id);
    const next = { ...u, ver: (u.ver || 1) + 1 };
    await this.ctx.storage.put(`user:${id}`, next);
    return next;
  }
  async dropTokens(id) {
    for (const [k, v] of await this.ctx.storage.list({ prefix: 'tok:' })) if (v === id) await this.ctx.storage.delete(k);
  }
  // Shared invite link (one link J can send to several people; each opener gets a new desk).
  async invite() { return (await this.ctx.storage.get('invite')) ?? null; }
  async setInvite(inv) { await this.ctx.storage.put('invite', inv); return inv; }
  // Checks and counts one use in a single step, so two people tapping at once cannot pass the limit.
  async useInvite(hash, user) {
    const inv = await this.ctx.storage.get('invite');
    if (!inv || !inv.active || inv.hash !== hash || inv.used >= inv.max) return null;
    await this.ctx.storage.put({ invite: { ...inv, used: inv.used + 1 }, [`user:${user.id}`]: user });
    return user;
  }
  async touch(id, when) {
    const u = await this.user(id);
    if (u && (u.last_seen || '').slice(0, 13) !== when.slice(0, 13)) await this.ctx.storage.put(`user:${id}`, { ...u, last_seen: when });
  }
}
