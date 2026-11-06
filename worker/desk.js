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
  async userByToken(hash) {
    const id = await this.ctx.storage.get(`tok:${hash}`);
    return id ? this.user(id) : null;
  }
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
  // New invite link: drop the old token hash and bump ver so cookies from the old link stop working.
  async rotateToken(id, hash) {
    const u = await this.user(id);
    if (!u) return null;
    const old = await this.ctx.storage.list({ prefix: 'tok:' });
    for (const [k, v] of old) if (v === id) await this.ctx.storage.delete(k);
    const next = { ...u, ver: (u.ver || 1) + 1 };
    await this.ctx.storage.put({ [`user:${id}`]: next, [`tok:${hash}`]: id });
    return next;
  }
  async touch(id, when) {
    const u = await this.user(id);
    if (u && (u.last_seen || '').slice(0, 13) !== when.slice(0, 13)) await this.ctx.storage.put(`user:${id}`, { ...u, last_seen: when });
  }
}
