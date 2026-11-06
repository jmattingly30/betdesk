// Honcho (plasticlabs) personalization for guest desks. Server side only: the key never reaches a browser.
// Each guest is one Honcho peer ("guest-<id>") with one session ("desk-<id>"). The Worker records what
// the guest does as that peer's messages, and once a day asks Honcho for a short profile that sets the
// guest's detail level, sport order and a personal note. J's own desk never talks to Honcho.

const API = 'https://api.honcho.dev/v3';

async function call(env, path, body) {
  const r = await fetch(`${API}/workspaces/${encodeURIComponent(env.HONCHO_WORKSPACE || 'betdesk')}${path}`, {
    method: 'POST',
    headers: { authorization: `Bearer ${env.HONCHO_API_KEY}`, 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`honcho ${path} ${r.status}`);
  return r.json();
}

const peerId = uid => `guest-${uid}`;
const sessionId = uid => `desk-${uid}`;

export const honchoOn = env => !!env.HONCHO_API_KEY;

// Workspace, peer and session are get-or-create; done once per guest and remembered in their Desk.
async function ensure(env, desk, uid) {
  if (await desk.getKey('honcho_ready')) return;
  const ws = env.HONCHO_WORKSPACE || 'betdesk';
  const r = await fetch(`${API}/workspaces`, {
    method: 'POST',
    headers: { authorization: `Bearer ${env.HONCHO_API_KEY}`, 'content-type': 'application/json' },
    body: JSON.stringify({ id: ws }),
  });
  if (!r.ok) throw new Error(`honcho workspace ${r.status}`);
  await call(env, '/peers', { id: peerId(uid) });
  await call(env, '/sessions', { id: sessionId(uid), peers: { [peerId(uid)]: {} } });
  await desk.setKey('honcho_ready', true);
}

export async function record(env, desk, uid, lines) {
  if (!honchoOn(env) || !lines.length) return;
  await ensure(env, desk, uid);
  await call(env, `/sessions/${sessionId(uid)}/messages`, {
    messages: lines.map(content => ({ peer_id: peerId(uid), content: String(content).slice(0, 2000) })),
  });
}

const PROFILE_SCHEMA = {
  type: 'object',
  properties: {
    level: { type: 'integer', description: '1 = plain picks only, 2 = picks plus reasons and less proven long shots, 3 = every number' },
    sports: { type: 'array', items: { type: 'string' }, description: 'Sports this person cares about, most first, from: NFL, NBA, CFB, CBB' },
    bet_types: { type: 'array', items: { type: 'string' }, description: 'Bet types they like, from: spread, total, moneyline' },
    note: { type: 'string', description: 'One friendly sentence (under 160 characters) to show at the top of their desk, plain English, no jargon' },
  },
  required: ['level', 'sports', 'bet_types', 'note'],
};

const QUERY = `This person uses a simple sports betting pick sheet. Based on everything you know about them:
which sports and bet types do they care about, how comfortable are they with betting math (level 1 = wants
plain picks only, 2 = likes short reasons and is fine seeing less proven long-shot picks, 3 = wants every number:
edges, win chances, model scores), and one short friendly note for the top of their desk that helps them bet
smarter (for example about sticking to strong picks, bet size, or a sport they do well in). If you know little
about them yet, answer level 1, all four sports, and a welcoming note.`;

export async function profile(env, desk, uid) {
  if (!honchoOn(env)) return null;
  await ensure(env, desk, uid);
  const r = await call(env, `/peers/${peerId(uid)}/chat`, { query: QUERY, reasoning_level: 'low', response_format: PROFILE_SCHEMA });
  let p;
  try { p = JSON.parse(r.content || '{}'); } catch (e) { return null; }
  const SPORTS = ['NFL', 'NBA', 'CFB', 'CBB'], TYPES = ['spread', 'total', 'moneyline'];
  return {
    level: Math.min(3, Math.max(1, Math.round(Number(p.level) || 1))),
    sports: (p.sports || []).map(s => String(s).toUpperCase()).filter(s => SPORTS.includes(s)),
    bet_types: (p.bet_types || []).map(s => String(s).toLowerCase()).filter(s => TYPES.includes(s)),
    note: String(p.note || '').slice(0, 220),
    at: new Date().toISOString(),
  };
}
