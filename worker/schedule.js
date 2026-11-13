// Reliable start times for the GitHub Actions data runs (2026-11-13).
//
// GitHub's own cron fired the Bet Desk runs 5 to 8 hours late on 11-12/13 and not at all 11-08 to 11-11 (GitHub says
// scheduled runs "can be delayed or dropped" under load). Cloudflare cron triggers fire on time, so the Worker starts
// each run with the same "Run workflow" call the Actions tab uses. The GitHub schedules stay as a backup; run.yml skips
// them when this already ran the workflow.
//
// Setup (J, once): a GitHub fine-grained token for jmattingly30/betdesk with "Actions: Read and write", saved as the
// Worker secret GH_DISPATCH_TOKEN (Cloudflare > Workers > betdesk > Settings > Variables and secrets). Without the
// secret this does nothing, so the GitHub schedules carry on alone.

// Denver times. dow: 0 = Sunday .. 6 = Saturday (omit = every day).
export const RUNS = [
  { wf: 'morning.yml', at: '06:30' },
  { wf: 'noon.yml', at: '12:05' },
  { wf: 'nfl-tuesday.yml', at: '15:30', dow: [2] },
  { wf: 'nfl-sunday.yml', at: '20:45', dow: [0] },
  { wf: 'night.yml', at: '22:15' },
];
const REPO = 'jmattingly30/betdesk';

function denver(ms) {
  const p = Object.fromEntries(new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/Denver', hourCycle: 'h23', hour: '2-digit', minute: '2-digit', weekday: 'short',
  }).formatToParts(new Date(ms)).map(x => [x.type, x.value]));
  return { hm: `${p.hour}:${p.minute}`, dow: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].indexOf(p.weekday) };
}

// Runs from wrangler.jsonc "triggers.crons" every 5 minutes; starts whatever is due in this 5-minute slot.
export async function scheduled(event, env) {
  const t = event.scheduledTime || Date.now();
  const { hm, dow } = denver(t - (t % 300e3));   // the slot's start, so a trigger that fires a few seconds late still matches
  const due = RUNS.filter(r => r.at === hm && (!r.dow || r.dow.includes(dow)));
  if (!due.length) return;
  if (!env.GH_DISPATCH_TOKEN) { console.log('schedule: GH_DISPATCH_TOKEN not set; leaving', due.map(r => r.wf).join(', '), 'to the GitHub schedule'); return; }
  for (const r of due) {
    const res = await fetch(`https://api.github.com/repos/${REPO}/actions/workflows/${r.wf}/dispatches`, {
      method: 'POST',
      headers: { authorization: `Bearer ${env.GH_DISPATCH_TOKEN}`, accept: 'application/vnd.github+json', 'user-agent': 'betdesk-worker', 'content-type': 'application/json' },
      body: JSON.stringify({ ref: 'main' }),
    });
    console.log('schedule:', r.wf, hm, res.status === 204 ? 'started' : `GitHub said ${res.status} ${await res.text()}`);
  }
}
