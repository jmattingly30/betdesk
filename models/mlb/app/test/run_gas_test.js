// Node version of run_gas_test.ps1: rebuilds apps_script/, then runs test/gas_driver.js on the fake Google services.
const fs = require('fs'), path = require('path'), vm = require('vm');
const root = path.join(__dirname, '..');
require('child_process').execFileSync(process.execPath, [path.join(root, 'build_apps_script.js')]);
const FIXTURES = {};
for (const f of fs.readdirSync(path.join(__dirname, 'fixtures'))) FIXTURES[f] = fs.readFileSync(path.join(__dirname, 'fixtures', f), 'utf8').replace(/^﻿+/, '');
const ctx = { FIXTURES, console, TextDecoder, setTimeout, Date, JSON, Math };
ctx.window = ctx; vm.createContext(ctx);
for (const src of ['test/gas_stubs.js', 'apps_script/Bundle.gs', 'apps_script/Server.gs', 'test/gas_driver.js'])
  vm.runInContext(fs.readFileSync(path.join(root, src), 'utf8'), ctx, { filename: src });
const r = ctx.GAS_RESULTS || [];
for (const x of r) console.log((x.ok ? 'PASS ' : 'FAIL ') + x.name + (x.ok ? '' : '  -> ' + x.detail));
console.log(`\n${r.filter(x => x.ok).length}/${r.length} passed`);
process.exit(r.length && r.every(x => x.ok) ? 0 : 1);
