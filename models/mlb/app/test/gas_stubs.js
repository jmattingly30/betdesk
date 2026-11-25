// In-memory stand-ins for the Google services apps_script/Server.gs uses, for testing it in a browser engine.
// The spreadsheet mimics the two Sheets behaviours that matter for the log: text without a leading apostrophe
// that looks like a date becomes a Date, and text starting with "=" becomes a formula.
var GAS_LOG = [], FETCHES = [], TRIGGERS = [], PROPS = {}, CACHE = {}, SHEETS_BY_ID = {}, ROUTES = [];

var PropertiesService = { getScriptProperties: function () { return {
  getProperty: function (k) { return Object.prototype.hasOwnProperty.call(PROPS, k) ? PROPS[k] : null; },
  setProperty: function (k, v) { if (String(v).length > 9000) throw new Error('property too large: ' + k); PROPS[k] = String(v); },
  getProperties: function () { var o = {}; Object.keys(PROPS).forEach(function (k) { o[k] = PROPS[k]; }); return o; },
  deleteProperty: function (k) { delete PROPS[k]; } }; } };
var CacheService = { getScriptCache: function () { return {
  get: function (k) { return Object.prototype.hasOwnProperty.call(CACHE, k) ? CACHE[k] : null; },
  put: function (k, v) { if (String(v).length > 100000) throw new Error('cache value too large'); CACHE[k] = String(v); } }; } };

// ROUTES: [{ match: RegExp, respond: function(url) -> { code, text, headers } }]
var UrlFetchApp = { fetch: function (url) {
  FETCHES.push(url);
  for (var i = 0; i < ROUTES.length; i++) if (ROUTES[i].match.test(url)) { var r = ROUTES[i].respond(url);
    return { getResponseCode: function () { return r.code; }, getContentText: function () { return r.text; }, getHeaders: function () { return r.headers || {}; } }; }
  return { getResponseCode: function () { return 404; }, getContentText: function () { return 'not found'; }, getHeaders: function () { return {}; } };
} };

function FakeSheet(name) { this.name = name; this.data = []; }
FakeSheet.prototype.getName = function () { return this.name; };
FakeSheet.prototype.getLastRow = function () { return this.data.length; };
FakeSheet.prototype.getLastColumn = function () { return this.data.reduce(function (m, r) { return Math.max(m, r.length); }, 0); };
FakeSheet.prototype.setFrozenRows = function () {};
FakeSheet.prototype.deleteRow = function (i) { this.data.splice(i - 1, 1); };
FakeSheet.prototype.getRange = function (r, c, nr, nc) { return new FakeRange(this, r, c, nr || 1, nc || 1); };
function sheetValue(v) {  // what Sheets stores for a value written with setValues
  if (typeof v !== 'string') return v;
  if (v.charAt(0) === "'") return v.slice(1);                           // quote prefix -> plain text
  if (v.charAt(0) === '=') return { formula: v };                        // would run as a formula
  if (/^\d{4}-\d{2}-\d{2}/.test(v)) return new Date(v);                   // Sheets auto-converts dates
  if (v !== '' && !isNaN(Number(v))) return Number(v);                    // and numbers
  return v;
}
function FakeRange(sh, r, c, nr, nc) { this.sh = sh; this.r = r; this.c = c; this.nr = nr; this.nc = nc; }
FakeRange.prototype.getValues = function () {
  var out = [];
  for (var i = 0; i < this.nr; i++) { var row = this.sh.data[this.r - 1 + i] || [], o = [];
    for (var j = 0; j < this.nc; j++) { var v = row[this.c - 1 + j]; o.push(v === undefined || v === null ? '' : (v && v.formula ? '#FORMULA ' + v.formula : v)); }
    out.push(o); }
  return out;
};
FakeRange.prototype.setValues = function (vals) {
  if (vals.length !== this.nr || vals[0].length !== this.nc) throw new Error('setValues size mismatch');
  for (var i = 0; i < this.nr; i++) { var idx = this.r - 1 + i; while (this.sh.data.length <= idx) this.sh.data.push([]);
    for (var j = 0; j < this.nc; j++) this.sh.data[idx][this.c - 1 + j] = sheetValue(vals[i][j]); }
  return this;
};
function FakeSpreadsheet(name) { this.name = name; this.id = 'ss' + Object.keys(SHEETS_BY_ID).length; this.sheets = [new FakeSheet('Sheet1')]; SHEETS_BY_ID[this.id] = this; }
FakeSpreadsheet.prototype.getId = function () { return this.id; };
FakeSpreadsheet.prototype.getUrl = function () { return 'https://docs.google.com/spreadsheets/d/' + this.id; };
FakeSpreadsheet.prototype.getSheets = function () { return this.sheets.slice(); };
FakeSpreadsheet.prototype.getSheetByName = function (n) { return this.sheets.filter(function (s) { return s.name === n; })[0] || null; };
FakeSpreadsheet.prototype.insertSheet = function (n) { var s = new FakeSheet(n); this.sheets.push(s); return s; };
FakeSpreadsheet.prototype.deleteSheet = function (s) { if (this.sheets.length < 2) throw new Error('cannot delete the only sheet'); this.sheets = this.sheets.filter(function (x) { return x !== s; }); };
var SpreadsheetApp = { create: function (n) { return new FakeSpreadsheet(n); }, openById: function (id) { if (!SHEETS_BY_ID[id]) throw new Error('no spreadsheet ' + id); return SHEETS_BY_ID[id]; } };

var LockService = { getScriptLock: function () { return { waitLock: function () {}, releaseLock: function () {} }; } };
var FAKE_TODAY = '2026-09-18';
var Utilities = { formatDate: function () { return FAKE_TODAY; }, sleep: function () {}, DigestAlgorithm: { MD5: 'md5' },
  computeDigest: function (alg, s) { var h = 0; s = String(s); for (var i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0; return [h]; },
  base64EncodeWebSafe: function (b) { return String(b[0]); } };
var DRIVE = {};   // folder name -> { file name -> text }
function FakeFolder(n) { this.name = n; DRIVE[n] = DRIVE[n] || {}; }
FakeFolder.prototype.getFilesByName = function (f) { var d = DRIVE[this.name], has = Object.prototype.hasOwnProperty.call(d, f), used = false;
  return { hasNext: function () { return has && !used; }, next: function () { used = true; return { setContent: function (t) { d[f] = t; } }; } }; };
FakeFolder.prototype.createFile = function (f, t) { DRIVE[this.name][f] = t; };
var DriveApp = { getFoldersByName: function (n) { var has = !!DRIVE[n], used = false; return { hasNext: function () { return has && !used; }, next: function () { used = true; return new FakeFolder(n); } }; },
  createFolder: function (n) { return new FakeFolder(n); } };
var Session = { getScriptTimeZone: function () { return 'America/New_York'; } };
var ContentService = { MimeType: { CSV: 'text/csv' }, createTextOutput: function (t) { return { text: t, mime: null, file: null,
  setMimeType: function (m) { this.mime = m; return this; }, downloadAsFile: function (f) { this.file = f; return this; } }; } };
var HtmlService = { createHtmlOutputFromFile: function (n) { return { file: n, title: null, meta: {},
  setTitle: function (t) { this.title = t; return this; }, addMetaTag: function (k, v) { this.meta[k] = v; return this; } }; } };
var ScriptApp = {
  getService: function () { return { getUrl: function () { return 'https://script.google.com/macros/s/TEST/exec'; } }; },
  getProjectTriggers: function () { return TRIGGERS.slice(); },
  deleteTrigger: function (t) { TRIGGERS = TRIGGERS.filter(function (x) { return x !== t; }); },
  newTrigger: function (fn) { var t = { fn: fn, minutes: null, getHandlerFunction: function () { return fn; } };
    return { timeBased: function () { return { everyMinutes: function (m) { t.minutes = m; return { create: function () { TRIGGERS.push(t); return t; } }; } }; } }; },
};
var Logger = { log: function (m) { GAS_LOG.push(String(m)); } };
