# Runs test/gas_driver.js (the Apps Script server on fake Google services) in headless Edge.
# Rebuilds apps_script/ first so the test always runs the current code.
$root = Split-Path $PSScriptRoot -Parent
powershell -NoProfile -File (Join-Path $root 'build_apps_script.ps1') | Out-Null
$work = Join-Path $env:TEMP 'gas_test'; New-Item -ItemType Directory -Force $work | Out-Null
function B64($p) { [Convert]::ToBase64String([IO.File]::ReadAllBytes($p)) }
$parts = New-Object System.Text.StringBuilder
[void]$parts.Append("<!doctype html><html><head><meta charset='utf-8'></head><body><pre id='out'>NOT RUN</pre>")
[void]$parts.Append("<script>function dec(b){var bin=atob(b),u=new Uint8Array(bin.length);for(var i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return new TextDecoder('utf-8').decode(u);} var FIXTURES={};")
foreach ($f in Get-ChildItem (Join-Path $root 'test\fixtures')) { [void]$parts.Append("FIXTURES['$($f.Name)']=dec('$(B64 $f.FullName)');") }
[void]$parts.Append("</script>")
foreach ($src in 'test\gas_stubs.js', 'apps_script\Bundle.gs', 'apps_script\Server.gs', 'test\gas_driver.js') {
  $text = [IO.File]::ReadAllText((Join-Path $root $src))
  if ($text -match '</script') { throw "$src contains </script>" }
  [void]$parts.Append("<script>`n$text`n</script>")
}
[void]$parts.Append("<script>document.getElementById('out').textContent = JSON.stringify(window.GAS_RESULTS || 'NO RESULTS');</script></body></html>")
$page = Join-Path $work 'gas_test.html'
[IO.File]::WriteAllText($page, $parts.ToString(), [Text.UTF8Encoding]::new($false))
$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
Start-Process -FilePath $edge -ArgumentList @('--headless=new', '--disable-gpu', '--no-sandbox', '--virtual-time-budget=120000', '--dump-dom', "file:///$($page.Replace('\','/'))") -RedirectStandardOutput (Join-Path $work 'dom.txt') -RedirectStandardError (Join-Path $work 'err.txt') -Wait -NoNewWindow
$m = [regex]::Match([IO.File]::ReadAllText((Join-Path $work 'dom.txt')), '<pre id="out">(.*?)</pre>', 'Singleline')
$txt = [Net.WebUtility]::HtmlDecode($m.Groups[1].Value)
if ($txt.StartsWith('[')) {
  $r = $txt | ConvertFrom-Json
  foreach ($x in $r) { "{0} {1}{2}" -f $(if ($x.ok) { 'PASS' } else { 'FAIL' }), $x.name, $(if (-not $x.ok) { "  -> $($x.detail)" } else { '' }) }
  "`n$(@($r | Where-Object ok).Count)/$(@($r).Count) passed"
} else { "HARNESS: $txt" }
