param([datetime]$Since = (Get-Date).AddDays(-1))
# Kill java/javaw processes started at or after $Since, and report what was left.
$left = Get-Process java, javaw -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -ge $Since }
if (-not $left) { "no leftover JVMs since $Since"; exit 0 }
$left | ForEach-Object {
    "killing pid {0} started {1} ws={2}MB" -f $_.Id, $_.StartTime, [int]($_.WS / 1MB)
    try { Stop-Process -Id $_.Id -Force -ErrorAction Stop } catch { "  failed: $_" }
}
Start-Sleep -Milliseconds 500
$still = Get-Process java, javaw -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -ge $Since }
if ($still) { "STILL ALIVE: " + (($still | ForEach-Object Id) -join ',') } else { "all reaped" }
exit 0
