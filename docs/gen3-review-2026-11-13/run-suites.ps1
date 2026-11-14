# Runs the heavy suites one after another under the heavy-job lock, logs beside the lock.
#   -Scratch <dir>   the session scratchpad: holds HEAVY.lock and the logs (required)
#   -Forge           reinstall ../forge first (when ../forge changed)
#   -NoConnector     skip the connector rebuild and suite
#   -NoPython        skip the Python unit suite
# Start it in the background and read only the tails:
#   connector:  grep -E 'Tests run:.*Skipped: [0-9]+$|BUILD' <Scratch>/connector-test.log | tail -2
#   python:     tail -1 <Scratch>/pytest.log
param(
    [Parameter(Mandatory = $true)][string]$Scratch,
    [switch]$Forge,
    [switch]$NoConnector,
    [switch]$NoPython
)
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$lock = Join-Path $Scratch "HEAVY.lock"
if (Test-Path $lock) { Write-Output "LOCK HELD: $lock"; exit 9 }
"COORD" | Out-File -Encoding ascii $lock
$start = Get-Date
try {
    if ($Forge) {
        Set-Location (Join-Path $root "..\forge")
        cmd /c "mvn -o -q -pl forge-core,forge-game,forge-ai -am install -DskipTests > `"$Scratch\forge-install.log`" 2>&1"
        Write-Output "forge install exit $LASTEXITCODE"
        if ($LASTEXITCODE -ne 0) { exit 1 }
    }
    if (-not $NoConnector) {
        Set-Location (Join-Path $root "forge-connector")
        cmd /c "mvn -o -q install -DskipTests > `"$Scratch\connector-install.log`" 2>&1"
        Write-Output "connector install exit $LASTEXITCODE"
        if ($LASTEXITCODE -ne 0) { exit 2 }
        cmd /c "mvn -o test > `"$Scratch\connector-test.log`" 2>&1"
        Write-Output "connector test exit $LASTEXITCODE"
    }
    if (-not $NoPython) {
        Set-Location $root
        $env:PYTHONPATH = "src;."
        cmd /c ".venv\Scripts\python.exe -u -m pytest tests/unit -q -p no:cacheprovider --maxfail=30 > `"$Scratch\pytest.log`" 2>&1"
        Write-Output "pytest exit $LASTEXITCODE"
    }
}
finally {
    Set-Location $root
    powershell -File (Join-Path $PSScriptRoot "reap-jvms.ps1") -Since $start.ToString("o") | Select-Object -Last 3
    Remove-Item $lock -Force
}
