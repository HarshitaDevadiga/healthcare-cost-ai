[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$processFile = Join-Path $projectRoot ".runtime\processes.json"
$expectedPython = Join-Path $projectRoot "backend\.venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $processFile)) {
    Write-Host "No application process record was found."
    exit 0
}

$records = @(Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json)

foreach ($record in $records) {
    $process = Get-Process -Id $record.id -ErrorAction SilentlyContinue

    if (-not $process) {
        continue
    }

    if ($process.Path -ne $expectedPython) {
        Write-Warning "Skipped process $($record.id) because it no longer belongs to this project."
        continue
    }

    $children = Get-Process | Where-Object {
        $_.Parent -and $_.Parent.Id -eq $record.id
    }

    foreach ($child in $children) {
        Stop-Process -Id $child.Id -ErrorAction SilentlyContinue
    }

    Stop-Process -Id $record.id -ErrorAction SilentlyContinue
    Write-Host "Stopped $($record.name)."
}

Remove-Item -LiteralPath $processFile
