[CmdletBinding()]
param(
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $projectRoot "backend"
$frontendDir = Join-Path $projectRoot "frontend"
$pythonPath = Join-Path $backendDir ".venv\Scripts\python.exe"
$runtimeDir = Join-Path $projectRoot ".runtime"
$processFile = Join-Path $runtimeDir "processes.json"
$backendUrl = "http://127.0.0.1:8000/api/health"
$frontendUrl = "http://127.0.0.1:5500/"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "The local environment is missing. Run .\setup.ps1 first."
}

New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null

function Test-ApplicationUrl {
    param([string]$Url)

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Wait-ForApplicationUrl {
    param(
        [string]$Url,
        [string]$ServiceName
    )

    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        if (Test-ApplicationUrl -Url $Url) {
            return
        }
        Start-Sleep -Milliseconds 500
    }

    throw "$ServiceName did not become ready. Check the logs in $runtimeDir."
}

$startedProcesses = @()

if (-not (Test-ApplicationUrl -Url $backendUrl)) {
    $backendProcess = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $backendDir `
        -RedirectStandardOutput (Join-Path $runtimeDir "backend.out.log") `
        -RedirectStandardError (Join-Path $runtimeDir "backend.err.log") `
        -WindowStyle Hidden `
        -PassThru

    $startedProcesses += [pscustomobject]@{
        name = "backend"
        id = $backendProcess.Id
    }
}

Wait-ForApplicationUrl -Url $backendUrl -ServiceName "Backend"

if (-not (Test-ApplicationUrl -Url $frontendUrl)) {
    $frontendProcess = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @("-m", "http.server", "5500", "--bind", "127.0.0.1") `
        -WorkingDirectory $frontendDir `
        -RedirectStandardOutput (Join-Path $runtimeDir "frontend.out.log") `
        -RedirectStandardError (Join-Path $runtimeDir "frontend.err.log") `
        -WindowStyle Hidden `
        -PassThru

    $startedProcesses += [pscustomobject]@{
        name = "frontend"
        id = $frontendProcess.Id
    }
}

Wait-ForApplicationUrl -Url $frontendUrl -ServiceName "Frontend"

$startedProcesses | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding utf8

Write-Host "Meridian is running:"
Write-Host "  Dashboard: $frontendUrl"
Write-Host "  API docs:  http://127.0.0.1:8000/docs"
Write-Host ""
Write-Host "Stop it later with .\stop-app.ps1"

if (-not $NoBrowser) {
    Start-Process $frontendUrl
}
