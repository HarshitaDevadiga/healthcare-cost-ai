[CmdletBinding()]
param(
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $projectRoot "backend"
$venvDir = Join-Path $backendDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$requirements = Join-Path $backendDir "requirements.txt"

if (-not (Test-Path -LiteralPath $venvPython)) {
    $launcher = $null
    $launcherArgs = @()

    if ($PythonPath) {
        if (-not (Test-Path -LiteralPath $PythonPath)) {
            throw "Python was not found at: $PythonPath"
        }
        $launcher = $PythonPath
    }
    elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $launcher = "py"
        $launcherArgs = @("-3")
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $launcher = "python"
    }
    else {
        throw "Python 3.10 or newer is required. Install Python, or run .\setup.ps1 -PythonPath 'C:\path\to\python.exe'."
    }

    Write-Host "Creating the backend virtual environment..."
    & $launcher @launcherArgs -m venv $venvDir
}

Write-Host "Installing application dependencies..."
& $venvPython -m pip install -r $requirements

Write-Host ""
Write-Host "Setup complete. Start the application with:"
Write-Host "  .\start-app.ps1"
