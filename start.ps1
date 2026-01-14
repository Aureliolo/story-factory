# Story Factory Launcher
# A mini control panel for starting/stopping the application

$Host.UI.RawUI.WindowTitle = "Story Factory Control Panel"

function Write-Header {
    Clear-Host
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "        STORY FACTORY CONTROL PANEL        " -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Menu {
    Write-Host "[1] Start Story Factory" -ForegroundColor Green
    Write-Host "[2] Stop Story Factory" -ForegroundColor Red
    Write-Host "[3] View Logs (live)" -ForegroundColor Yellow
    Write-Host "[4] Open in Browser" -ForegroundColor Magenta
    Write-Host "[Q] Quit" -ForegroundColor Gray
    Write-Host ""
}

function Get-PythonProcess {
    Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
        $_.MainWindowTitle -match "gradio" -or $_.CommandLine -match "main.py"
    }
}

function Start-StoryFactory {
    $existing = Get-Process -Name python -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Python process already running. Stop it first." -ForegroundColor Yellow
        return
    }

    Write-Host "Starting Story Factory..." -ForegroundColor Green
    $scriptDir = Split-Path -Parent $MyInvocation.ScriptName
    if (-not $scriptDir) { $scriptDir = Get-Location }

    Start-Process -FilePath "python" -ArgumentList "main.py" -WorkingDirectory $scriptDir -WindowStyle Hidden
    Start-Sleep -Seconds 2

    Write-Host "Story Factory started!" -ForegroundColor Green
    Write-Host "Web UI available at: http://localhost:7860" -ForegroundColor Cyan
}

function Stop-StoryFactory {
    $processes = Get-Process -Name python -ErrorAction SilentlyContinue
    if ($processes) {
        Write-Host "Stopping Story Factory..." -ForegroundColor Red
        $processes | Stop-Process -Force
        Write-Host "Story Factory stopped." -ForegroundColor Red
    } else {
        Write-Host "No Python process found." -ForegroundColor Yellow
    }
}

function Show-Logs {
    $scriptDir = Split-Path -Parent $MyInvocation.ScriptName
    if (-not $scriptDir) { $scriptDir = Get-Location }
    $logFile = Join-Path $scriptDir "logs\story_factory.log"

    if (Test-Path $logFile) {
        Write-Host "Streaming logs (Ctrl+C to stop)..." -ForegroundColor Yellow
        Write-Host "============================================" -ForegroundColor Gray
        Get-Content -Path $logFile -Tail 20 -Wait
    } else {
        Write-Host "Log file not found. Start the app first." -ForegroundColor Yellow
    }
}

function Open-Browser {
    Start-Process "http://localhost:7860"
    Write-Host "Opening browser..." -ForegroundColor Magenta
}

# Main loop
while ($true) {
    Write-Header

    # Check status
    $running = Get-Process -Name python -ErrorAction SilentlyContinue
    if ($running) {
        Write-Host "Status: " -NoNewline
        Write-Host "RUNNING" -ForegroundColor Green
        Write-Host "URL: http://localhost:7860" -ForegroundColor Cyan
    } else {
        Write-Host "Status: " -NoNewline
        Write-Host "STOPPED" -ForegroundColor Red
    }
    Write-Host ""

    Show-Menu

    $choice = Read-Host "Enter choice"

    switch ($choice.ToUpper()) {
        "1" { Start-StoryFactory; Start-Sleep -Seconds 2 }
        "2" { Stop-StoryFactory; Start-Sleep -Seconds 1 }
        "3" { Show-Logs }
        "4" { Open-Browser; Start-Sleep -Seconds 1 }
        "Q" {
            Write-Host "Goodbye!" -ForegroundColor Cyan
            exit
        }
        default { Write-Host "Invalid choice." -ForegroundColor Red; Start-Sleep -Seconds 1 }
    }
}
