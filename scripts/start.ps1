# Story Factory Launcher
# A mini control panel for starting/stopping the application

$Host.UI.RawUI.WindowTitle = "Story Factory Control Panel"

# Get the project root directory (parent of scripts/)
$projectRoot = Split-Path -Parent $PSScriptRoot

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
    Write-Host "[3] Restart Story Factory" -ForegroundColor Yellow
    Write-Host "[4] View Logs (live)" -ForegroundColor Yellow
    Write-Host "[5] Open in Browser" -ForegroundColor Magenta
    Write-Host "[6] Clear Logs" -ForegroundColor DarkYellow
    Write-Host "[Q] Quit" -ForegroundColor Gray
    Write-Host ""
}

function Get-PythonProcess {
    Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
        $_.MainWindowTitle -match "Story Factory" -or $_.CommandLine -match "main.py"
    }
}

function Start-StoryFactory {
    $existing = Get-Process -Name python -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Python process already running. Stop it first." -ForegroundColor Yellow
        return
    }

    Write-Host "Starting Story Factory..." -ForegroundColor Green
    Start-Process -FilePath "python" -ArgumentList "main.py" -WorkingDirectory $projectRoot -WindowStyle Hidden
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
        return $true
    } else {
        Write-Host "No Python process found." -ForegroundColor Yellow
        return $false
    }
}

function Restart-StoryFactory {
    Write-Host "Restarting Story Factory..." -ForegroundColor Yellow
    $wasRunning = Stop-StoryFactory
    Start-Sleep -Seconds 1

    Start-Process -FilePath "python" -ArgumentList "main.py" -WorkingDirectory $projectRoot -WindowStyle Hidden
    Start-Sleep -Seconds 2

    Write-Host "Story Factory restarted!" -ForegroundColor Green
    Write-Host "Web UI available at: http://localhost:7860" -ForegroundColor Cyan
}

function Show-Logs {
    $logFile = Join-Path $projectRoot "logs\story_factory.log"

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

function Clear-Logs {
    $logFile = Join-Path $projectRoot "logs\story_factory.log"

    if (Test-Path $logFile) {
        $size = (Get-Item $logFile).Length
        Clear-Content -Path $logFile
        Write-Host "Log file cleared. ($([math]::Round($size/1KB, 2)) KB freed)" -ForegroundColor DarkYellow
    } else {
        Write-Host "Log file not found." -ForegroundColor Yellow
    }
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
        "3" { Restart-StoryFactory; Start-Sleep -Seconds 2 }
        "4" { Show-Logs }
        "5" { Open-Browser; Start-Sleep -Seconds 1 }
        "6" { Clear-Logs; Start-Sleep -Seconds 1 }
        "Q" {
            Write-Host "Goodbye!" -ForegroundColor Cyan
            exit
        }
        default { Write-Host "Invalid choice." -ForegroundColor Red; Start-Sleep -Seconds 1 }
    }
}
