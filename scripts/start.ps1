# Story Factory Launcher
# A mini control panel for starting/stopping the application

$Host.UI.RawUI.WindowTitle = "Story Factory Control Panel"

# Get the project root directory (parent of scripts/)
$projectRoot = Split-Path -Parent $PSScriptRoot

# Store last action message
$script:lastAction = ""
$script:lastActionColor = "Gray"

function Set-ActionMessage {
    param([string]$Message, [string]$Color = "Gray")
    $script:lastAction = $Message
    $script:lastActionColor = $Color
}

function Get-RecentLogLines {
    param([int]$Lines = 6)

    $logFile = Join-Path $projectRoot "logs\story_factory.log"
    $result = @()

    if (Test-Path $logFile) {
        $logs = Get-Content -Path $logFile -Tail $Lines -ErrorAction SilentlyContinue
        if ($logs) {
            foreach ($line in $logs) {
                # Truncate long lines
                if ($line.Length -gt 100) {
                    $line = $line.Substring(0, 97) + "..."
                }
                $result += $line
            }
        }
    }
    return $result
}

function Write-LogLine {
    param([string]$Line)

    # Color code by log level
    if ($Line -match "ERROR|EXCEPTION|Traceback|AttributeError") {
        Write-Host $Line -ForegroundColor Red
    } elseif ($Line -match "WARNING") {
        Write-Host $Line -ForegroundColor Yellow
    } elseif ($Line -match "INFO") {
        Write-Host $Line -ForegroundColor Gray
    } else {
        Write-Host $Line -ForegroundColor DarkGray
    }
}

function Write-Screen {
    Clear-Host

    # Header
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "        STORY FACTORY CONTROL PANEL        " -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan

    # Status
    $running = Get-Process -Name python -ErrorAction SilentlyContinue
    Write-Host ""
    if ($running) {
        Write-Host "  Status: " -NoNewline
        Write-Host "RUNNING" -ForegroundColor Green -NoNewline
        Write-Host "  |  URL: " -NoNewline
        Write-Host "http://localhost:7860" -ForegroundColor Cyan
    } else {
        Write-Host "  Status: " -NoNewline
        Write-Host "STOPPED" -ForegroundColor Red
    }

    # Last action (if any)
    if ($script:lastAction) {
        Write-Host "  > $($script:lastAction)" -ForegroundColor $script:lastActionColor
    }
    Write-Host ""

    # Menu
    Write-Host "  [1] Start    [2] Stop    [3] Restart" -ForegroundColor White
    Write-Host "  [4] Logs     [5] Browser [6] Clear Logs" -ForegroundColor White
    Write-Host "  [Q] Quit" -ForegroundColor DarkGray

    # Logs section - visible while waiting for input
    Write-Host ""
    Write-Host "--------------------------------------------" -ForegroundColor DarkGray
    Write-Host "  Recent Logs:" -ForegroundColor DarkCyan
    $logs = Get-RecentLogLines -Lines 6
    if ($logs.Count -eq 0) {
        Write-Host "  (no logs yet)" -ForegroundColor DarkGray
    } else {
        foreach ($line in $logs) {
            Write-Host "  " -NoNewline
            Write-LogLine $line
        }
    }
    Write-Host "--------------------------------------------" -ForegroundColor DarkGray
    Write-Host ""
}

function Start-StoryFactory {
    $existing = Get-Process -Name python -ErrorAction SilentlyContinue
    if ($existing) {
        Set-ActionMessage "Already running. Stop first." "Yellow"
        return
    }

    Set-ActionMessage "Starting..." "Green"

    Start-Process -FilePath "python" -ArgumentList "main.py" -WorkingDirectory $projectRoot -WindowStyle Hidden
    Start-Sleep -Seconds 2

    Set-ActionMessage "Started! Open http://localhost:7860" "Green"
}

function Stop-StoryFactory {
    $processes = Get-Process -Name python -ErrorAction SilentlyContinue
    if ($processes) {
        $processes | Stop-Process -Force
        Set-ActionMessage "Stopped." "Red"
        return $true
    } else {
        Set-ActionMessage "Not running." "Yellow"
        return $false
    }
}

function Restart-StoryFactory {
    Set-ActionMessage "Restarting..." "Yellow"

    $processes = Get-Process -Name python -ErrorAction SilentlyContinue
    if ($processes) {
        $processes | Stop-Process -Force
    }
    Start-Sleep -Seconds 1

    Start-Process -FilePath "python" -ArgumentList "main.py" -WorkingDirectory $projectRoot -WindowStyle Hidden
    Start-Sleep -Seconds 2

    Set-ActionMessage "Restarted! Open http://localhost:7860" "Green"
}

function Show-Logs {
    $logFile = Join-Path $projectRoot "logs\story_factory.log"

    if (Test-Path $logFile) {
        Clear-Host
        Write-Host "Streaming logs (Ctrl+C to stop)..." -ForegroundColor Yellow
        Write-Host "============================================" -ForegroundColor Gray
        Get-Content -Path $logFile -Tail 30 -Wait
    } else {
        Set-ActionMessage "No log file yet." "Yellow"
    }
}

function Open-Browser {
    Start-Process "http://localhost:7860"
    Set-ActionMessage "Opening browser..." "Magenta"
}

function Clear-LogFile {
    $logFile = Join-Path $projectRoot "logs\story_factory.log"

    if (Test-Path $logFile) {
        $size = (Get-Item $logFile).Length
        Clear-Content -Path $logFile
        Set-ActionMessage "Logs cleared ($([math]::Round($size/1KB, 1)) KB)" "DarkYellow"
    } else {
        Set-ActionMessage "No log file." "Yellow"
    }
}

# Main loop with auto-refresh every 3 seconds
$refreshInterval = 3000  # milliseconds
$lastRefresh = [DateTime]::Now

Write-Screen
Write-Host "  Choice: " -NoNewline

while ($true) {
    # Check if key is available (non-blocking)
    if ([Console]::KeyAvailable) {
        $key = [Console]::ReadKey($true)
        $choice = $key.KeyChar.ToString().ToUpper()

        Write-Host $choice  # Echo the key

        switch ($choice) {
            "1" { Start-StoryFactory }
            "2" { Stop-StoryFactory }
            "3" { Restart-StoryFactory }
            "4" { Show-Logs; $script:lastAction = "" }
            "5" { Open-Browser }
            "6" { Clear-LogFile }
            "Q" { exit 0 }
            default {
                if ($choice -and $choice -ne "`r" -and $choice -ne "`n") {
                    Set-ActionMessage "Invalid: $choice" "Red"
                }
            }
        }

        # Redraw screen after action
        Write-Screen
        Write-Host "  Choice: " -NoNewline
        $lastRefresh = [DateTime]::Now
    }

    # Auto-refresh logs every 3 seconds
    $elapsed = ([DateTime]::Now - $lastRefresh).TotalMilliseconds
    if ($elapsed -ge $refreshInterval) {
        Write-Screen
        Write-Host "  Choice: " -NoNewline
        $lastRefresh = [DateTime]::Now
    }

    # Small sleep to avoid CPU spinning
    Start-Sleep -Milliseconds 100
}
