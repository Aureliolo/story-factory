# Story Factory Launcher
# A mini control panel for starting/stopping the application

$Host.UI.RawUI.WindowTitle = "Story Factory Control Panel"

# Get the project root directory (parent of scripts/)
$projectRoot = Split-Path -Parent $PSScriptRoot

# Store last action message
$script:lastAction = ""
$script:lastActionColor = "Gray"
$script:previousLogs = @()

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
    Write-Host "  [1] Start    [2] Stop      [3] Restart" -ForegroundColor White
    Write-Host "  [4] Browser  [5] Clear Logs [6] Clear & Restart" -ForegroundColor White
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

function Clear-AndRestart {
    Set-ActionMessage "Clearing logs and restarting..." "Yellow"

    # Stop if running
    $processes = Get-Process -Name python -ErrorAction SilentlyContinue
    if ($processes) {
        $processes | Stop-Process -Force
    }

    # Clear logs
    $logFile = Join-Path $projectRoot "logs\story_factory.log"
    if (Test-Path $logFile) {
        Clear-Content -Path $logFile
    }

    Start-Sleep -Seconds 1

    # Start fresh
    Start-Process -FilePath "python" -ArgumentList "main.py" -WorkingDirectory $projectRoot -WindowStyle Hidden
    Start-Sleep -Seconds 2

    Set-ActionMessage "Logs cleared & restarted!" "Green"
}

# Main loop with auto-refresh only when logs change
$checkInterval = 2000  # milliseconds between log checks
$lastCheck = [DateTime]::Now

Write-Screen
$script:previousLogs = Get-RecentLogLines -Lines 6
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
            "4" { Open-Browser }
            "5" { Clear-LogFile }
            "6" { Clear-AndRestart }
            "Q" { [Environment]::Exit(0) }
            default {
                if ($choice -and $choice -ne "`r" -and $choice -ne "`n") {
                    Set-ActionMessage "Invalid: $choice" "Red"
                }
            }
        }

        # Redraw screen after action
        Write-Screen
        $script:previousLogs = Get-RecentLogLines -Lines 6
        Write-Host "  Choice: " -NoNewline
        $lastCheck = [DateTime]::Now
    }

    # Check for log changes periodically
    $elapsed = ([DateTime]::Now - $lastCheck).TotalMilliseconds
    if ($elapsed -ge $checkInterval) {
        $currentLogs = Get-RecentLogLines -Lines 6
        $logsChanged = ($currentLogs -join "`n") -ne ($script:previousLogs -join "`n")

        if ($logsChanged) {
            Write-Screen
            $script:previousLogs = $currentLogs
            Write-Host "  Choice: " -NoNewline
        }
        $lastCheck = [DateTime]::Now
    }

    # Small sleep to avoid CPU spinning
    Start-Sleep -Milliseconds 100
}
