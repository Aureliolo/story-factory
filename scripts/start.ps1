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
    param([int]$Lines = 12)

    $logFile = Join-Path $projectRoot "logs\story_factory.log"
    $result = @()

    if (Test-Path $logFile) {
        $logs = Get-Content -Path $logFile -Tail $Lines -ErrorAction SilentlyContinue
        if ($logs) {
            foreach ($line in $logs) {
                # Truncate long lines
                if ($line.Length -gt 150) {
                    $line = $line.Substring(0, 147) + "..."
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
    $ollamaRunning = Test-OllamaRunning
    $ollamaUrl = Get-OllamaUrl
    Write-Host ""
    Write-Host "  Story Factory: " -NoNewline
    if ($running) {
        Write-Host "RUNNING" -ForegroundColor Green -NoNewline
        Write-Host "  |  URL: " -NoNewline
        Write-Host "http://localhost:7860" -ForegroundColor Cyan
    } else {
        Write-Host "STOPPED" -ForegroundColor Red
    }
    Write-Host "  Ollama:        " -NoNewline
    if ($ollamaRunning) {
        Write-Host "RUNNING" -ForegroundColor Green -NoNewline
        Write-Host "  |  API: " -NoNewline
        Write-Host $ollamaUrl -ForegroundColor Cyan
    } else {
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
    Write-Host "  [7] Ollama   [Q] Quit" -ForegroundColor White

    # Logs section - visible while waiting for input
    Write-Host ""
    Write-Host "--------------------------------------------" -ForegroundColor DarkGray
    Write-Host "  Recent Logs:" -ForegroundColor DarkCyan
    $logs = Get-RecentLogLines -Lines 12
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

function Get-OllamaUrl {
    # Read ollama_url from settings.json, or use default
    $settingsFile = Join-Path $projectRoot "settings.json"
    $defaultUrl = "http://localhost:11434"

    if (Test-Path $settingsFile) {
        try {
            $settings = Get-Content $settingsFile -Raw | ConvertFrom-Json
            if ($settings.ollama_url) {
                return $settings.ollama_url
            }
        } catch {
            # Ignore parse errors, use default
        }
    }
    return $defaultUrl
}

function Test-OllamaRunning {
    param([int]$TimeoutSec = 1)
    $ollamaUrl = Get-OllamaUrl
    try {
        $response = Invoke-WebRequest -Uri "$ollamaUrl/api/tags" -TimeoutSec $TimeoutSec -UseBasicParsing -ErrorAction Stop
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Start-Ollama {
    $ollamaUrl = Get-OllamaUrl

    # Check if Ollama is already responding via API
    if (Test-OllamaRunning) {
        Set-ActionMessage "Ollama already running at $ollamaUrl" "Green"
        return
    }

    # Check if Ollama is installed as a Windows service
    $ollamaService = Get-Service -Name "ollama" -ErrorAction SilentlyContinue
    if ($ollamaService) {
        if ($ollamaService.Status -eq "Running") {
            Set-ActionMessage "Ollama service running but not responding" "Yellow"
            return
        }

        Set-ActionMessage "Starting Ollama service..." "Yellow"
        try {
            Start-Service -Name "ollama" -ErrorAction Stop
            Start-Sleep -Seconds 1

            if (Test-OllamaRunning -TimeoutSec 3) {
                Set-ActionMessage "Ollama service started at $ollamaUrl" "Green"
            } else {
                Set-ActionMessage "Service started, API warming up..." "Yellow"
            }
        } catch {
            Set-ActionMessage "Failed to start service: $_" "Red"
        }
        return
    }

    # Not a service - try to start ollama serve
    $ollamaPath = Get-Command "ollama" -ErrorAction SilentlyContinue
    if (-not $ollamaPath) {
        Set-ActionMessage "Ollama not found. Install from ollama.ai" "Red"
        return
    }

    Set-ActionMessage "Starting Ollama..." "Yellow"
    try {
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction Stop
        Start-Sleep -Seconds 1

        if (Test-OllamaRunning -TimeoutSec 3) {
            Set-ActionMessage "Ollama started at $ollamaUrl" "Green"
        } else {
            Set-ActionMessage "Started, API warming up..." "Yellow"
        }
    } catch {
        Set-ActionMessage "Failed to start Ollama: $_" "Red"
    }
}

# Main loop with auto-refresh only when logs change
$checkInterval = 2000  # milliseconds between log checks
$lastCheck = [DateTime]::Now

Write-Screen
$script:previousLogs = Get-RecentLogLines -Lines 12
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
            "7" { Start-Ollama }
            "Q" { [Environment]::Exit(0) }
            default {
                if ($choice -and $choice -ne "`r" -and $choice -ne "`n") {
                    Set-ActionMessage "Invalid: $choice" "Red"
                }
            }
        }

        # Redraw screen after action
        Write-Screen
        $script:previousLogs = Get-RecentLogLines -Lines 12
        Write-Host "  Choice: " -NoNewline
        $lastCheck = [DateTime]::Now
    }

    # Check for log changes periodically
    $elapsed = ([DateTime]::Now - $lastCheck).TotalMilliseconds
    if ($elapsed -ge $checkInterval) {
        $currentLogs = Get-RecentLogLines -Lines 12
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
