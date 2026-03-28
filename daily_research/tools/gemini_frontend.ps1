param(
    [ValidateSet("open", "close", "status", "sessions")]
    [string]$Action = "status",
    [string]$Session = "latest",
    [string]$Title = "Gemini Frontend - daily_research",
    [switch]$ForceNew
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$stateDir = Join-Path $repoRoot "daily_research\cache\gemini_frontend"
$statePath = Join-Path $stateDir "state.json"
$geminiPath = "C:\Users\ASUS\AppData\Roaming\npm\gemini.cmd"

function Get-State {
    if (-not (Test-Path -LiteralPath $statePath)) {
        return $null
    }
    return Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
}

function Save-State([int]$RootPid, [string]$Workspace, [string]$SessionName, [string]$WindowTitle) {
    New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
    $state = [ordered]@{
        root_pid    = $RootPid
        workspace   = $Workspace
        session     = $SessionName
        title       = $WindowTitle
        launched_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    }
    $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
}

function Clear-State {
    if (Test-Path -LiteralPath $statePath) {
        Remove-Item -LiteralPath $statePath -Force
    }
}

function Get-ProcessSnapshot {
    return Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, Name, CommandLine
}

function Get-DescendantProcessIds([int]$RootPid) {
    $snapshot = Get-ProcessSnapshot
    $childrenByParent = @{}
    foreach ($proc in $snapshot) {
        $parentKey = [int]$proc.ParentProcessId
        if (-not $childrenByParent.ContainsKey($parentKey)) {
            $childrenByParent[$parentKey] = New-Object System.Collections.Generic.List[object]
        }
        $childrenByParent[$parentKey].Add($proc)
    }

    $queue = New-Object System.Collections.Generic.Queue[int]
    $seen = New-Object System.Collections.Generic.HashSet[int]
    $queue.Enqueue($RootPid)
    [void]$seen.Add($RootPid)

    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        if (-not $childrenByParent.ContainsKey($current)) {
            continue
        }
        foreach ($child in $childrenByParent[$current]) {
            $childPid = [int]$child.ProcessId
            if ($seen.Add($childPid)) {
                $queue.Enqueue($childPid)
            }
        }
    }

    return [int[]]$seen
}

function Get-FrontendStatus {
    $state = Get-State
    if ($null -eq $state) {
        return [pscustomobject]@{
            running    = $false
            message    = "No Gemini frontend state file."
            root_pid   = $null
            session    = $null
            workspace  = $repoRoot
            title      = $Title
            processes  = @()
        }
    }

    $rootPid = [int]$state.root_pid
    $rootProcess = Get-Process -Id $rootPid -ErrorAction SilentlyContinue
    if ($null -eq $rootProcess) {
        Clear-State
        return [pscustomobject]@{
            running    = $false
            message    = "Gemini frontend state was stale and has been cleared."
            root_pid   = $rootPid
            session    = $state.session
            workspace  = $state.workspace
            title      = $state.title
            processes  = @()
        }
    }

    $snapshot = Get-ProcessSnapshot
    $descendantIds = Get-DescendantProcessIds -RootPid $rootPid
    $processes = foreach ($proc in $snapshot) {
        if ($descendantIds -contains [int]$proc.ProcessId) {
            [pscustomobject]@{
                pid        = [int]$proc.ProcessId
                parent_pid = [int]$proc.ParentProcessId
                name       = [string]$proc.Name
                command    = [string]$proc.CommandLine
            }
        }
    }

    return [pscustomobject]@{
        running   = $true
        message   = "Gemini frontend is running."
        root_pid  = $rootPid
        session   = $state.session
        workspace = $state.workspace
        title     = $state.title
        processes = @($processes | Sort-Object pid)
    }
}

function Open-Frontend {
    if (-not (Test-Path -LiteralPath $geminiPath)) {
        throw "Gemini CLI not found at $geminiPath"
    }

    $status = Get-FrontendStatus
    if ($status.running -and -not $ForceNew) {
        Write-Output ("Gemini frontend already running. root_pid={0} session={1}" -f $status.root_pid, $status.session)
        return
    }

    if ($status.running -and $ForceNew) {
        Close-Frontend | Out-Null
    }

    $boot = @"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location -LiteralPath '$repoRoot'
`$Host.UI.RawUI.WindowTitle = '$Title'
Write-Host 'Gemini frontend attached to workspace:' '$repoRoot'
Write-Host 'Session:' '$Session'
Write-Host 'Close this window manually when finished, or run gemini_frontend.ps1 close from another shell.'
& '$geminiPath' --resume $Session
"@

    $process = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $boot -WindowStyle Normal -PassThru
    Save-State -RootPid $process.Id -Workspace $repoRoot -SessionName $Session -WindowTitle $Title
    Write-Output ("Opened Gemini frontend. root_pid={0} session={1}" -f $process.Id, $Session)
}

function Close-Frontend {
    $state = Get-State
    if ($null -eq $state) {
        Write-Output "Gemini frontend is not running."
        return
    }

    $rootPid = [int]$state.root_pid
    $rootProcess = Get-Process -Id $rootPid -ErrorAction SilentlyContinue
    if ($null -eq $rootProcess) {
        Clear-State
        Write-Output ("Cleared stale Gemini frontend state. root_pid={0}" -f $rootPid)
        return
    }

    $pids = Get-DescendantProcessIds -RootPid $rootPid | Sort-Object -Descending
    foreach ($targetPid in $pids) {
        Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
    }
    Clear-State
    Write-Output ("Closed Gemini frontend. root_pid={0}" -f $rootPid)
}

switch ($Action) {
    "open" {
        Open-Frontend
    }
    "close" {
        Close-Frontend
    }
    "status" {
        Get-FrontendStatus | ConvertTo-Json -Depth 5
    }
    "sessions" {
        & $geminiPath --list-sessions
    }
}
