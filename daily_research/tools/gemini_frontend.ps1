param(
    [ValidateSet("open", "close", "status", "sessions", "ask", "closeout", "doctor", "pin", "unpin")]
    [string]$Action = "status",
    [string]$Session = "latest",
    [string]$Title = "Gemini Frontend - daily_research",
    [switch]$ForceNew,
    [string]$Prompt = "",
    [string]$WorkSummary = "",
    [string]$NextStep = "",
    [string]$Model = "",
    [switch]$FreshSession,
    [switch]$Escalate,
    [int]$TimeoutSec = 60
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$stateDir = Join-Path $repoRoot "daily_research\cache\gemini_frontend"
$statePath = Join-Path $stateDir "state.json"
$prefsPath = Join-Path $stateDir "preferences.json"
$moduleState = "suspended"
$suspendedAt = "2026-03-29"
$suspensionMessage = "Gemini collaboration module is temporarily suspended by user request on 2026-03-29. Do not use it in the default workflow."

function Ensure-StateDir {
    New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
}

function Get-State {
    if (-not (Test-Path -LiteralPath $statePath)) {
        return $null
    }
    return Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
}

function Clear-StateFile {
    if (Test-Path -LiteralPath $statePath) {
        Remove-Item -LiteralPath $statePath -Force
    }
}

function Clear-PreferencesFile {
    if (Test-Path -LiteralPath $prefsPath) {
        Remove-Item -LiteralPath $prefsPath -Force
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

function Stop-ProcessTree([int]$RootPid) {
    $pids = Get-DescendantProcessIds -RootPid $RootPid | Sort-Object -Descending
    foreach ($targetPid in $pids) {
        Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
    }
}

function Get-SuspensionStatus {
    $state = Get-State
    return [pscustomobject]@{
        module_state         = $moduleState
        suspended_at         = $suspendedAt
        message              = $suspensionMessage
        workspace            = $repoRoot
        running              = $false
        root_pid             = if ($null -ne $state) { $state.root_pid } else { $null }
        session              = if ($null -ne $state) { $state.session } else { $null }
        title                = $Title
        cached_state_exists  = (Test-Path -LiteralPath $statePath)
        cached_prefs_exists  = (Test-Path -LiteralPath $prefsPath)
        allowed_actions      = @("status", "close")
        blocked_actions      = @("open", "sessions", "ask", "closeout", "doctor", "pin", "unpin")
    }
}

function Close-SuspendedModule {
    $state = Get-State
    if ($null -ne $state -and $null -ne $state.root_pid) {
        $rootPid = [int]$state.root_pid
        $rootProcess = Get-Process -Id $rootPid -ErrorAction SilentlyContinue
        if ($null -ne $rootProcess) {
            Stop-ProcessTree -RootPid $rootPid
        }
    }
    Clear-StateFile
    Clear-PreferencesFile
    Write-Output "Gemini collaboration module is suspended. Local Gemini frontend cache has been cleared."
}

function Throw-SuspendedAction([string]$RequestedAction) {
    throw "$suspensionMessage Action '$RequestedAction' is disabled until the module is explicitly restarted."
}

switch ($Action) {
    "status" {
        Get-SuspensionStatus | ConvertTo-Json -Depth 4
    }
    "close" {
        Close-SuspendedModule
    }
    default {
        Throw-SuspendedAction -RequestedAction $Action
    }
}
