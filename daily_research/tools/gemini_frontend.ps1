param(
    [ValidateSet("open", "close", "status", "sessions", "ask", "closeout")]
    [string]$Action = "status",
    [string]$Session = "latest",
    [string]$Title = "Gemini Frontend - daily_research",
    [switch]$ForceNew,
    [string]$Prompt = "",
    [string]$WorkSummary = "",
    [string]$NextStep = "",
    [string]$Model = "",
    [switch]$FreshSession,
    [switch]$Escalate
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$stateDir = Join-Path $repoRoot "daily_research\cache\gemini_frontend"
$statePath = Join-Path $stateDir "state.json"
$geminiPath = "C:\Users\ASUS\AppData\Roaming\npm\gemini.cmd"
$defaultMode = "background_resume"
$defaultEscalationMode = "fresh_session_plus_model"
$defaultEscalationModel = "gemini-3.1-pro-preview"
$freshSessionLabel = "fresh"

function Get-State {
    if (-not (Test-Path -LiteralPath $statePath)) {
        return $null
    }
    return Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
}

function Save-State(
    [int]$RootPid,
    [string]$Workspace,
    [string]$SessionName,
    [string]$WindowTitle,
    [string]$ModelName,
    [bool]$FreshSessionEnabled
) {
    New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
    $state = [ordered]@{
        root_pid      = $RootPid
        workspace     = $Workspace
        session       = $SessionName
        title         = $WindowTitle
        model         = $ModelName
        fresh_session = $FreshSessionEnabled
        launched_at   = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
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

function Get-RequestedFreshSession {
    return [bool]($FreshSession -or $Escalate)
}

function Resolve-EffectiveModel([string]$RequestedModel) {
    $modelName = ""
    if ($null -ne $RequestedModel) {
        $modelName = [string]$RequestedModel
    }
    if (-not [string]::IsNullOrWhiteSpace($modelName)) {
        return $modelName
    }
    if ($Escalate) {
        return $defaultEscalationModel
    }
    return ""
}

function Resolve-ResumeTarget([string]$PreferredSession) {
    $explicitSession = ""
    if ($null -ne $PreferredSession) {
        $explicitSession = [string]$PreferredSession
    }
    if (-not [string]::IsNullOrWhiteSpace($explicitSession) -and $explicitSession -ne "latest") {
        return $explicitSession
    }

    $state = Get-State
    if ($null -ne $state) {
        $stateSession = ""
        if ($null -ne $state.session) {
            $stateSession = [string]$state.session
        }
        if (
            -not [string]::IsNullOrWhiteSpace($stateSession) -and
            $stateSession -ne $freshSessionLabel
        ) {
            return $stateSession
        }
    }
    return "latest"
}

function Get-InvocationSpec {
    $useFreshSession = Get-RequestedFreshSession
    $effectiveModel = Resolve-EffectiveModel -RequestedModel $Model
    $resumeTarget = $null
    $sessionName = $freshSessionLabel

    if (-not $useFreshSession) {
        $resumeTarget = Resolve-ResumeTarget -PreferredSession $Session
        $sessionName = $resumeTarget
    }

    return [pscustomobject]@{
        use_fresh_session = $useFreshSession
        model             = $effectiveModel
        resume_target     = $resumeTarget
        session_name      = $sessionName
        escalate          = [bool]$Escalate
    }
}

function Build-GeminiArgs([pscustomobject]$Spec, [string]$PromptText) {
    $args = @()
    if (-not [string]::IsNullOrWhiteSpace([string]$Spec.model)) {
        $args += @("--model", [string]$Spec.model)
    }
    if (-not [bool]$Spec.use_fresh_session) {
        $args += @("--resume", [string]$Spec.resume_target)
    }
    if (-not [string]::IsNullOrWhiteSpace($PromptText)) {
        $args += @("-p", $PromptText)
    }
    return $args
}

function Get-FrontendStatus {
    $base = [ordered]@{
        default_mode              = $defaultMode
        escalation_mode           = $defaultEscalationMode
        default_escalation_model  = $defaultEscalationModel
        fresh_session_supported   = $true
        model_override_supported  = $true
        open_new_window_supported = $true
    }

    $state = Get-State
    if ($null -eq $state) {
        return [pscustomobject]($base + @{
            running       = $false
            message       = "No Gemini frontend state file."
            root_pid      = $null
            session       = $null
            workspace     = $repoRoot
            title         = $Title
            model         = $null
            fresh_session = $null
            processes     = @()
        })
    }

    $rootPid = [int]$state.root_pid
    $rootProcess = Get-Process -Id $rootPid -ErrorAction SilentlyContinue
    if ($null -eq $rootProcess) {
        Clear-State
        return [pscustomobject]($base + @{
            running       = $false
            message       = "Gemini frontend state was stale and has been cleared."
            root_pid      = $rootPid
            session       = $state.session
            workspace     = $state.workspace
            title         = $state.title
            model         = $state.model
            fresh_session = $state.fresh_session
            processes     = @()
        })
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

    return [pscustomobject]($base + @{
        running       = $true
        message       = "Gemini frontend is running."
        root_pid      = $rootPid
        session       = $state.session
        workspace     = $state.workspace
        title         = $state.title
        model         = $state.model
        fresh_session = $state.fresh_session
        processes     = @($processes | Sort-Object pid)
    })
}

function Open-Frontend {
    if (-not (Test-Path -LiteralPath $geminiPath)) {
        throw "Gemini CLI not found at $geminiPath"
    }

    $status = Get-FrontendStatus
    if ($status.running -and -not $ForceNew) {
        Write-Output (
            "Gemini frontend already running. root_pid={0} session={1} fresh_session={2} model={3}" -f
            $status.root_pid,
            $status.session,
            $status.fresh_session,
            $status.model
        )
        return
    }

    if ($status.running -and $ForceNew) {
        Close-Frontend | Out-Null
    }

    $spec = Get-InvocationSpec
    $resumeTarget = ""
    if ($null -ne $spec.resume_target) {
        $resumeTarget = [string]$spec.resume_target
    }
    $effectiveModel = ""
    if ($null -ne $spec.model) {
        $effectiveModel = [string]$spec.model
    }
    $sessionBanner = if ($spec.use_fresh_session) { "fresh" } else { $resumeTarget }
    $modelBanner = if ([string]::IsNullOrWhiteSpace($effectiveModel)) { "default" } else { $effectiveModel }

    $boot = @"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location -LiteralPath '$repoRoot'
`$Host.UI.RawUI.WindowTitle = '$Title'
`$geminiArgs = @()
if ('$effectiveModel' -ne '') {
    `$geminiArgs += @('--model', '$effectiveModel')
}
if ('$resumeTarget' -ne '') {
    `$geminiArgs += @('--resume', '$resumeTarget')
}
Write-Host 'Gemini frontend attached to workspace:' '$repoRoot'
Write-Host 'Session mode:' '$sessionBanner'
Write-Host 'Model:' '$modelBanner'
Write-Host 'This window is optional interactive mode only.'
Write-Host 'Codex may escalate to fresh session or 3.1 pro when stale context is detected.'
Write-Host 'Close this window manually when finished, or run gemini_frontend.ps1 close from another shell.'
& '$geminiPath' @geminiArgs
"@

    $process = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $boot -WindowStyle Normal -PassThru
    Save-State `
        -RootPid $process.Id `
        -Workspace $repoRoot `
        -SessionName $spec.session_name `
        -WindowTitle $Title `
        -ModelName $effectiveModel `
        -FreshSessionEnabled ([bool]$spec.use_fresh_session)
    Write-Output (
        "Opened Gemini frontend. root_pid={0} session={1} fresh_session={2} model={3}" -f
        $process.Id,
        $spec.session_name,
        $spec.use_fresh_session,
        $modelBanner
    )
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

function Invoke-FrontendPrompt {
    if ([string]::IsNullOrWhiteSpace($Prompt)) {
        throw "Prompt is required for Action=ask."
    }

    $spec = Get-InvocationSpec
    $args = Build-GeminiArgs -Spec $spec -PromptText $Prompt
    & $geminiPath @args
}

function Invoke-FrontendCloseout {
    if ([string]::IsNullOrWhiteSpace($WorkSummary)) {
        throw "WorkSummary is required for Action=closeout."
    }

    $nextLine = if ([string]::IsNullOrWhiteSpace($NextStep)) {
        "Next-step context: not specified; infer the most sensible next move."
    } else {
        "Next-step context: $NextStep"
    }

$closeoutPrompt = @"
You are the Gemini closeout reviewer for the current workspace.
We are about to send the user a final reply.
Treat the supplied summary below as the authoritative source of truth for this turn.
If any earlier session memory conflicts with the supplied summary, ignore the older memory.
Do not introduce stale experiments, outdated branches, or superseded execution paths unless they are explicitly mentioned in the supplied summary.

Completed work summary:
$WorkSummary

$nextLine

Reply in Chinese, concise and concrete.
Use exactly these three headings:
已完成：
下一步：
风险/待确认：

Under each heading, provide 1-3 short bullet points.
Focus on confirming what has already been done and what we should do next.
"@

    $script:Prompt = $closeoutPrompt
    Invoke-FrontendPrompt
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
    "ask" {
        Invoke-FrontendPrompt
    }
    "closeout" {
        Invoke-FrontendCloseout
    }
}
