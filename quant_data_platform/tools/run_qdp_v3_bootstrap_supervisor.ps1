[CmdletBinding()]
param(
    [string]$WorkspaceRoot = "H:\quant_project",
    [string]$BootstrapStart = "2010-01-01",
    [string]$BootstrapCutoff = "2026-07-13",
    [switch]$Once,
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Python = "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
$Contract = "qdp_v3_20260715_trusted_source_5m"
$MinimumFreeBytes = 200GB
$QuotaResumeMinute = 10
$TransientRetrySeconds = 300
$MutexName = "Local\QDPV3Trusted5MBootstrapSupervisor"
$script:ProxyToken = ""
$script:SupervisorRoot = ""
$script:LogRoot = ""
$script:StatePath = ""

function Get-UserEnvironmentValue {
    param([Parameter(Mandatory = $true)][string]$Name)

    $value = [Environment]::GetEnvironmentVariable($Name, "User")
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "required_user_environment_missing:$Name"
    }
    return $value
}

function Write-AtomicJson {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )

    $directory = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $temporary = Join-Path $directory (".{0}.{1}.tmp" -f ([IO.Path]::GetFileName($Path)), [Guid]::NewGuid().ToString("N"))
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Protect-OutputFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Secret
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    [string]$text = Get-Content -LiteralPath $Path -Raw -ErrorAction Stop
    if ($text.Contains($Secret)) {
        $text.Replace($Secret, "<redacted>") | Set-Content -LiteralPath $Path -Encoding utf8
    }
}

function Write-SupervisorState {
    param(
        [Parameter(Mandatory = $true)][string]$Status,
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][int]$Attempt,
        [hashtable]$Extra = @{}
    )

    $payload = @{
        schema_version = 1
        contract = $Contract
        status = $Status
        phase = $Phase
        attempt = $Attempt
        pid = $PID
        workspace_root = $WorkspaceRoot
        bootstrap_start = $BootstrapStart
        bootstrap_cutoff = $BootstrapCutoff
        heartbeat_at = [DateTimeOffset]::Now.ToString("o")
    }
    foreach ($key in $Extra.Keys) {
        $payload[$key] = $Extra[$key]
    }
    Write-AtomicJson -Path $script:StatePath -Value $payload
}

function Get-LatestCompletedDate {
    $now = Get-Date
    if ($now.Hour -lt 16) {
        return $now.Date.AddDays(-1).ToString("yyyy-MM-dd")
    }
    return $now.Date.ToString("yyyy-MM-dd")
}

function Wait-WithHeartbeat {
    param(
        [Parameter(Mandatory = $true)][DateTime]$Until,
        [Parameter(Mandatory = $true)][string]$Reason,
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][int]$Attempt
    )

    while ((Get-Date) -lt $Until) {
        $remaining = [Math]::Max(0, [int][Math]::Ceiling(($Until - (Get-Date)).TotalSeconds))
        Write-SupervisorState -Status $Reason -Phase $Phase -Attempt $Attempt -Extra @{
            resume_at = $Until.ToString("o")
            remaining_seconds = $remaining
        }
        Start-Sleep -Seconds ([Math]::Min(60, [Math]::Max(1, $remaining)))
    }
}

function Invoke-QdpUpdate {
    param(
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][int]$Attempt
    )

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdoutPath = Join-Path $script:LogRoot ("{0}_attempt_{1:D3}_{2}.json" -f $Phase, $Attempt, $stamp)
    $stderrPath = Join-Path $script:LogRoot ("{0}_attempt_{1:D3}_{2}.stderr.log" -f $Phase, $Attempt, $stamp)
    $arguments = @(
        "-m", "quant_data_platform.cli",
        "--generation", "v3",
        "update",
        "--workspace-root", $WorkspaceRoot,
        "--as-of-date"
    )
    if ($Phase -eq "bootstrap") {
        $arguments += @(
            $BootstrapCutoff,
            "--bootstrap",
            "--historical-provider", "tushare-proxy",
            "--start-date", $BootstrapStart,
            "--json"
        )
    }
    else {
        $latestCompleted = Get-LatestCompletedDate
        $arguments += @($latestCompleted, "--json")
    }

    Write-SupervisorState -Status "running" -Phase $Phase -Attempt $Attempt -Extra @{
        stdout_path = $stdoutPath
        stderr_path = $stderrPath
    }
    & $Python @arguments 1> $stdoutPath 2> $stderrPath
    $exitCode = $LASTEXITCODE
    Protect-OutputFile -Path $stdoutPath -Secret $script:ProxyToken
    Protect-OutputFile -Path $stderrPath -Secret $script:ProxyToken
    if ($exitCode -ne 0) {
        return [pscustomobject]@{
            status = "process_failed"
            exit_code = $exitCode
            stdout_path = $stdoutPath
            stderr_path = $stderrPath
            run_id = ""
            stage_count = 0
        }
    }
    try {
        $payload = Get-Content -LiteralPath $stdoutPath -Raw | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        return [pscustomobject]@{
            status = "invalid_json"
            exit_code = $exitCode
            stdout_path = $stdoutPath
            stderr_path = $stderrPath
            run_id = ""
            stage_count = 0
        }
    }
    return [pscustomobject]@{
        status = [string]$payload.status
        exit_code = $exitCode
        stdout_path = $stdoutPath
        stderr_path = $stderrPath
        run_id = [string]$payload.run_id
        stage_count = @($payload.stages).Count
    }
}

$mutex = New-Object System.Threading.Mutex($false, $MutexName)
$ownsMutex = $false
try {
    $ownsMutex = $mutex.WaitOne(0)
    if (-not $ownsMutex) {
        throw "qdp_v3_bootstrap_supervisor_already_running"
    }

    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "fixed_python_missing:$Python"
    }
    $resolvedWorkspace = [IO.Path]::GetFullPath($WorkspaceRoot)
    if ($resolvedWorkspace -ne [IO.Path]::GetFullPath("H:\quant_project")) {
        throw "unexpected_workspace_root:$resolvedWorkspace"
    }

    $env:QDP_DATA_ROOT = Get-UserEnvironmentValue -Name "QDP_DATA_ROOT"
    $env:QDP_RUNTIME_ROOT = Get-UserEnvironmentValue -Name "QDP_RUNTIME_ROOT"
    $env:QDP_TUSHARE_PROXY_URL = Get-UserEnvironmentValue -Name "QDP_TUSHARE_PROXY_URL"
    $script:ProxyToken = Get-UserEnvironmentValue -Name "QDP_TUSHARE_PROXY_TOKEN"
    $env:QDP_TUSHARE_PROXY_TOKEN = $script:ProxyToken
    $env:PYTHONPATH = Join-Path $resolvedWorkspace "quant_data_platform\src"

    $script:SupervisorRoot = Join-Path $env:QDP_RUNTIME_ROOT "bootstrap_supervisor"
    $script:LogRoot = Join-Path $script:SupervisorRoot "attempts"
    $script:StatePath = Join-Path $script:SupervisorRoot "state.json"
    New-Item -ItemType Directory -Path $script:LogRoot -Force | Out-Null

    if ($ValidateOnly) {
        Write-SupervisorState -Status "validated" -Phase "bootstrap" -Attempt 0 -Extra @{
            fixed_python = $Python
            data_root = $env:QDP_DATA_ROOT
            runtime_root = $env:QDP_RUNTIME_ROOT
        }
        exit 0
    }

    $phase = "bootstrap"
    $attempt = 0
    $consecutiveFailures = 0
    while ($true) {
        $drive = Get-PSDrive -Name H
        if ([int64]$drive.Free -lt [int64]$MinimumFreeBytes) {
            $until = (Get-Date).AddMinutes(10)
            Wait-WithHeartbeat -Until $until -Reason "paused_disk" -Phase $phase -Attempt $attempt
            continue
        }

        $attempt += 1
        $result = Invoke-QdpUpdate -Phase $phase -Attempt $attempt
        Write-SupervisorState -Status $result.status -Phase $phase -Attempt $attempt -Extra @{
            exit_code = $result.exit_code
            run_id = $result.run_id
            stage_count = $result.stage_count
            stdout_path = $result.stdout_path
            stderr_path = $result.stderr_path
        }
        if ($Once) {
            exit 0
        }

        if ($phase -eq "bootstrap" -and $result.status -eq "published_awaiting_incremental_update") {
            $phase = "incremental"
            $attempt = 0
            $consecutiveFailures = 0
            continue
        }
        if ($phase -eq "incremental" -and $result.status -eq "completed") {
            Write-SupervisorState -Status "completed" -Phase $phase -Attempt $attempt -Extra @{
                run_id = $result.run_id
                stdout_path = $result.stdout_path
                stderr_path = $result.stderr_path
            }
            exit 0
        }
        if ($result.status -eq "paused_quota") {
            $resumeAt = (Get-Date).Date.AddDays(1).AddMinutes($QuotaResumeMinute)
            Wait-WithHeartbeat -Until $resumeAt -Reason "waiting_quota_reset" -Phase $phase -Attempt $attempt
            $consecutiveFailures = 0
            continue
        }
        if ($result.status -in @("paused_rate_limit_quality_gate", "paused_disk", "partial", "process_failed", "invalid_json", "failed")) {
            $consecutiveFailures += 1
            if ($consecutiveFailures -ge 3) {
                Write-SupervisorState -Status "blocked_after_retries" -Phase $phase -Attempt $attempt -Extra @{
                    last_status = $result.status
                    stdout_path = $result.stdout_path
                    stderr_path = $result.stderr_path
                }
                exit 1
            }
            $until = (Get-Date).AddSeconds($TransientRetrySeconds)
            Wait-WithHeartbeat -Until $until -Reason "waiting_transient_retry" -Phase $phase -Attempt $attempt
            continue
        }

        Write-SupervisorState -Status "blocked_unexpected_terminal_status" -Phase $phase -Attempt $attempt -Extra @{
            last_status = $result.status
            stdout_path = $result.stdout_path
            stderr_path = $result.stderr_path
        }
        exit 1
    }
}
catch {
    $message = [string]$_.Exception.Message
    if ($script:ProxyToken -and $message.Contains($script:ProxyToken)) {
        $message = $message.Replace($script:ProxyToken, "<redacted>")
    }
    if ($script:StatePath) {
        Write-SupervisorState -Status "failed" -Phase "bootstrap" -Attempt 0 -Extra @{
            error_type = $_.Exception.GetType().Name
            error = $message
        }
    }
    exit 1
}
finally {
    if ($ownsMutex) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
