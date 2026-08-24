<#
    One-line installer for bkht-coder, Windows edition.

        irm https://thebkht.com/install.ps1 | iex

    Installs uv, Ollama and the model if they are missing, then puts `coder` on
    PATH. Every step is skipped when it is already satisfied, so re-running
    this is an upgrade rather than an error.

    Environment:
      MODEL                   model tag to pull (default: picked from host RAM)
      OLLAMA_HOST_URL         where the server should answer
      BKHT_CODER_REF          git branch/tag to install instead of the default
      BKHT_CODER_NO_MODEL=1   skip the model pull entirely
      BKHT_CODER_YES=1        assume yes; don't prompt
#>

$ErrorActionPreference = 'Stop'

if ($PSVersionTable.PSVersion.Major -lt 5) {
    throw 'bkht-coder needs PowerShell 5.1 or newer.'
}

$RepoUrl      = 'git+https://github.com/thebkht/bkht-coder.git'
$ModelDefault = 'qwen2.5-coder:14b'
$ModelSmall   = 'qwen2.5-coder:7b'
# Not $Host -- that name is taken by PowerShell's own console object.
$HostUrl      = if ($env:OLLAMA_HOST_URL) { $env:OLLAMA_HOST_URL } else { 'http://127.0.0.1:11434' }

$UvInstaller  = 'https://astral.sh/uv/install.ps1'
$OllamaDownload = 'https://ollama.com/download'

function Write-Pass($m) { Write-Host 'PASS ' -ForegroundColor Green -NoNewline; Write-Host $m }
function Write-Step($m) { Write-Host '==> ' -ForegroundColor Blue  -NoNewline; Write-Host $m }
function Write-Warn($m) { Write-Host 'WARN ' -ForegroundColor Yellow -NoNewline; Write-Host $m }
function Write-Fail($m) { Write-Host 'FAIL ' -ForegroundColor Red   -NoNewline; Write-Host $m; throw 'install aborted' }

function Test-Have($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

function Test-Server {
    try {
        Invoke-WebRequest -Uri "$HostUrl/api/tags" -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    } catch { return $false }
}

function Wait-Server($seconds) {
    for ($i = 0; $i -lt $seconds; $i++) {
        if (Test-Server) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

# ---------------------------------------------------------------- preflight

# $IsWindows only exists on PowerShell Core; on 5.1 its absence means Windows.
if ((Test-Path Variable:IsWindows) -and -not $IsWindows) {
    Write-Fail 'this script is the Windows one. On Linux and macOS use:
     curl -fsSL https://thebkht.com/install.sh | sh'
}

if (-not (Test-Have git)) {
    Write-Fail 'git not on PATH. Install Git for Windows and re-run:
     winget install --id Git.Git -e'
}

# ------------------------------------------------------------------ consent

$ramGb = 0
try {
    $ramGb = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
} catch { }

if ($env:MODEL) {
    $ModelTag = $env:MODEL
} elseif ($ramGb -gt 0 -and $ramGb -lt 12) {
    $ModelTag = $ModelSmall
    Write-Warn "detected $ramGb GB of RAM - using $ModelSmall instead of $ModelDefault"
} else {
    $ModelTag = $ModelDefault
}

# A server that already answers is the whole job done: don't install a second
# Ollama next to it.
$serverUp = Test-Server

# The CLI reads OLLAMA_HOST, not OLLAMA_HOST_URL. Without this the `ollama`
# commands below would talk to localhost while everything else talks to $HostUrl.
$env:OLLAMA_HOST = $HostUrl

$needUv     = -not (Test-Have uv)
$needOllama = (-not $serverUp) -and (-not (Test-Have ollama))
$skipModel  = $env:BKHT_CODER_NO_MODEL -eq '1'

Write-Host "`nbkht-coder installer`n"
Write-Host 'This will install:'
if ($needUv)     { Write-Host "  * uv          ($UvInstaller)" }
if ($needOllama) { Write-Host "  * Ollama      (winget, or $OllamaDownload)" }
if ($skipModel)  { Write-Host '  * the model   (skipped: BKHT_CODER_NO_MODEL=1)' }
else             { Write-Host "  * $ModelTag   (several GB, if not already pulled)" }
Write-Host "  * coder       (uv tool install $RepoUrl)"
if ($serverUp)   { Write-Host "`nAn Ollama server is already answering at $HostUrl - using it as is." }
Write-Host ''

if ($env:BKHT_CODER_YES -eq '1') {
    Write-Step 'BKHT_CODER_YES=1 - proceeding without asking'
} else {
    $reply = Read-Host 'Continue? [y/N]'
    if ($reply -notmatch '^(y|Y|yes|YES)$') {
        Write-Host "`nAborted."
        return
    }
}

# ----------------------------------------------------------------------- uv

if (Test-Have uv) {
    Write-Pass "uv already installed ($(uv --version))"
} else {
    Write-Step 'installing uv'
    try {
        Invoke-RestMethod $UvInstaller | Invoke-Expression
    } catch {
        Write-Fail "uv install failed - see https://docs.astral.sh/uv/`n     $_"
    }
    # The installer edits the user PATH in the registry; this process was
    # started before that and never sees it.
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Test-Have uv)) { Write-Fail 'uv installed but not on PATH - open a new terminal and re-run' }
    Write-Pass 'uv installed'
}

# --------------------------------------------------------- ollama and server

if ($serverUp) {
    Write-Pass "ollama already reachable at $HostUrl"
} else {
    if (Test-Have ollama) {
        Write-Pass 'ollama already installed'
    } else {
        Write-Step 'installing ollama'
        if (Test-Have winget) {
            winget install --id Ollama.Ollama -e --source winget `
                --accept-source-agreements --accept-package-agreements
            if ($LASTEXITCODE -ne 0) { Write-Fail "winget install Ollama.Ollama failed - install it from $OllamaDownload and re-run" }
        } else {
            Write-Fail "winget not available. Install Ollama from $OllamaDownload, then re-run."
        }
        $env:Path = "$env:LOCALAPPDATA\Programs\Ollama;$env:Path"
        if (-not (Test-Have ollama)) { Write-Fail 'ollama installed but not on PATH - open a new terminal and re-run' }
        Write-Pass 'ollama installed'
    }

    # The Windows installer registers a background service, so the server
    # usually comes up on its own; only start one if it doesn't.
    if (-not (Wait-Server 15)) {
        Write-Step 'starting ollama serve in the background'
        Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden | Out-Null
        if (-not (Wait-Server 30)) {
            Write-Fail "ollama did not answer at $HostUrl after 30s.
     Start it yourself and re-run:  ollama serve"
        }
    }
    Write-Pass "ollama reachable at $HostUrl"
}

# -------------------------------------------------------------------- model

if ($skipModel) {
    Write-Warn "skipping the model pull (BKHT_CODER_NO_MODEL=1) - 'ollama pull $ModelTag' before first use"
} elseif (Test-Have ollama) {
    $present = @(ollama list | Select-Object -Skip 1 |
        Where-Object { ($_ -split '\s+')[0] -eq $ModelTag })
    if ($present.Count -gt 0) {
        Write-Pass "model $ModelTag already pulled"
    } else {
        Write-Step "pulling $ModelTag - this is a multi-gigabyte download"
        ollama pull $ModelTag
        if ($LASTEXITCODE -ne 0) { Write-Fail "pull failed - retry with:  ollama pull $ModelTag" }
        Write-Pass "model $ModelTag pulled"
    }
} else {
    # A remote server with no local CLI: the same two operations over HTTP.
    $tags = Invoke-RestMethod -Uri "$HostUrl/api/tags" -TimeoutSec 10
    if ($tags.models.name -contains $ModelTag) {
        Write-Pass "model $ModelTag already on the server at $HostUrl"
    } else {
        Write-Step "pulling $ModelTag on the server at $HostUrl - several GB, and this prints nothing until it finishes"
        try {
            $body = @{ model = $ModelTag; stream = $false } | ConvertTo-Json
            $result = Invoke-RestMethod -Uri "$HostUrl/api/pull" -Method Post `
                -ContentType 'application/json' -Body $body -TimeoutSec 0
            if ($result.status -ne 'success') { throw "server said: $($result.status)" }
        } catch {
            Write-Fail "pull failed on $HostUrl - pull it there with:  ollama pull $ModelTag`n     $_"
        }
        Write-Pass "model $ModelTag pulled"
    }
}

# -------------------------------------------------------------------- coder

$target = $RepoUrl
if ($env:BKHT_CODER_REF) { $target = "$RepoUrl@$($env:BKHT_CODER_REF)" }

Write-Step "installing coder ($target)"
uv tool install --force $target
if ($LASTEXITCODE -ne 0) { Write-Fail "uv tool install failed for $target" }
Write-Pass 'coder installed'

if (-not (Test-Have coder)) {
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

if (Test-Have coder) {
    coder --help | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Fail "coder is on PATH but 'coder --help' failed" }
    Write-Pass 'coder --help'
} else {
    Write-Warn 'coder is installed but not on this terminal''s PATH.'
    Write-Host "     Run 'uv tool update-shell', or open a new terminal."
}

# --------------------------------------------------------------------- done

# coder defaults to localhost, so a server anywhere else has to be named on
# every run - say so here rather than let the first session fail.
$hostFlag = ''
if ($HostUrl -notin @('http://localhost:11434', 'http://127.0.0.1:11434')) {
    $hostFlag = " --host `"$HostUrl`""
}

Write-Host @"

Done. Try:

  coder$hostFlag
      an interactive session in the current directory
  coder$hostFlag "add a --verbose flag"
      one task, then exit
  coder --help
      every flag

Model:   $ModelTag
Server:  $HostUrl
State:   %USERPROFILE%\.bkht-coder\sessions\

"@
