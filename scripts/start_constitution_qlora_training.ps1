[CmdletBinding()]
param(
    [string]$Config = "configs/constitution_qlora.toml",
    [string]$DataDir = "data/constitution-analysis",
    [string]$Model = "Qwen/Qwen3-4B",
    [string]$OutputDir = "artifacts/qwen3-4b-constitution-qlora",
    [string]$CacheDir = "artifacts/hf_cache",
    [string]$ResumeFromCheckpoint,
    [switch]$AllowDownload
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $projectRoot "artifacts\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutLog = Join-Path $logDir "constitution_qlora_$timestamp.out.log"
$stderrLog = Join-Path $logDir "constitution_qlora_$timestamp.err.log"

$arguments = @(
    "run",
    "python",
    "-u",
    "scripts/train_constitution_qlora.py",
    "--config",
    $Config,
    "--data-dir",
    $DataDir,
    "--model",
    $Model,
    "--output-dir",
    $OutputDir,
    "--cache-dir",
    $CacheDir
)
if ($ResumeFromCheckpoint) {
    $arguments += "--resume-from-checkpoint", $ResumeFromCheckpoint
}
if ($AllowDownload) {
    $arguments += "--allow-download"
}

$hadPythonPath = Test-Path Env:PYTHONPATH
$previousPythonPath = $env:PYTHONPATH
$sourcePath = Join-Path $projectRoot "src"
$env:PYTHONPATH = if ($previousPythonPath) { "$sourcePath;$previousPythonPath" } else { $sourcePath }

try {
    $process = Start-Process `
        -FilePath "uv" `
        -ArgumentList $arguments `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -WindowStyle Hidden `
        -PassThru
}
finally {
    if ($hadPythonPath) {
        $env:PYTHONPATH = $previousPythonPath
    }
    else {
        Remove-Item Env:PYTHONPATH
    }
}

Write-Output "Constitution-analysis training started in the background and will continue after this terminal is closed."
Write-Output "PID: $($process.Id)"
Write-Output "Training log: $stdoutLog"
Write-Output "Error log: $stderrLog"
