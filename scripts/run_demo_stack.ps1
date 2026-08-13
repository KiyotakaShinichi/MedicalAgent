param(
  [int]$ApiPort = 8017,
  [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

function Assert-PortAvailable {
  param(
    [int]$Port,
    [string]$ServiceName
  )

  $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if ($null -ne $listener) {
    throw "$ServiceName port $Port is already occupied by PID $($listener.OwningProcess). Stop that process or choose another port."
  }
}

function Wait-ForHttpReady {
  param(
    [string]$Url,
    [System.Diagnostics.Process]$Process,
    [string]$ServiceName,
    [int]$TimeoutSeconds = 120
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    if ($Process.HasExited) {
      throw "$ServiceName exited before becoming ready (exit code $($Process.ExitCode))."
    }
    try {
      $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
      if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
        return
      }
    } catch {
      Start-Sleep -Milliseconds 750
    }
  }
  throw "$ServiceName did not become ready at $Url within $TimeoutSeconds seconds."
}

Assert-PortAvailable -Port $ApiPort -ServiceName "Backend"
Assert-PortAvailable -Port $FrontendPort -ServiceName "Frontend"

Write-Host "NLCare demo stack"
Write-Host "Backend:  http://127.0.0.1:$ApiPort"
Write-Host "Frontend: http://127.0.0.1:$FrontendPort/login"
Write-Host ""
Write-Host "Demo credentials:"
Write-Host "  Patient:   P001 / patient-demo"
Write-Host "  Patient 2: P002 / patient-demo"
Write-Host "  Clinician: clinician / clinician-demo"
Write-Host "  Admin:     admin / admin-demo"
Write-Host ""

$bootstrapVariables = @{
  "NLCARE_BOOTSTRAP_SYNTHETIC_DEMO" = "true"
  "NLCARE_SYNTHETIC_ONLY" = "true"
  "NLCARE_DATA_CLASSIFICATION" = "synthetic"
  "ENVIRONMENT" = "development"
}
$previousValues = @{}
try {
  foreach ($entry in $bootstrapVariables.GetEnumerator()) {
    $previousValues[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, "Process")
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
  }
  & python -m scripts.bootstrap_synthetic_demo
  if ($LASTEXITCODE -ne 0) {
    throw "Synthetic demo bootstrap failed with exit code $LASTEXITCODE."
  }
} finally {
  foreach ($entry in $previousValues.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
  }
}

$backend = Start-Process powershell -WindowStyle Hidden -PassThru -WorkingDirectory $Root -ArgumentList @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-Command",
  "`$env:NLCARE_PATIENT_ENRICHMENT_PREWARM_ENABLED='false'; python -m uvicorn backend.api.main:app --host 127.0.0.1 --port $ApiPort"
)

$frontendRoot = Join-Path $Root "frontend-react"
$frontend = Start-Process powershell -WindowStyle Hidden -PassThru -WorkingDirectory $frontendRoot -ArgumentList @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-Command",
  "`$env:VITE_API_BASE='http://127.0.0.1:$ApiPort'; npm run dev -- --host 127.0.0.1 --port $FrontendPort"
)

try {
  Wait-ForHttpReady -Url "http://127.0.0.1:$ApiPort/health" -Process $backend -ServiceName "Backend"
  Wait-ForHttpReady -Url "http://127.0.0.1:$FrontendPort/login" -Process $frontend -ServiceName "Frontend"
} catch {
  foreach ($process in @($backend, $frontend)) {
    if ($null -ne $process -and -not $process.HasExited) {
      Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
  }
  throw
}

Write-Host "Started backend PID $($backend.Id) and frontend PID $($frontend.Id)."
Write-Host "Both services passed their HTTP readiness checks."
Write-Host "Close those PowerShell processes or stop their PIDs when done."
