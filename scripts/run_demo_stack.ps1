param(
  [int]$ApiPort = 8017,
  [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Host "OncoTrack demo stack"
Write-Host "Backend:  http://127.0.0.1:$ApiPort"
Write-Host "Frontend: http://127.0.0.1:$FrontendPort/login"
Write-Host ""
Write-Host "Demo credentials:"
Write-Host "  Patient:   P001 / patient-demo"
Write-Host "  Clinician: clinician / clinician-demo"
Write-Host "  Admin:     admin / admin-demo"
Write-Host ""

$backend = Start-Process powershell -WindowStyle Hidden -PassThru -WorkingDirectory $Root -ArgumentList @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-Command",
  "python -m uvicorn backend.api.main:app --host 127.0.0.1 --port $ApiPort --reload"
)

$frontendRoot = Join-Path $Root "frontend-react"
$frontend = Start-Process powershell -WindowStyle Hidden -PassThru -WorkingDirectory $frontendRoot -ArgumentList @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-Command",
  "`$env:VITE_API_BASE='http://127.0.0.1:$ApiPort'; npm run dev -- --host 127.0.0.1 --port $FrontendPort"
)

Write-Host "Started backend PID $($backend.Id) and frontend PID $($frontend.Id)."
Write-Host "Close those PowerShell processes or stop their PIDs when done."

