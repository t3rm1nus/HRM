#!/usr/bin/env pwsh
# Script de inicio para HRM Trading System
# Configuración inicial: 3000 USDT, 0 BTC, 0 ETH, modo paper

$ErrorActionPreference = "Stop"

Write-Host @"
╔══════════════════════════════════════════════════════════════════╗
║                    HRM TRADING SYSTEM                            ║
║                     Initial Startup Script                       ║
╚══════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Configuración inicial
$InitialState = @{
    capital_usdt = 3000.0
    btc = 0.0
    eth = 0.0
    mode = "paper"
    auto_learning = "fix"
    initial_balances = @{
        USDT = 3000.0
        BTC = 0.0
        ETH = 0.0
    }
    timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss.fffffff")
    reset_singletons = $true
}

Write-Host "📋 Configuration:" -ForegroundColor Yellow
Write-Host "   Capital USD: $($InitialState.capital_usdt)"
Write-Host "   BTC: $($InitialState.btc)"
Write-Host "   ETH: $($InitialState.eth)"
Write-Host "   Mode: $($InitialState.mode)"
Write-Host "   Auto-Learning: $($InitialState.auto_learning)"
Write-Host "   Reset Singletons: $($InitialState.reset_singletons)"
Write-Host ""

# Guardar configuración inicial
$InitialState | ConvertTo-Json -Depth 10 | Set-Content -Path "initial_state.json" -Encoding UTF8
Write-Host "✅ Initial state saved to initial_state.json" -ForegroundColor Green

# Limpiar archivos de estado previos
Write-Host "🧹 Cleaning previous state files..." -ForegroundColor Yellow
$filesToClean = @(
    "persistent_state\*.json",
    "persistent_state\*.bak",
    "portfolio_state*.json",
    "paper_trades\*.json",
    "global_system_state.json"
)

foreach ($pattern in $filesToClean) {
    $files = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        Remove-Item -Path $file.FullName -Force
        Write-Host "   🗑️  Removed: $($file.Name)" -ForegroundColor DarkGray
    }
}
Write-Host "✅ Cleanup complete" -ForegroundColor Green
Write-Host ""

# Verificar entorno virtual
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "❌ Virtual environment not found. Creating..." -ForegroundColor Red
    python -m venv .venv
}

# Activar entorno virtual
Write-Host "🐍 Activating virtual environment..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1

# Verificar dependencias
Write-Host "📦 Checking dependencies..." -ForegroundColor Yellow
pip install -q python-dotenv colorama pandas numpy aiohttp websockets

Write-Host ""
Write-Host "🚀 Starting HRM System..." -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Ejecutar el sistema
python main.py

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "👋 HRM System stopped" -ForegroundColor Yellow
