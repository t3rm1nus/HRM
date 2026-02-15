# Script to change DNS servers to Cloudflare (1.1.1.1) and Google (8.8.8.8)
try {
    $adapter = Get-NetAdapter -Name "Wi-Fi" | Where-Object { $_.Status -eq "Up" } | Select-Object -First 1
    
    if ($adapter) {
        Write-Host "Cambiando DNS para adaptador: $($adapter.Name)" -ForegroundColor Green
        
        # Cambiar DNS a 1.1.1.1 y 8.8.8.8
        Set-DnsClientServerAddress -InterfaceIndex $adapter.ifIndex -ServerAddresses @("1.1.1.1", "8.8.8.8")
        
        Write-Host "DNS cambiado exitosamente!" -ForegroundColor Green
        
        # Verificar la configuración
        $currentDns = Get-DnsClientServerAddress -InterfaceIndex $adapter.ifIndex -AddressFamily IPv4
        Write-Host "DNS actuales: $($currentDns.ServerAddresses -join ', ')" -ForegroundColor Cyan
    } else {
        Write-Host "No se encontró adaptador Wi-Fi activo" -ForegroundColor Red
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Ejecute PowerShell como Administrador" -ForegroundColor Yellow
}