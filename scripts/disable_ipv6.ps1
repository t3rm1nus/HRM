# Script to disable IPv6 on Wi-Fi adapter
try {
    $adapter = Get-NetAdapter -Name "Wi-Fi" | Where-Object { $_.Status -eq "Up" } | Select-Object -First 1
    
    if ($adapter) {
        Write-Host "Desactivando IPv6 para adaptador: $($adapter.Name)" -ForegroundColor Green
        
        # Desactivar IPv6
        Disable-NetAdapterBinding -InterfaceIndex $adapter.ifIndex -ComponentID ms_tcpip6
        
        Write-Host "IPv6 desactivado exitosamente!" -ForegroundColor Green
        
        # Verificar el estado
        $ipv6Binding = Get-NetAdapterBinding -InterfaceIndex $adapter.ifIndex -ComponentID ms_tcpip6
        Write-Host "Estado IPv6: $($ipv6Binding.Enabled)" -ForegroundColor Cyan
    } else {
        Write-Host "No se encontró adaptador Wi-Fi activo" -ForegroundColor Red
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Ejecute PowerShell como Administrador" -ForegroundColor Yellow
}