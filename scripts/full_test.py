import subprocess
import requests
import time

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def test_dns_resolution():
    print("=== TESTING DNS RESOLUTION ===")
    print("1. Resolviendo api.binance.com con DNS actual...")
    
    # Test nslookup
    return_code, stdout, stderr = run_command('nslookup api.binance.com')
    print(f"Resultado nslookup: {return_code}")
    if stdout:
        print("Salida:")
        print(stdout)
    if stderr:
        print("Error:")
        print(stderr)
    
    # Test con dig (si está disponible)
    print("\n2. Resolviendo con DNS 1.1.1.1 directamente...")
    return_code, stdout, stderr = run_command('nslookup api.binance.com 1.1.1.1')
    print(f"Resultado: {return_code}")
    if stdout:
        print("Salida:")
        print(stdout)
    
    print()

def test_api_connection():
    print("=== TESTING API CONNECTION ===")
    print("Conectando a Binance API...")
    
    try:
        response = requests.get('https://api.binance.com/api/v3/ping', timeout=5)
        print(f"✅ Conexión exitosa: {response.status_code}")
        print(f"Respuesta: {response.text}")
        print(f"Tiempo de respuesta: {response.elapsed.total_seconds():.2f} segundos")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_ipv6_status():
    print("=== TESTING IPv6 STATUS ===")
    
    # Check if IPv6 is enabled
    return_code, stdout, stderr = run_command('Get-NetIPv6Protocol')
    if return_code == 0:
        print("✅ IPv6 está activado globalmente")
    else:
        print("❌ IPv6 está desactivado globalmente")
    
    # Check adapter status
    print("\nEstado de IPv6 en adaptador Wi-Fi:")
    return_code, stdout, stderr = run_command('Get-NetAdapterBinding -Name "Wi-Fi" -ComponentID ms_tcpip6')
    if stdout:
        lines = stdout.strip().split('\n')
        for line in lines:
            if 'Enabled' in line:
                parts = line.strip().split()
                if len(parts) >= 2:
                    status = parts[-1]
                    if status == 'True':
                        print("✅ IPv6 está activado en Wi-Fi")
                    else:
                        print("❌ IPv6 está desactivado en Wi-Fi")
    
    print()

def test_multiple_attempts(count=5):
    print(f"=== TESTING {count} CONNECTIONS ===")
    successes = 0
    failures = 0
    
    for i in range(count):
        print(f"Intento {i+1}: ", end="", flush=True)
        try:
            response = requests.get('https://api.binance.com/api/v3/ping', timeout=5)
            if response.status_code == 200:
                print("✅")
                successes += 1
            else:
                print(f"❌ Status {response.status_code}")
                failures += 1
        except Exception as e:
            print(f"❌ {type(e).__name__}")
            failures += 1
        
        time.sleep(0.5)
    
    print(f"\nResumen: {successes} exitosas, {failures} fallidas")
    
    if failures > 0:
        print("⚠️  Se detectaron fallos de conexión")
    else:
        print("✅ Todas las conexiones fueron exitosas")
    
    print()

def main():
    print("=== TESTING BINANCE API CONNECTIVITY ===")
    print()
    
    test_ipv6_status()
    test_dns_resolution()
    
    # Test API connectivity
    api_ok = test_api_connection()
    
    if not api_ok:
        print("\n⚠️  Conexión a API fallida, realizando pruebas adicionales...")
        test_multiple_attempts(3)
    else:
        print("\nPruebas adicionales de estabilidad:")
        test_multiple_attempts(5)
    
    print()
    print("=== FIN DE PRUEBAS ===")

if __name__ == "__main__":
    main()