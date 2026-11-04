[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('start', 'stop', 'restart')]
    [string]$Accion
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Obtener-RutaProyecto {
    return (Resolve-Path -LiteralPath $PSScriptRoot).Path
}

function Obtener-CarpetaFrontend {
    param([string]$Raiz)

    $nombres = @('frontend', 'front end', 'front-end')
    foreach ($nombre in $nombres) {
        $ruta = Join-Path $Raiz $nombre
        $paquete = Join-Path $ruta 'package.json'
        if ((Test-Path -LiteralPath $ruta -PathType Container) -and
            (Test-Path -LiteralPath $paquete -PathType Leaf)) {
            return $ruta
        }
    }

    throw "No encuentro una carpeta frontend, front end o front-end con package.json en: $Raiz"
}

function Obtener-EjecutablePython {
    param([string]$Raiz)

    $candidatos = @(
        (Join-Path $Raiz '.venv\Scripts\python.exe'),
        (Join-Path $Raiz 'venv\Scripts\python.exe'),
        (Join-Path $Raiz 'env\Scripts\python.exe')
    )
    foreach ($candidato in $candidatos) {
        if (Test-Path -LiteralPath $candidato -PathType Leaf) {
            return $candidato
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) { return $python.Source }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { return $py.Source }

    throw 'No encuentro Python. Instálalo o crea .venv en la raíz del proyecto.'
}

function Obtener-ComandoPython {
    param([string]$Ejecutable)

    if ([IO.Path]::GetFileName($Ejecutable) -ieq 'py.exe') {
        return 'py -3.10 main.py'
    }
    return ('"{0}" main.py' -f $Ejecutable)
}

function Obtener-ComandoFrontend {
    param([string]$Carpeta)

    $package = Get-Content -LiteralPath (Join-Path $Carpeta 'package.json') -Raw | ConvertFrom-Json
    if ($package.scripts -and $package.scripts.dev) { return 'npm run dev' }
    if ($package.scripts -and $package.scripts.start) { return 'npm run start' }
    throw 'package.json no define ningún script "dev" ni "start".'
}

function Obtener-DirectorioEstado {
    param([string]$Raiz)
    $directorio = Join-Path $Raiz '.hrm_runtime'
    if (-not (Test-Path -LiteralPath $directorio)) {
        New-Item -ItemType Directory -Path $directorio | Out-Null
    }
    return $directorio
}

function Iniciar-VentanaServicio {
    param(
        [string]$Nombre,
        [string]$Directorio,
        [string]$Comando,
        [string]$ArchivoPid
    )

    if (Test-Path -LiteralPath $ArchivoPid) {
        $pidAnterior = 0
        $contenido = Get-Content -LiteralPath $ArchivoPid -ErrorAction SilentlyContinue | Select-Object -First 1
        if ([int]::TryParse([string]$contenido, [ref]$pidAnterior)) {
            $proceso = Get-Process -Id $pidAnterior -ErrorAction SilentlyContinue
            if ($proceso) {
                Write-Host "$Nombre ya parece estar iniciado (PID $pidAnterior)." -ForegroundColor Yellow
                return
            }
        }
        Remove-Item -LiteralPath $ArchivoPid -Force -ErrorAction SilentlyContinue
    }

    $linea = 'title HRM - {0} & cd /d "{1}" && {2}' -f $Nombre, $Directorio, $Comando
    $proceso = Start-Process -FilePath $env:ComSpec -ArgumentList @('/k', $linea) -WorkingDirectory $Directorio -PassThru
    Set-Content -LiteralPath $ArchivoPid -Value $proceso.Id -Encoding ASCII
    Write-Host "$Nombre iniciado en una ventana independiente (PID $($proceso.Id))." -ForegroundColor Green
}

function Iniciar-HRM {
    param([string]$Raiz)

    $archivoMain = Join-Path $Raiz 'main.py'
    if (-not (Test-Path -LiteralPath $archivoMain -PathType Leaf)) {
        throw "No encuentro main.py en la raíz: $Raiz"
    }

    $carpetaFrontend = Obtener-CarpetaFrontend -Raiz $Raiz
    $python = Obtener-EjecutablePython -Raiz $Raiz
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npm) { throw 'No encuentro npm. Instala Node.js y vuelve a intentarlo.' }

    $comandoBackend = Obtener-ComandoPython -Ejecutable $python
    $comandoFrontend = Obtener-ComandoFrontend -Carpeta $carpetaFrontend
    $estado = Obtener-DirectorioEstado -Raiz $Raiz

    Iniciar-VentanaServicio -Nombre 'BACKEND' -Directorio $Raiz -Comando $comandoBackend -ArchivoPid (Join-Path $estado 'backend.pid')
    Iniciar-VentanaServicio -Nombre 'FRONTEND' -Directorio $carpetaFrontend -Comando $comandoFrontend -ArchivoPid (Join-Path $estado 'frontend.pid')
}

function Detener-VentanaServicio {
    param([string]$Nombre, [string]$ArchivoPid)

    if (-not (Test-Path -LiteralPath $ArchivoPid)) {
        Write-Host "${Nombre}: no hay PID registrado; no se ha terminado ningún proceso." -ForegroundColor DarkYellow
        return
    }

    $contenido = Get-Content -LiteralPath $ArchivoPid -ErrorAction SilentlyContinue | Select-Object -First 1
    $idProceso = 0
    if (-not [int]::TryParse([string]$contenido, [ref]$idProceso)) {
        Remove-Item -LiteralPath $ArchivoPid -Force -ErrorAction SilentlyContinue
        Write-Host "${Nombre}: archivo PID no válido; se ha limpiado el registro." -ForegroundColor Yellow
        return
    }

    $proceso = Get-Process -Id $idProceso -ErrorAction SilentlyContinue
    if ($proceso) {
        & taskkill.exe /PID $idProceso /T /F | Out-Null
        Write-Host "$Nombre detenido (PID $idProceso)." -ForegroundColor Green
    } else {
        Write-Host "$Nombre ya estaba detenido." -ForegroundColor DarkYellow
    }
    Remove-Item -LiteralPath $ArchivoPid -Force -ErrorAction SilentlyContinue
}

function Detener-HRM {
    param([string]$Raiz)
    $estado = Join-Path $Raiz '.hrm_runtime'
    if (-not (Test-Path -LiteralPath $estado)) {
        Write-Host 'HRM no tiene registros de procesos iniciados por estos scripts.' -ForegroundColor Yellow
        return
    }
    Detener-VentanaServicio -Nombre 'FRONTEND' -ArchivoPid (Join-Path $estado 'frontend.pid')
    Detener-VentanaServicio -Nombre 'BACKEND' -ArchivoPid (Join-Path $estado 'backend.pid')
}

$raizProyecto = Obtener-RutaProyecto
switch ($Accion) {
    'start'   { Iniciar-HRM -Raiz $raizProyecto }
    'stop'    { Detener-HRM -Raiz $raizProyecto }
    'restart' {
        Detener-HRM -Raiz $raizProyecto
        Start-Sleep -Seconds 2
        Iniciar-HRM -Raiz $raizProyecto
    }
}
