from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from monitoring.telemetry import telemetry
import time

console = Console()

def render_dashboard(state):
    """Construye los objetos Rich pero no borra la consola"""
    table = Table(title="📊 Estado Global", expand=True)
    table.add_column("Activo", justify="center")
    table.add_column("Unidades", justify="right")
    table.add_column("Precio", justify="right")
    table.add_column("Valor USD", justify="right")

    total_valor = 0
    for activo, qty in state["portfolio"].items():
        # Extraer último precio del activo si existe en el mercado
        if activo in state.get("mercado", {}) and "close" in state["mercado"][activo]:
            precio = state["mercado"][activo]["close"].iloc[-1]  # último precio como float
        else:
            precio = 0.0

        valor = qty * precio
        total_valor += valor
        table.add_row(activo, f"{qty:.4f}", f"{precio:.2f}", f"{valor:.2f}")

    metrics = telemetry.snapshot()
    metrics_panel = Panel.fit(
        (
            f"[bold cyan]Ciclos totales:[/bold cyan] {metrics['counters'].get('ciclos_total',0)}\n"
            f"[bold cyan]Valor cartera:[/bold cyan] {total_valor:.2f} USD\n"
            f"[bold cyan]Último tiempo ciclo:[/bold cyan] {metrics['timings'][-1][1]:.4f}s"
        ) if metrics["timings"] else "",
        title="📈 Métricas"
    )

    # Devolvemos ambos elementos como lista
    return table, metrics_panel

def dashboard_loop(state):
    """Actualiza el dashboard sin borrar los logs"""
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            table, panel = render_dashboard(state)
            # Agrupamos ambos en un objeto que Live puede renderizar
            live.update(Panel.fit(table, title="📊 Estado Global"))
            live.update(panel, refresh=True)
            time.sleep(1)
