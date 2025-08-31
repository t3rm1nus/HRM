# core/logging.py
import logging
from loguru import logger
import sys

class InterceptHandler(logging.Handler):
    """Handler para interceptar logs de logging estándar y redirigirlos a loguru."""
    def emit(self, record):
        # Obtener el nivel de loguru correspondiente
        log_entry = self.format(record)
        level = logger.level(record.levelname).name
        logger.opt(depth=6, exception=record.exc_info).log(level, log_entry)

def setup_logger(level: int = logging.INFO):
    """
    Configura el logger combinando logging estándar y loguru.
    
    Args:
        level: Nivel de logging (por ejemplo, logging.DEBUG, logging.INFO).
    
    Returns:
        Logger configurado.
    """
    # Configurar loguru
    logger.remove()  # Eliminar handlers predeterminados
    logger.add(sys.stderr, format="{time} | {level} | {name} | {message}", level=level)

    # Configurar logging estándar para redirigir a loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=level, force=True)

    return logger

if __name__ == "__main__":
    # Prueba del logger
    logger = setup_logger(level=logging.DEBUG)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")