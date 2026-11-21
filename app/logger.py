"""
Модуль централизованного логирования и сквозной трассировки.

Настраивает библиотеку loguru для вывода логов в stdout и файл с ротацией.
Обеспечивает привязку единого идентификатора процесса (request_id) ко всем
записям лога, включая фоновые потоки аудиозахвата и инференса.
"""

import sys
import uuid
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"

# Глобальный сквозной идентификатор жизненного цикла процесса
_APP_REQUEST_ID: str = ""


def get_request_id() -> str:
    """
    Возвращает текущий сквозной идентификатор процесса.

    :return: Строковый идентификатор request_id.
    """
    return _APP_REQUEST_ID


def setup_logger(request_id: str | None = None) -> str:
    """
    Инициализирует систему логирования loguru с привязкой сквозного request_id.

    Создаёт директорию logs/, настраивает вывод в консоль (уровень INFO)
    и запись в файл с ротацией по 10 МБ и архивацией (уровень DEBUG).

    :param request_id: Внешний идентификатор трассировки (например, от родительского процесса).
                       Если не передан или пуст, генерируется автоматически (12 hex-символов).
    :return: Итоговый установленный request_id.
    """
    global _APP_REQUEST_ID
    _APP_REQUEST_ID = request_id.strip() if request_id and request_id.strip() else uuid.uuid4().hex[:12]

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()

    # Глобальная привязка extra-поля ко всем логам в любых потоках приложения
    logger.configure(extra={"request_id": _APP_REQUEST_ID})

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>req:{extra[request_id]}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(sys.stdout, format=log_format, level="INFO")
    logger.add(
        LOGS_DIR / "stt_service.log",
        format=log_format,
        level="DEBUG",
        rotation="10 MB",
        retention=5,
        compression="zip",
        encoding="utf-8",
    )
    return _APP_REQUEST_ID
