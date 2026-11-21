"""
Точка входа в локальный STT-микросервис.

Обрабатывает аргументы командной строки (--host, --port, --request-id),
инициализирует сквозное логирование, загружает конфигурацию и запускает
веб-сервер Uvicorn с приложением FastAPI.
"""

import argparse
import os
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request
from loguru import logger

from app.logger import setup_logger, get_request_id
from app.config import load_config
from app.engine import STTEngine
from app.api import create_router


def parse_args() -> argparse.Namespace:
    """
    Парсит параметры запуска из командной строки или переменных окружения.

    :return: Пространство имён с аргументами host, port и request_id.
    """
    parser = argparse.ArgumentParser(description="Local STT Microservice")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("STT_HOST", ""),
        help="Хост для запуска сервера (переопределяет config.json)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("STT_PORT", 0)),
        help="Порт для запуска сервера (переопределяет config.json)",
    )
    parser.add_argument(
        "--request-id",
        type=str,
        default=os.getenv("STT_REQUEST_ID", ""),
        help="Сквозной request_id на всё время жизни процесса",
    )
    ps_args, _ = parser.parse_known_args()
    return ps_args


args = parse_args()
setup_logger(args.request_id)

config = load_config()
if args.host:
    config.host = args.host
if args.port:
    config.port = args.port

engine = STTEngine(config)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Управляет событиями запуска и корректного завершения работы FastAPI."""
    logger.info(f"Микросервис STT инициализирован (request_id={get_request_id()})")
    yield
    logger.info("Завершение работы микросервиса, освобождение ресурсов...")
    engine.stop()


app = FastAPI(title="Local STT Microservice", lifespan=lifespan)


@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next):
    """Логирует HTTP-вызовы и добавляет заголовок X-Request-ID во все ответы."""
    logger.debug(f"--> {request.method} {request.url.path}")
    response = await call_next(request)
    response.headers["X-Request-ID"] = get_request_id()
    logger.debug(f"<-- {request.method} {request.url.path} [{response.status_code}]")
    return response


app.include_router(create_router(engine))

# подключение фронтенда
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.config import BASE_DIR

UI_DIR = BASE_DIR / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


    @app.get("/", include_in_schema=False)
    def serve_ui():
        return FileResponse(str(UI_DIR / "index.html"))

if __name__ == "__main__":
    logger.info(f"Запуск HTTP-сервера на {config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port, log_config=None)
