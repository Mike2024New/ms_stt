"""
Модуль REST API эндпоинтов микросервиса (FastAPI).

Предоставляет HTTP-маршруты для мониторинга состояния, управления моделями,
записи речи и повторной транскрибации из буфера.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger
from app.engine import STTEngine
from app.logger import get_request_id


class SetModelRequest(BaseModel):
    """Схема тела запроса для смены активной STT-модели."""

    model_name: str


def create_router(engine: STTEngine) -> APIRouter:
    """
    Создаёт и настраивает роутер FastAPI, привязанный к экземпляру STTEngine.

    :param engine: Инициализированный оркестратор STTEngine.
    :return: Настроенный APIRouter.
    """
    router = APIRouter()

    @router.get("/health")
    def health_check():
        """Возвращает базовую телеметрию и статус работоспособности сервиса."""
        return {
            "status": "ok",
            "request_id": get_request_id(),
            "engine_state": engine.state,
            "model_loaded": engine.stt is not None,
            "current_model": engine.config.current_model,
            "has_cached_audio": engine.last_audio_pcm is not None,
        }

    @router.get("/info")
    def get_info():
        """Возвращает подробную документацию по статусам, сценарию работы и эндпоинтам."""
        return {
            "service": "Local STT Microservice",
            "request_id": get_request_id(),
            "description": (
                "Локальный микросервис для записи голоса с микрофона и его преобразования в текст (Speech-to-Text). "
                "Поддерживает плагины Whisper (для высокой точности и англицизмов) и Vosk (для лёгкой работы на CPU)."
            ),
            "current_state": {
                "status": engine.state,
                "active_model": engine.config.current_model or "не выбрана",
                "sample_rate": engine.config.sample_rate,
                "has_cached_audio": engine.last_audio_pcm is not None,
            },
            "states_manual": {
                "stopped": "Сервис запущен, но нейросеть выгружена из памяти (RAM/VRAM свободна). Запись невозможна.",
                "ready": "Модель загружена в память. Сервис готов к приёму команды /speech_start или /repeat.",
                "recording": "Идёт активный захват звука с микрофона в оперативный буфер.",
                "processing": "Запись остановлена, нейросеть распознаёт аудиофрагмент.",
            },
            "quick_start_workflow": [
                "1. Вызовите GET /models, чтобы увидеть доступные папки моделей.",
                "2. Вызовите POST /set-model (если нужно сменить текущую модель).",
                "3. Вызовите POST /start для загрузки модели в память (статус станет 'ready').",
                "4. Вызовите POST /speech_start — начнётся запись с микрофона (статус станет 'recording').",
                "5. Вызовите POST /speech_end — запись остановится, аудио сохранится в буфер, и вернётся текст.",
                "6. (Опционально) Вызовите POST /repeat, чтобы заново распознать последнюю фразу из буфера (в т.ч. после смены модели).",
                "7. По завершении работы вызовите POST /stop, чтобы освободить видеопамять/ОЗУ.",
            ],
            "endpoints": {
                "GET /health": "Быстрая проверка жизни сервиса (healthcheck).",
                "GET /info": "Данная справка с описанием статусов, логики работы и эндпоинтов.",
                "GET /models": "Список моделей из папки models/ с определением движка (vosk / whisper).",
                "POST /set-model": "Выбор модели по имени папки (JSON: {'model_name': '...'}). Делает hot-reload, сохраняя буфер аудио.",
                "POST /start": "Загрузка выбранной модели в память (переход в статус 'ready').",
                "POST /stop": "Остановка записи, очистка буфера и выгрузка модели из памяти (переход в 'stopped').",
                "POST /speech_start": "Очистка старого буфера и старт захвата PCM-аудио с микрофона (переход в 'recording').",
                "POST /speech_end": "Стоп записи, сохранение аудио в буфер, распознавание и возврат текста.",
                "POST /repeat": "Повторное распознавание последнего сохранённого в буфере аудиофрагмента.",
            },
        }

    @router.post("/start")
    def start_engine():
        """Инициализирует и загружает активную модель в память."""
        try:
            res = engine.start()
            logger.info(f"Движок запущен: {res}")
            return res
        except Exception as e:
            logger.error(f"Ошибка при /start: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/stop")
    def stop_engine():
        """Останавливает запись и выгружает модель из памяти."""
        res = engine.stop()
        logger.info("Движок остановлен через API")
        return res

    @router.get("/models")
    def get_models():
        """Возвращает список доступных моделей в папке models/."""
        return {"models": engine.list_models()}

    @router.post("/set-model")
    def set_model(req: SetModelRequest):
        """Устанавливает текущую модель и при необходимости перезагружает движок."""
        try:
            res = engine.set_model(req.model_name)
            logger.info(f"Установлена модель: {req.model_name}")
            return res
        except Exception as e:
            logger.error(f"Ошибка при /set-model ({req.model_name}): {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/speech_start")
    def speech_start():
        """Начинает запись речи с микрофона в оперативный буфер."""
        try:
            res = engine.speech_start()
            logger.info("Начата запись речи (/speech_start)")
            return res
        except Exception as e:
            logger.warning(f"Отклонён /speech_start: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/speech_end")
    def speech_end():
        """Завершает запись речи, запускает транскрибацию и возвращает текст."""
        try:
            logger.info("Остановка записи и запуск распознавания (/speech_end)")
            res = engine.speech_end()
            logger.info(
                f"Распознано (символов: {len(res['text'])}) | "
                f"Тайминги: запись={res['duration_sec']}с, распознавание={res['processing_sec']}с, total={res['total_sec']}с"
            )
            return res
        except Exception as e:
            logger.error(f"Ошибка при /speech_end: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/repeat")
    def repeat_transcription():
        """Повторно распознаёт последний сохранённый в буфере аудиофрагмент."""
        try:
            logger.info("Повторное распознавание из буфера (/repeat)")
            res = engine.repeat()
            logger.info(
                f"Распознано (/repeat, модель={res['model']}, символов: {len(res['text'])}) | "
                f"Тайминги: аудио={res['duration_sec']}с, распознавание={res['processing_sec']}с"
            )
            return res
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Отклонён /repeat: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Ошибка при /repeat: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
