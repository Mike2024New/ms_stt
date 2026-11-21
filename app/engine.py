"""
Модуль оркестрации (движок) STT-микросервиса.

Управляет жизненным циклом моделей, конечным автоматом состояний
(STOPPED -> READY -> RECORDING -> PROCESSING), захватом аудио,
замером таймингов и кэшированием последней фразы для повторного распознавания.
"""

import threading
import time
from enum import Enum
import numpy as np
from app.config import AppConfig, MODELS_DIR, save_config
from app.audio_input import AudioRecorder
from app.stt_module import BaseSTT, create_stt_instance, detect_stt_type


class EngineState(str, Enum):
    """Возможные состояния работы STT-движка."""

    STOPPED = "stopped"
    READY = "ready"
    RECORDING = "recording"
    PROCESSING = "processing"


class STTEngine:
    """Главный оркестратор, связывающий конфигурацию, аудиорекордер и плагины STT."""

    def __init__(self, config: AppConfig):
        """
        :param config: Загруженный экземпляр конфигурации AppConfig.
        """
        self.config = config
        self.state = EngineState.STOPPED
        self.stt: BaseSTT | None = None
        self.recorder = AudioRecorder(
            sample_rate=config.sample_rate,
            channels=config.channels,
        )
        self._lock = threading.Lock()
        self._record_start_time: float = 0.0
        self.last_audio_pcm: np.ndarray | None = None

    def list_models(self) -> list[dict]:
        """
        Сканирует директорию models/ и возвращает список доступных моделей.

        :return: Список словарей с именем папки, типом плагина и флагом активности.
        """
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        models = []
        for item in sorted(MODELS_DIR.iterdir()):
            if item.is_dir():
                models.append({
                    "name": item.name,
                    "engine": detect_stt_type(item.name),
                    "active": item.name == self.config.current_model,
                })
        return models

    def start(self) -> dict:
        """
        Загружает выбранную в конфигурации модель в память и переводит движок в READY.

        :return: Словарь со статусом, именем модели и типом движка.
        """
        with self._lock:
            if self.state != EngineState.STOPPED:
                return {"status": self.state, "message": "Движок уже запущен"}

            if not self.config.current_model:
                raise ValueError("В config.json не выбрана модель (current_model пуст)")

            model_path = MODELS_DIR / self.config.current_model
            if not model_path.exists() or not model_path.is_dir():
                raise FileNotFoundError(f"Папка модели не найдена: {model_path}")

            self.stt = create_stt_instance(
                model_path=model_path,
                sample_rate=self.config.sample_rate,
                device=self.config.device,
                compute_type=self.config.compute_type,
                language=self.config.language,
            )
            self.stt.load()
            self.state = EngineState.READY
            return {
                "status": self.state,
                "model": self.config.current_model,
                "engine": detect_stt_type(self.config.current_model),
            }

    def stop(self) -> dict:
        """
        Останавливает активную запись, очищает буферы, выгружает модель и переводит в STOPPED.

        :return: Словарь со статусом остановки.
        """
        with self._lock:
            if self.state == EngineState.RECORDING:
                self.recorder.stop()

            if self.stt is not None:
                self.stt.unload()
                self.stt = None

            self.last_audio_pcm = None
            self.state = EngineState.STOPPED
            return {"status": self.state, "message": "Модель выгружена, буфер очищен, движок остановлен"}

    def set_model(self, model_name: str) -> dict:
        """
        Устанавливает новую активную модель и сохраняет выбор в config.json.
        Если движок был запущен, выполняет горячую перезагрузку (reload),
        сохраняя последний аудиофрагмент в буфере для функции /repeat.

        :param model_name: Имя папки модели внутри models/.
        :return: Словарь с результатом установки модели.
        """
        model_path = MODELS_DIR / model_name
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(f"Модель '{model_name}' не найдена в папке models/")

        was_running = self.state != EngineState.STOPPED
        saved_audio = self.last_audio_pcm

        if was_running:
            self.stop()

        self.config.current_model = model_name
        save_config(self.config)

        if was_running:
            res = self.start()
            self.last_audio_pcm = saved_audio
            return res

        self.last_audio_pcm = saved_audio
        return {
            "status": self.state,
            "model": model_name,
            "message": "Модель сохранена в конфиг (будет загружена при /start)",
        }

    def speech_start(self) -> dict:
        """
        Очищает предыдущий кэш аудио, запускает запись с микрофона и переводит в RECORDING.

        :return: Словарь со статусом начала записи.
        """
        with self._lock:
            if self.state == EngineState.STOPPED:
                raise RuntimeError("Движок остановлен. Сначала вызовите /start")
            if self.state == EngineState.RECORDING:
                return {"status": self.state, "message": "Запись уже идёт"}
            if self.state == EngineState.PROCESSING:
                raise RuntimeError("Идёт распознавание предыдущей фразы")

            self.last_audio_pcm = None
            self._record_start_time = time.perf_counter()
            self.recorder.start()
            self.state = EngineState.RECORDING
            return {"status": self.state, "message": "Запись аудио началась"}

    def speech_end(self) -> dict:
        """
        Останавливает запись, сохраняет PCM-буфер для /repeat, запускает распознавание
        и возвращает распознанный текст с детальными таймингами.

        :return: Словарь с полями text, duration_sec, processing_sec и total_sec.
        """
        with self._lock:
            if self.state != EngineState.RECORDING:
                raise RuntimeError(f"Невозможно остановить запись из состояния '{self.state}'")

            self.state = EngineState.PROCESSING
            audio_pcm = self.recorder.stop()

        self.last_audio_pcm = audio_pcm if len(audio_pcm) > 0 else None

        duration_sec = round(len(audio_pcm) / self.config.sample_rate, 2)
        if len(audio_pcm) == 0:
            self.state = EngineState.READY
            total_sec = round(time.perf_counter() - self._record_start_time, 3) if self._record_start_time else 0.0
            return {
                "text": "",
                "duration_sec": 0.0,
                "processing_sec": 0.0,
                "total_sec": total_sec,
            }

        t_stt = time.perf_counter()
        try:
            text = self.stt.transcribe(audio_pcm)
        finally:
            self.state = EngineState.READY

        now = time.perf_counter()
        processing_sec = round(now - t_stt, 3)
        total_sec = (
            round(now - self._record_start_time, 3)
            if self._record_start_time
            else round(duration_sec + processing_sec, 3)
        )

        return {
            "text": text,
            "duration_sec": duration_sec,
            "processing_sec": processing_sec,
            "total_sec": total_sec,
        }

    def repeat(self) -> dict:
        """
        Повторно распознаёт последний записанный аудиофрагмент из буфера
        на текущей активной модели без повторной записи с микрофона.

        :return: Словарь с полями text, duration_sec, processing_sec, total_sec и model.
        """
        with self._lock:
            if self.state == EngineState.STOPPED:
                raise RuntimeError("Движок остановлен. Сначала вызовите /start")
            if self.state == EngineState.RECORDING:
                raise RuntimeError("Сейчас идёт запись. Сначала завершите её через /speech_end")
            if self.state == EngineState.PROCESSING:
                raise RuntimeError("Идёт распознавание, дождитесь завершения")
            if self.last_audio_pcm is None or len(self.last_audio_pcm) == 0:
                raise ValueError(
                    "В буфере нет сохранённого аудио. Сначала запишите фразу (/speech_start -> /speech_end)")

            self.state = EngineState.PROCESSING
            audio_pcm = self.last_audio_pcm

        duration_sec = round(len(audio_pcm) / self.config.sample_rate, 2)
        t_stt = time.perf_counter()
        try:
            text = self.stt.transcribe(audio_pcm)
        finally:
            self.state = EngineState.READY

        processing_sec = round(time.perf_counter() - t_stt, 3)
        return {
            "text": text,
            "duration_sec": duration_sec,
            "processing_sec": processing_sec,
            "total_sec": processing_sec,
            "model": self.config.current_model,
        }


if __name__ == '__main__':
    # быстрый набросок, чтобы понять
    from app.config import AppConfig

    app_config = AppConfig()
    app_config.current_model = 'gigaam-v3-e2e-ctc'
    stt_eng = STTEngine(config=app_config)
    print(stt_eng.list_models())
    print(app_config)
    stt_eng.start()
    input('. начать говорить.')
    stt_eng.speech_start()
    print('. закончить говорить.')
    input('...')
    print(stt_eng.speech_end())
