"""
Плагинный модуль распознавания речи (Speech-to-Text).

Реализует единый абстрактный интерфейс BaseSTT, реестр плагинов STT_REGISTRY
и конкретные адаптеры для моделей Vosk (CPU) и Faster-Whisper (CUDA с
автоматическим fallback на CPU/int8 при ошибках GPU).
"""

import gc
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type
import numpy as np
from loguru import logger


class BaseSTT(ABC):
    """Абстрактный базовый класс для всех плагинов распознавания речи."""

    def __init__(self, model_path: Path, sample_rate: int = 16000, **kwargs):
        """
        :param model_path: Путь к директории с весами и конфигурацией модели.
        :param sample_rate: Частота дискретизации входного PCM-аудио (Гц).
        :param kwargs: Дополнительные специфичные параметры движка.
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.kwargs = kwargs

    @abstractmethod
    def load(self) -> None:
        """Загружает веса модели в оперативную или видеопамять."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Выгружает модель и принудительно освобождает ресурсы памяти."""
        pass

    @abstractmethod
    def transcribe(self, audio_pcm: np.ndarray) -> str:
        """
        Выполняет распознавание речи из массива PCM-данных.

        :param audio_pcm: Одномерный numpy-массив формата int16 (моно).
        :return: Распознанная текстовая строка.
        """
        pass


# Глобальный реестр зарегистрированных STT-плагинов
STT_REGISTRY: dict[str, Type[BaseSTT]] = {}


def register_stt(name: str):
    """
    Декоратор для регистрации нового класса распознавателя в STT_REGISTRY.

    :param name: Уникальное строковое имя движка (например, 'vosk', 'whisper').
    """

    def decorator(cls: Type[BaseSTT]):
        STT_REGISTRY[name] = cls
        return cls

    return decorator


def detect_stt_type(model_name: str) -> str:
    """
    Автоматически определяет тип STT-движка по названию папки модели.

    :param model_name: Имя директории модели.
    :return: 'vosk', если в имени содержится 'vosk', иначе 'whisper'.
    """
    name_lower = model_name.lower()
    if "vosk" in name_lower:
        return "vosk"
    if "gigaam" in name_lower:
        return "gigaam"
    return "whisper"


@register_stt("vosk")
class VoskSTT(BaseSTT):
    """Реализация распознавателя на базе офлайн-библиотеки Vosk (работает на CPU)."""

    def __init__(self, model_path: Path, sample_rate: int = 16000, **kwargs):
        super().__init__(model_path, sample_rate, **kwargs)
        self.model = None

    def load(self) -> None:
        """Инициализирует модель Vosk в оперативной памяти."""
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        logger.info(f"Загрузка Vosk модели: {self.model_path.name}")
        self.model = Model(str(self.model_path))

    def unload(self) -> None:
        """Удаляет ссылку на модель Vosk и запускает сборщик мусора."""
        self.model = None
        gc.collect()
        logger.info("Vosk модель выгружена из памяти")

    def transcribe(self, audio_pcm: np.ndarray) -> str:
        """Распознаёт PCM-аудио через KaldiRecognizer."""
        if self.model is None:
            raise RuntimeError("Модель Vosk не загружена")

        from vosk import KaldiRecognizer
        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.AcceptWaveform(audio_pcm.tobytes())
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip()


@register_stt("whisper")
class WhisperSTT(BaseSTT):
    """
    Реализация распознавателя на базе Faster-Whisper (CTranslate2).
    Поддерживает graceful degradation: при сбое CUDA при загрузке или первом
    инференсе автоматически переключается на CPU (int8).
    """

    def __init__(self, model_path: Path, sample_rate: int = 16000, **kwargs):
        super().__init__(model_path, sample_rate, **kwargs)
        self.model = None
        self.device = kwargs.get("device", "cuda")
        self.compute_type = kwargs.get("compute_type", "float16")
        self.language = kwargs.get("language", None)
        self._is_fallback = False

    def _init_model(self, device: str, compute_type: str):
        """Создаёт экземпляр WhisperModel на указанном устройстве."""
        from faster_whisper import WhisperModel
        logger.info(f"Загрузка Whisper ({self.model_path.name}) на {device} [{compute_type}]...")
        return WhisperModel(
            str(self.model_path),
            device=device,
            compute_type=compute_type,
        )

    def load(self) -> None:
        """Загружает модель Whisper; при ошибке GPU переключается на CPU (int8)."""
        try:
            self.model = self._init_model(self.device, self.compute_type)
            self._is_fallback = (self.device == "cpu")
        except Exception as e:
            if self.device != "cpu":
                logger.warning(f"Ошибка загрузки Whisper на {self.device}: {e}. Fallback на CPU (int8)...")
                self.unload()
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = self._init_model("cpu", "int8")
                self._is_fallback = True
            else:
                raise

    def unload(self) -> None:
        """Выгружает модель Whisper и очищает кэш видеопамяти CUDA (если доступен)."""
        self.model = None
        gc.collect()
        try:
            import torch  # noqa
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Whisper модель выгружена из памяти")

    def _run_transcribe(self, audio_float32: np.ndarray) -> str:
        """Внутренний метод запуска инференса и склейки сегментов текста."""
        segments, _ = self.model.transcribe(
            audio_float32,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe(self, audio_pcm: np.ndarray) -> str:
        """
        Конвертирует int16 PCM во float32 и выполняет распознавание.
        При сбое CUDA во время первого вызова перезагружает модель на CPU (int8).
        """
        if self.model is None:
            raise RuntimeError("Модель Whisper не загружена")

        audio_float32 = audio_pcm.astype(np.float32) / 32768.0
        try:
            return self._run_transcribe(audio_float32)
        except Exception as e:
            if not self._is_fallback and self.device != "cpu":
                logger.warning(f"Сбой CUDA при транскрибации: {e}. Перезагрузка на CPU (int8)...")
                self.unload()
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = self._init_model("cpu", "int8")
                self._is_fallback = True
                return self._run_transcribe(audio_float32)
            raise


@register_stt("gigaam")
class GigaAMSTT(BaseSTT):
    """Плагин для моделей Sber GigaAM (включая Multilingual) в формате ONNX через onnx-asr."""

    def __init__(self, model_path: Path, sample_rate: int = 16000, **kwargs):
        super().__init__(model_path, sample_rate, **kwargs)
        self.model = None

    def load(self) -> None:
        """
        Загружает модель GigaAM (ONNX) из локальной директории через onnx-asr.
        """
        import onnx_asr
        logger.info(f"Загрузка GigaAM (ONNX) из {self.model_path.name}...")

        name_lower = self.model_path.name.lower()
        if "v3" in name_lower and "e2e" in name_lower:
            model_id = "gigaam-v3-e2e-rnnt" if "rnnt" in name_lower else "gigaam-v3-e2e-ctc"
        elif "v3" in name_lower:
            model_id = "gigaam-v3-rnnt" if "rnnt" in name_lower else "gigaam-v3-ctc"
        elif (self.model_path / "config.json").exists():
            model_id = f"istupakov/{self.model_path.name}"
        else:
            model_id = "gigaam-v2-ctc"

        # Автоматически подхватываем int8, если скачан только квантованный файл
        quantization = None
        if list(self.model_path.glob("*.int8.onnx")) and not [
            f for f in self.model_path.glob("*.onnx") if not f.name.endswith(".int8.onnx")
        ]:
            quantization = "int8"

        self.model = onnx_asr.load_model(
            model=model_id,
            path=str(self.model_path),
            quantization=quantization,
        )

    def unload(self) -> None:
        self.model = None
        gc.collect()
        logger.info("GigaAM модель выгружена из памяти")

    def transcribe(self, audio_pcm: np.ndarray) -> str:
        if self.model is None:
            raise RuntimeError("Модель GigaAM не загружена")
        audio_float32 = audio_pcm.astype(np.float32) / 32768.0
        return self.model.recognize(audio_float32, sample_rate=self.sample_rate).strip()


def create_stt_instance(model_path: Path, sample_rate: int = 16000, **kwargs) -> BaseSTT:
    """
    Фабрика для создания нужного STT-плагина по пути к папке модели.

    :param model_path: Путь к директории модели.
    :param sample_rate: Частота дискретизации аудио.
    :return: Инициализированный (но ещё не загруженный в память) объект BaseSTT.
    """
    stt_type = detect_stt_type(model_path.name)
    stt_cls = STT_REGISTRY.get(stt_type)
    if not stt_cls:
        raise ValueError(f"Неизвестный тип STT плагина: {stt_type}")
    return stt_cls(model_path=model_path, sample_rate=sample_rate, **kwargs)
