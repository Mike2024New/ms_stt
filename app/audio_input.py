"""
Модуль захвата аудиопотока с системного микрофона.

Использует библиотеку sounddevice для записи сырых PCM-данных (int16), значений отклонений мембраны микрофона (массив вида [-100, 0...,-100]),
в оперативный буфер в неблокирующем режиме.
"""

import threading
import numpy as np
import sounddevice as sd


class AudioRecorder:
    """Потокобезопасный рекордер PCM-аудио (int16) с микрофона по умолчанию."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Инициализирует параметры аудиозаписи.

        :param sample_rate: Частота дискретизации в Гц (по умолчанию 16000).
        :param channels: Количество каналов записи (по умолчанию 1 — моно).
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._is_recording = False

    def _callback(self, indata: np.ndarray, _frames: int, _time_info, _status) -> None:
        """
        Системный коллбэк аудиодрайвера. Вызывается в отдельном потоке PortAudio.

        :param indata: Входящий блок аудиосэмплов формы (frames, channels).
        :param _frames: Количество сэмплов в блоке.
        :param _time_info: Временные метки потока.
        :param _status: Флаги состояния/переполнения буфера драйвера.
        """
        if self._is_recording:
            with self._lock:
                self._buffer.append(indata.copy())

    def start(self) -> None:
        """
        Очищает внутренний буфер и открывает поток захвата звука с микрофона.
        Если запись уже активна, вызов игнорируется.
        """
        if self._is_recording:
            return
        with self._lock:
            self._buffer.clear()
        self._is_recording = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """
        Останавливает захват звука, закрывает аудиопоток и возвращает записанный фрагмент.

        :return: Одномерный numpy-массив формата int16 со всеми записанными сэмплами.
                 Если данных не поступило, возвращает пустой массив.
        """
        self._is_recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.int16)
            audio = np.concatenate(self._buffer, axis=0).flatten()
            self._buffer.clear()
        return audio
