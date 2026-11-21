"""
Модуль управления конфигурацией микросервиса.

Описывает Pydantic-схему настроек приложения и обеспечивает их загрузку
и сохранение в JSON-файл (config.json) в корневой директории проекта.
"""

import json
from pydantic import BaseModel, Field
import sys
from pathlib import Path


def get_base_dir() -> Path:
    """Возвращает корень проекта (при запуске из .exe — папку рядом с исполняемым файлом)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
CONFIG_PATH = BASE_DIR / "config.json"
MODELS_DIR = BASE_DIR / "models"


class AppConfig(BaseModel):
    """Модель настроек микросервиса распознавания речи."""

    host: str = Field(default="127.0.0.1", description="IP-адрес или хост для запуска HTTP-сервера")
    port: int = Field(default=8000, description="TCP-порт для запуска HTTP-сервера")
    current_model: str = Field(default="", description="Имя активной папки модели внутри директории models/")
    sample_rate: int = Field(default=16000, description="Частота дискретизации захвата аудио (Гц)")
    channels: int = Field(default=1, description="Количество аудиоканалов (1 = моно)")
    device: str = Field(default="cuda", description="Целевое устройство для Whisper ('cuda' или 'cpu')")
    compute_type: str = Field(default="float16",
                              description="Точность вычислений Whisper ('float16', 'int8', 'float32')")
    language: str | None = Field(default=None,
                                 description="Код языка распознавания (например, 'ru') или None для автоопределения")


def load_config() -> AppConfig:
    """
    Загружает конфигурацию из config.json.

    Если директория models/ или файл config.json отсутствуют, создаёт их
    автоматически со значениями по умолчанию.

    :return: Валидированный экземпляр AppConfig.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        cfg = AppConfig()
        save_config(cfg)
        return cfg

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AppConfig(**data)


def save_config(cfg: AppConfig) -> None:
    """
    Сохраняет текущую конфигурацию в файл config.json.

    :param cfg: Экземпляр AppConfig для сериализации.
    """
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg.model_dump(), f, indent=4, ensure_ascii=False)
