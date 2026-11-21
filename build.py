"""
Скрипт сборки микросервиса в многофайловый дистрибутив Windows (.exe)
с помощью PyInstaller в директорию releases/stt_service/.
"""

import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RELEASES_DIR = BASE_DIR / "releases"
BUILD_TMP_DIR = BASE_DIR / "build_tmp"
APP_NAME = "stt_service"


def run_build() -> None:
    """Запускает сборку PyInstaller через subprocess и подготавливает папку релиза."""
    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = RELEASES_DIR / APP_NAME

    # Удаляем предыдущую сборку, если она была
    if target_dir.exists():
        print(f"Удаление старой сборки: {target_dir}")
        shutil.rmtree(target_dir)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",  # Многофайловая сборка (быстрый старт, без распаковки в Temp)
        "--console",  # Оставляем консоль для вывода логов и корректного перехвата Ctrl+C
        "--name", APP_NAME,
        "--distpath", str(RELEASES_DIR),
        "--workpath", str(BUILD_TMP_DIR),
        "--specpath", str(BUILD_TMP_DIR),

        # Скрытые импорты uvicorn / fastapi / pydantic
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",

        # Сбор нативных библиотек, моделей VAD, словарей и DLL
        "--collect-all", "vosk",
        "--collect-all", "faster_whisper",
        "--collect-all", "ctranslate2",
        "--collect-all", "onnx_asr",
        "--collect-all", "onnxruntime",
        "--collect-all", "sounddevice",

        str(BASE_DIR / "main.py"),
    ]

    print("Запуск PyInstaller...")
    subprocess.run(cmd, check=True)

    # Очищаем временные файлы сборки
    if BUILD_TMP_DIR.exists():
        shutil.rmtree(BUILD_TMP_DIR)

    # Создаём рядом с .exe пустую папку models/ для удобства
    (target_dir / "models").mkdir(exist_ok=True)

    # Если в корне проекта есть config.json, копируем его как шаблон в релиз
    src_config = BASE_DIR / "config.json"
    if src_config.exists():
        shutil.copy2(src_config, target_dir / "config.json")

    print("\n" + "=" * 50)
    print(f"Сборка успешно завершена!")
    print(f"Готовый релиз находится здесь: {target_dir}")
    print(f"Запуск: {target_dir / (APP_NAME + '.exe')} --port 8002 --request-id 1234")
    print("=" * 50)


if __name__ == "__main__":
    run_build()
