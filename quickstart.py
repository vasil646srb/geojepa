#!/usr/bin/env python3
"""
GeoJEPA Quickstart — один скрипт для быстрого старта.
Запуск: python quickstart.py
"""
import os
import sys
import subprocess
import argparse


def run(cmd, desc=None):
    """Выполняет shell-команду с выводом."""
    if desc:
        print(f"\n{'='*60}")
        print(f"➡️  {desc}")
        print(f"{'='*60}")
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Команда завершилась с кодом {result.returncode}")
        return False
    return True


def check_deps():
    """Проверяет и устанавливает зависимости."""
    print("="*60)
    print("🔧 Проверка зависимостей")
    print("="*60)
    
    try:
        import torch
        print(f"   ✅ torch {torch.__version__}")
    except ImportError:
        print("   ❌ torch не найден")
        return False

    try:
        import gradio
        print(f"   ✅ gradio")
    except ImportError:
        print("   ❌ gradio не найден")
        return False

    try:
        import earthaccess
        print(f"   ✅ earthaccess")
    except ImportError:
        print("   ⚠️  earthaccess не найден — нужен для реальных данных")
        if input("   Установить? [Y/n]: ").lower() in ('', 'y', 'yes'):
            run("pip install earthaccess rasterio", "Установка earthaccess + rasterio")
        else:
            print("   Пропущено. Без earthaccess нельзя скачать реальные снимки.")

    try:
        import rasterio
        print(f"   ✅ rasterio")
    except ImportError:
        print("   ⚠️  rasterio не найден")
        if input("   Установить? [Y/n]: ").lower() in ('', 'y', 'yes'):
            run("pip install rasterio", "Установка rasterio")

    print("\n✅ Проверка зависимостей завершена")
    return True


def setup_auth():
    """Настройка авторизации NASA Earthdata."""
    print("\n" + "="*60)
    print("🔐 Авторизация NASA Earthdata")
    print("="*60)
    print("Нужна для скачивания реальных спутниковых снимков HLS.")
    print("Регистрация: https://urs.earthdata.nasa.gov/\n")

    username = os.environ.get('EARTHDATA_USERNAME', '')
    password = os.environ.get('EARTHDATA_PASSWORD', '')

    if username and password:
        print(f"   ✅ Найдены переменные окружения:")
        print(f"      EARTHDATA_USERNAME={username}")
        print("      EARTHDATA_PASSWORD=***")
        return True

    print("   ⚠️  Переменные окружения не найдены.")
    print("   Введите данные (сохранятся в ~/.netrc):")
    
    username = input("   Username (email): ").strip()
    password = input("   Password: ").strip()

    if not username or not password:
        print("   ❌ Данные не введены. Автозагрузка не будет работать.")
        return False

    # Сохраняем в окружение текущей сессии
    os.environ['EARTHDATA_USERNAME'] = username
    os.environ['EARTHDATA_PASSWORD'] = password

    # Сохраняем в ~/.bashrc для будущих сессий
    bashrc = os.path.expanduser("~/.bashrc")
    with open(bashrc, "a") as f:
        f.write(f"\n# GeoJEPA NASA Earthdata\n")
        f.write(f"export EARTHDATA_USERNAME='{username}'\n")
        f.write(f"export EARTHDATA_PASSWORD='{password}'\n")
    print(f"   ✅ Сохранено в ~/.bashrc")

    return True


def download_data():
    """Скачивание тестовых данных."""
    print("\n" + "="*60)
    print("📡 Загрузка реальных спутниковых снимков")
    print("="*60)

    n = input("   Сколько сэмплов скачать? [100]: ").strip()
    n = int(n) if n.isdigit() else 100

    temporal = input("   Период (YYYY-MM-DD YYYY-MM-DD) [2025-05-01 2025-08-31]: ").strip()
    if not temporal:
        temporal = "2025-05-01 2025-08-31"
    dates = temporal.split()

    cmd = (
        f"python scripts/train.py --auto-download "
        f"--num-samples {n} "
        f"--temporal-start {dates[0]} "
        f"--temporal-end {dates[1]} "
        f"--epochs 0"
    )
    # epochs=0 — только скачивание, без обучения
    return run(cmd, f"Скачивание {n} сэмплов")


def train_model():
    """Запуск обучения."""
    print("\n" + "="*60)
    print("🎓 Обучение модели")
    print("="*60)

    n = input("   Количество сэмплов для обучения [1000]: ").strip()
    n = int(n) if n.isdigit() else 1000

    epochs = input("   Эпох [50]: ").strip()
    epochs = int(epochs) if epochs.isdigit() else 50

    batch = input("   Batch size [32]: ").strip()
    batch = int(batch) if batch.isdigit() else 32

    auto = input("   Автозагрузка недостающих снимков? [Y/n]: ").strip().lower()
    auto_flag = "--auto-download" if auto in ('', 'y', 'yes') else ""

    cmd = (
        f"python scripts/train.py "
        f"{auto_flag} "
        f"--num-samples {n} "
        f"--epochs {epochs} "
        f"--batch-size {batch}"
    )
    return run(cmd, "Запуск обучения")


def run_demo():
    """Запуск демо."""
    print("\n" + "="*60)
    print("🚀 Запуск демо")
    print("="*60)

    checkpoint = input("   Путь к чекпоинту [checkpoints/best_model.pt]: ").strip()
    if not checkpoint:
        checkpoint = "checkpoints/best_model.pt"

    if not os.path.exists(checkpoint):
        print(f"   ❌ Чекпоинт не найден: {checkpoint}")
        print("   Сначала обучите модель или укажите другой путь.")
        return False

    share = input("   Публичная ссылка Gradio? [y/N]: ").strip().lower()
    share_flag = "--share" if share in ('y', 'yes') else ""

    cmd = f"python scripts/demo.py --checkpoint {checkpoint} {share_flag}"
    return run(cmd, "Запуск Gradio демо")


def main():
    parser = argparse.ArgumentParser(description="GeoJEPA Quickstart")
    parser.add_argument('--skip-checks', action='store_true', help='Пропустить проверку зависимостей')
    args = parser.parse_args()

    print("="*60)
    print("🌍 GeoJEPA Quickstart")
    print("="*60)
    print("1 — Проверить зависимости")
    print("2 — Настроить NASA авторизацию")
    print("3 — Скачать данные")
    print("4 — Обучить модель")
    print("5 — Запустить демо")
    print("0 — Выход")
    print("="*60)

    if not args.skip_checks:
        if not check_deps():
            print("\n❌ Установите зависимости: pip install -r requirements.txt")
            sys.exit(1)

    while True:
        choice = input("\nВыберите действие [0-5]: ").strip()

        if choice == "1":
            check_deps()
        elif choice == "2":
            setup_auth()
        elif choice == "3":
            download_data()
        elif choice == "4":
            train_model()
        elif choice == "5":
            run_demo()
        elif choice == "0":
            print("👋 До свидания!")
            break
        else:
            print("   Неверный ввод. Введите 0-5.")


if __name__ == "__main__":
    main()

