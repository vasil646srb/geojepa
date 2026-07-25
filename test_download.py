import os
import earthaccess

def test_nasa_download():
    output_dir = "./test_hls_download"
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Авторизация в NASA Earthdata...")
    # Авторизация (токен уже сохранен в системе, повторный ввод не потребуется)
    auth = earthaccess.login(strategy="interactive", persist=True)
    
    if not auth.authenticated:
        print("❌ Ошибка авторизации!")
        return

    print("2. Поиск снимков (регион Ниш, Сербия)...")
    bbox_nis = (21.75, 43.20, 22.05, 43.40)
    
    # Ищем всего 1 гранулу для быстрого теста, так как полная гранулы HLS весит много
    results = earthaccess.search_data(
        short_name="HLSS30",
        bounding_box=bbox_nis,
        temporal=("2026-06-01", "2026-06-05"),
        count=1  
    )
    
    if not results:
        print("❌ Снимки не найдены. Попробуйте изменить даты.")
        return
    
    print(f"✅ Найдено гранул: {len(results)}. Запускаем загрузку...")
    
    # Передаем список гранул напрямую в download(). 
    # Библиотека сама подставит правильный HTTPS-протокол для внешнего скачивания.
    downloaded_files = earthaccess.download(results, local_path=output_dir)
    
    print("\n📦 Успешно скачанные файлы:")
    for file in downloaded_files:
        print(f"   - {file}")

if __name__ == "__main__":
    test_nasa_download()

