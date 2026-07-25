import os
import earthaccess

def test_nasa_download():
    # Папка для сохранения скачанных файлов
    output_dir = "./test_hls_download"
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Авторизация в NASA Earthdata...")
    # При первом запуске скрипт запросит логин и пароль в терминале.
    # Токен сохранится автоматически, и дальше ввод не потребуется.
    auth = earthaccess.login(strategy="interactive", persist=True)
    
    if not auth.authenticated:
        print("❌ Ошибка авторизации!")
        return

    print("2. Поиск снимков (регион Ниш, Сербия)...")
    # Bounding Box: (Западная долгота, Южная широта, Восточная долгота, Северная широта)
    bbox_nis = (21.75, 43.20, 22.05, 43.40)
    
    results = earthaccess.search_data(
        short_name="HLSS30",
        bounding_box=bbox_nis,
        temporal=("2026-06-01", "2026-06-10"),
        count=2  # Берем всего 2 гранулы для быстрой проверки
    )
    
    if not results:
        print("❌ Снимки не найдены. Попробуйте изменить даты.")
        return
    
    print(f"✅ Найдено гранул: {len(results)}. Собираем ссылки на файлы...")
    
    # Собираем прямые ссылки на файлы (например, возьмем только красный канал .B04.tif для теста)
    download_links = []
    for result in results:
        links = [link for link in result.data_links(access='direct') if link.endswith('.B04.tif')]
        if links:
            download_links.append(links[0]) # Берем первый попавшийся файл каналов для теста

    if not download_links:
        print("❌ Не удалось сформировать ссылки на файлы.")
        return

    print(3. Скачиваем {len(download_links)} файла(ов) в папку {output_dir}...)
    
    # Скачивание файлов
    downloaded_files = earthaccess.download(download_links, local_path=output_dir)
    
    print("\n Результат загрузки:")
    for file in downloaded_files:
        print(f"   - {file}")

if __name__ == "__main__":
    test_nasa_download()

