import os
import random
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import earthaccess
import rasterio


WORLD_REGIONS = [
    (-125.0, 32.0, -114.0, 42.0, "California_USA"),
    (12.0, 36.0, 19.0, 47.0, "Italy"),
    (21.5, 43.0, 30.0, 48.0, "Balkans"),
    (100.0, 20.0, 122.0, 35.0, "China_South"),
    (72.0, 20.0, 78.0, 28.0, "India_North"),
    (-60.0, -35.0, -53.0, -22.0, "Argentina"),
    (18.0, -35.0, 33.0, -22.0, "South_Africa"),
    (139.0, 35.0, 146.0, 42.0, "Japan"),
    (25.0, 60.0, 30.0, 70.0, "Finland"),
    (-10.0, 51.0, 2.0, 59.0, "UK"),
    (37.0, 55.0, 40.0, 56.0, "Moscow_Russia"),
    (-80.0, 25.0, -81.0, 30.0, "Florida_USA"),
    (130.0, -25.0, 150.0, -11.0, "Australia"),
    (-3.0, 4.0, 15.0, 18.0, "West_Africa"),
    (55.0, 24.0, 56.0, 27.0, "UAE"),
]


class HLSDownloader:
    def __init__(self, output_dir="data/hls_real"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = self.output_dir / ".tmp"
        self.tmp_dir.mkdir(exist_ok=True)
        self._auth_done = False

    def _auth(self):
        if self._auth_done:
            return
        try:
            auth = earthaccess.login(strategy="environment", persist=True)
        except Exception:
            print("   ⚠️ Environment auth не сработал, пробуем интерактивно...")
            auth = earthaccess.login(strategy="interactive", persist=True)
        if not auth or not getattr(auth, 'authenticated', False):
            raise RuntimeError(
                "❌ Авторизация NASA Earthdata не удалась.\n"
                "   export EARTHDATA_USERNAME=vasil646srb@gmail.com\n"
                "   export EARTHDATA_PASSWORD=your_password"
            )
        self._auth_done = True
        print("   ✅ Авторизация NASA прошла")

    def _random_bbox(self, min_lon, min_lat, max_lon, max_lat, size=0.25):
        lon = random.uniform(min_lon, max_lon - size)
        lat = random.uniform(min_lat, max_lat - size)
        return (lon, lat, lon + size, lat + size)

    def _download_group(self, bbox, temporal, count=5):
        try:
            results = earthaccess.search_data(
                short_name="HLSS30",
                bounding_box=bbox,
                temporal=temporal,
                count=count
            )
            if not results:
                return []
            files = earthaccess.download(results, local_path=str(self.tmp_dir))
            return files if files else []
        except Exception as e:
            print(f"   ⚠️ Ошибка загрузки: {e}")
            return []

    def _group_by_granule(self, files):
        granules = {}
        for f in files:
            fname = os.path.basename(f)
            parts = fname.split('.')
            if len(parts) < 6 or not fname.endswith('.tif'):
                continue
            granule_id = '.'.join(parts[:5])
            granules.setdefault(granule_id, []).append(f)
        return granules

    def _process_granule(self, granule_id, files, sample_idx):
        band_map = {}
        for f in files:
            fname = os.path.basename(f)
            for band in ['B02', 'B03', 'B04', 'B08']:
                if f'.{band}.' in fname:
                    band_map[band] = f

        if not all(b in band_map for b in ['B02', 'B03', 'B04', 'B08']):
            return None

        bands = []
        first_bounds = None
        for b in ['B04', 'B03', 'B02', 'B08']:
            with rasterio.open(band_map[b]) as src:
                img = src.read(1).astype(np.float32)
                if first_bounds is None:
                    first_bounds = src.bounds

                valid = img[img > 0]
                if len(valid) > 0:
                    p2, p98 = np.percentile(valid, (2, 98))
                    if p98 > p2:
                        img = np.clip((img - p2) / (p98 - p2), 0, 1)
                    else:
                        img = np.clip(img / 10000.0, 0, 1)
                else:
                    img = np.zeros_like(img)

                if img.shape != (256, 256):
                    pil = Image.fromarray((img * 255).astype(np.uint8))
                    pil = pil.resize((256, 256), Image.Resampling.LANCZOS)
                    img = np.array(pil, dtype=np.float32) / 255.0

                bands.append(img)

        rgbnir = np.stack(bands, axis=0)
        lat = (first_bounds.bottom + first_bounds.top) / 2.0
        lon = (first_bounds.left + first_bounds.right) / 2.0

        out_path = self.output_dir / f"sample_{sample_idx:06d}.npz"
        np.savez_compressed(out_path, image=rgbnir, lat=lat, lon=lon, granule=granule_id)
        return out_path

    def _cleanup_tmp(self):
        for item in self.tmp_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception:
                pass

    def download(self, n_samples, temporal=("2025-05-01", "2025-08-31")):
        self._auth()

        existing = sorted(self.output_dir.glob("sample_*.npz"))
        n_existing = len(existing)
        print(f"   📦 Уже есть: {n_existing} сэмплов")

        needed = n_samples - n_existing
        if needed <= 0:
            print(f"   ✅ Достаточно данных ({n_existing} >= {n_samples})")
            return n_existing

        print(f"   ⬇️ Нужно скачать: {needed}")
        downloaded = 0
        sample_idx = n_existing
        regions = random.sample(WORLD_REGIONS, len(WORLD_REGIONS))

        for region_idx, region in enumerate(regions):
            if downloaded >= needed:
                break

            print(f"\n   🌍 [{region_idx+1}/{len(regions)}] {region[4]}")
            for attempt in range(3):
                if downloaded >= needed:
                    break

                bbox = self._random_bbox(*region[:4])
                print(f"      Попытка {attempt+1}: bbox {bbox}")

                files = self._download_group(bbox, temporal, count=3)
                if not files:
                    continue

                granules = self._group_by_granule(files)
                print(f"      📥 {len(files)} файлов -> {len(granules)} гранул")

                for gid, gfiles in granules.items():
                    if downloaded >= needed:
                        break
                    out = self._process_granule(gid, gfiles, sample_idx)
                    if out:
                        print(f"         ✓ sample_{sample_idx:06d}.npz")
                        downloaded += 1
                        sample_idx += 1

            self._cleanup_tmp()

        total = n_existing + downloaded
        print(f"\n   ✅ Готово. Всего: {total} (новых: {downloaded})")
        return total
