from setuptools import setup, find_packages

setup(
    name="geojepa",
    version="0.1.0",
    description="GeoJEPA: Satellite Image Geolocation with JEPA",
    author="GeoJEPA Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "tqdm>=4.65.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "data": ["sentinelhub>=3.9.0", "openeo>=0.28.0", "rasterio>=1.3.0"],
        "demo": ["gradio>=3.35.0"],
        "osm": ["osmnx>=1.6.0", "geopandas>=0.13.0"],
        "dev": ["pytest>=7.0", "black>=23.0", "flake8>=6.0"],
        "all": [
            "sentinelhub>=3.9.0", "openeo>=0.28.0", "rasterio>=1.3.0",
            "gradio>=3.35.0", "osmnx>=1.6.0", "geopandas>=0.13.0"
        ],
    },
    entry_points={
        "console_scripts": [
            "geojepa-train=scripts.train:main",
            "geojepa-predict=scripts.predict:main",
            "geojepa-demo=scripts.demo:main",
        ],
    },
)
