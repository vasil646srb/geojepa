#!/usr/bin/env python3
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import gradio as gr
import torch
import numpy as np
from PIL import Image

try:
    from src.inference.predictor import GeoPredictor
except ImportError:
    from src.inference.predictor import Predictor as GeoPredictor


def get_continent(lat, lon):
    if lat > 35 and lon > -10 and lon < 40:
        return "Europe/Asia"
    elif lat > 15 and lon > -20 and lon < 55:
        return "Africa"
    elif lat > 10 and lon > 55 and lon < 145:
        return "Asia"
    elif lat > -35 and lon > 110:
        return "Australia/Oceania"
    elif lat > 15 and lon < -30:
        return "North America"
    elif lat < 15 and lon < -30:
        return "South America"
    elif lat < -60:
        return "Antarctica"
    return "Unknown"


def predict(image):
    if image is None:
        return "Нет изображения", "—", "0%", "—", "Загрузите спутниковый снимок"

    if not hasattr(predict, 'predictor'):
        checkpoint = getattr(predict, 'checkpoint', 'checkpoints/best_model.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predict.predictor = GeoPredictor(checkpoint, device=device)

    pred = predict.predictor.predict(image)
    lat = pred.get('lat', 0.0)
    lon = pred.get('lon', 0.0)
    confidence = pred.get('confidence', 0.0)

    coords = f"{lat:.4f}°, {lon:.4f}°"
    continent = get_continent(lat, lon)
    conf_str = f"{confidence*100:.1f}%"
    map_link = f"https://www.google.com/maps/@{lat:.4f},{lon:.4f},12z"
    description = (
        f"Модель GeoJEPA предсказывает координаты по спутниковому снимку. "
        f"Континент: {continent}. "
        f"Точность зависит от качества обучения."
    )

    return coords, continent, conf_str, map_link, description


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()

    predict.checkpoint = args.checkpoint

    print(f"🚀 Запуск GeoJEPA Demo")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   CUDA: {torch.cuda.is_available()}")

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Загрузите спутниковый снимок (RGB+NIR, 256×256)"),
        outputs=[
            gr.Textbox(label="Координаты"),
            gr.Textbox(label="Континент"),
            gr.Textbox(label="Уверенность"),
            gr.Textbox(label="Ссылка на карту"),
            gr.Textbox(label="Описание местности"),
        ],
        title="🌍 GeoJEPA — Определение координат по спутниковым снимкам",
        description="Загрузите снимок Sentinel-2/HLS (4 канала: RGB+NIR). Модель предскажет координаты.",
    )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
