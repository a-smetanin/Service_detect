from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import torch
import os
from PIL import Image
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Загрузка модели YOLOv5


@app.route('/upload', methods=['POST'])
def upload_video():
    logger.info('Получен запрос на загрузку видео')
    file = request.files['video']
    if file:
        video_path = os.path.join('uploads', file.filename)
        file.save(video_path)
        logger.info(f'Видео сохранено по пути: {video_path}')

        frames_info = process_video(video_path)
        logger.info('Видео обработано')
        return jsonify(frames_info)
    logger.warning('Видео не загружено')
    return "No file uploaded", 400


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_info = []
    frame_count = 0
    logger.info('Начата обработка видео')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info('Видео достигло конца или произошла ошибка чтения')
            break

        logger.info(f'Обработка кадра {frame_count}')
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img)

        # Получение кадра с наложенными объектами
        rendered_frame = results.render()[0]

        # Сохранение кадра
        frame_filename = f'frame_{frame_count}.jpg'
        frame_path = os.path.join('frames', frame_filename)
        Image.fromarray(rendered_frame).save(frame_path)
        logger.info(f'Кадр {frame_count} сохранен по пути: {frame_path}')

        # Преобразование результатов в читаемый формат
        result_info = []
        for *box, conf, cls in results.xyxy[0]:
            result_info.append({
                'box': [float(x) for x in box],
                'confidence': float(conf),
                'class': int(cls)
            })

        frames_info.append({
            'frame_id': frame_count,
            'frame_path': frame_filename,
            'detections': result_info
        })

        frame_count += 1

    cap.release()
    logger.info('Обработка видео завершена')
    return frames_info


@app.route('/frame/<int:frame_id>', methods=['GET'])
def get_frame(frame_id):
    frame_path = os.path.join('frames', f'frame_{frame_id}.jpg')
    if os.path.exists(frame_path):
        logger.info(f'Отправка кадра {frame_id}')
        return send_file(frame_path, mimetype='image/jpeg')
    else:
        logger.warning(f'Кадр {frame_id} не найден')
        return "Frame not found", 404


if __name__ == '__main__':
    # Создание директорий, если они не существуют
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('frames', exist_ok=True)
    logger.info('Запуск веб-сервиса')
    app.run(debug=True)
