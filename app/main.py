from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np
import torch
import os
from PIL import Image
import logging
from ultralytics import YOLO
import json
from flask_socketio import SocketIO, emit

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# Загрузка моделей YOLO
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)  # YOLOv5
model_yolov8m_custom = YOLO("640m.pt")  # Кастомная YOLOv8


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    logger.info('Получен запрос на загрузку видео')
    if 'video' not in request.files:
        logger.error('Ключ "video" отсутствует в запросе')
        return "No video file part", 400

    file = request.files['video']
    model_type = request.form.get('model_type', 'all')  # Получаем параметр model_type из запроса

    if file.filename == '':
        logger.error('Имя файла не указано')
        return "No selected file", 400

    if file:
        video_path = os.path.join('uploads', file.filename)
        file.save(video_path)
        logger.info(f'Видео сохранено по пути: {video_path}')

        socketio.start_background_task(target=process_video, video_path=video_path, model_type=model_type)
        return jsonify({'status': 'Видео загружено, обработка началась.'}), 202

    logger.warning('Видео не загружено')
    return "No file uploaded", 400


def draw_label(image, text, pos, bg_color, text_color=(255, 255, 255)):
    font_scale = 0.7
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    x, y = pos
    cv2.rectangle(image, (x, y - text_h - 5), (x + text_w, y), bg_color, -1)
    cv2.putText(image, text, (x, y - 5), font, font_scale, text_color, font_thickness)


def process_video(video_path, model_type):
    cap = cv2.VideoCapture(video_path)
    frames_info = []
    frame_count = 0
    logger.info('Начата обработка видео')

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # Установим значение по умолчанию, если не удается получить FPS

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info('Видео достигло конца или произошла ошибка чтения')
            break

        logger.info(f'Обработка кадра {frame_count}')
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обработка кадра выбранными моделями
        result_info = {'time': frame_count / fps}
        rendered_frame = frame.copy()

        if model_type in ['all', 'yolov5']:
            results_yolov5 = model_yolov5(img)
            result_info['yolov5'] = []
            for *box, conf, cls in results_yolov5.xyxy[0]:
                x1, y1, x2, y2 = map(int, box)
                label = f"{model_yolov5.names[int(cls)]}: {conf:.2f}"
                cv2.rectangle(rendered_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                draw_label(rendered_frame, label, (x1, y1), (0, 0, 255))
                result_info['yolov5'].append({
                    'box': [float(x) for x in box],
                    'confidence': float(conf),
                    'class': int(cls)
                })

        if model_type in ['all', 'yolov8']:
            results_yolov8m_custom = model_yolov8m_custom(img)
            result_info['yolov8m_custom'] = []
            for i, box in enumerate(results_yolov8m_custom[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                label = f"{results_yolov8m_custom[0].names[int(results_yolov8m_custom[0].boxes.cls[i])]}: {results_yolov8m_custom[0].boxes.conf[i]:.2f}"
                cv2.rectangle(rendered_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                draw_label(rendered_frame, label, (x1, y1), (255, 0, 0))
                result_info['yolov8m_custom'].append({
                    'box': [x1, y1, x2, y2],
                    'confidence': float(results_yolov8m_custom[0].boxes.conf[i]),
                    'class': int(results_yolov8m_custom[0].boxes.cls[i])
                })

        # Сохранение кадра
        frame_filename = f'frame_{frame_count}.jpg'
        frame_path = os.path.join('frames', frame_filename)
        Image.fromarray(cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)).save(frame_path)
        logger.info(f'Кадр {frame_count} сохранен по пути: {frame_path}')

        frames_info.append(result_info)
        frame_count += 1

    cap.release()
    logger.info('Обработка видео завершена')

    json_filename = os.path.join('uploads', os.path.splitext(os.path.basename(video_path))[0] + '.json')
    with open(json_filename, 'w') as json_file:
        json.dump(frames_info, json_file)

    socketio.emit('processing_done', {'message': 'Обработка видео завершена', 'json_path': json_filename})
    logger.info('JSON файл сохранен')


@app.route('/status')
def status():
    return render_template('status.html')


@app.route('/json/<filename>', methods=['GET'])
def get_json(filename):
    json_path = os.path.join('uploads', filename)
    if os.path.exists(json_path):
        return send_file(json_path, mimetype='application/json')
    else:
        return "JSON файл не найден", 404


if __name__ == '__main__':
    # Создание директорий, если они не существуют
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('frames', exist_ok=True)
    logger.info('Запуск веб-сервиса')
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
