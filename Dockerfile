# Использование официального образа Python
FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Установка зависимостей Python
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Копирование исходного кода
COPY app /app
WORKDIR /app

# Создание необходимых директорий
RUN mkdir /app/uploads
RUN mkdir /app/frames

# Запуск приложения
CMD ["python", "main.py"]
