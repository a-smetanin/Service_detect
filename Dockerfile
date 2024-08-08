# Использование официального образа Python
FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install numpy==1.23.5  # Установка определенной версии numpy
RUN pip install -r requirements.txt

# Копирование исходного кода
COPY app /app
WORKDIR /app

# Копирование кастомных весов
COPY 640m.pt /app/640m.pt
COPY 640m.pt .
# Создание необходимых директорий
RUN mkdir /app/uploads
RUN mkdir /app/frames

# Запуск приложения
CMD ["python", "main.py"]
