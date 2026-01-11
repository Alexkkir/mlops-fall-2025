### Цель проекта
Pointwise метрика качества бездефектности ai-генераций. Модель получает на вход изображение (генерация text2image модели), выдает число, характеризующее степень дефектности изображения. Метрика может в дальнейшем использоваться для RL

### Целевые метрики для продакшена:
- Среднее время отклика сервиса ≤ 200 мс
- Доля неуспешных запросов ≤ 1 %
- Использование памяти/CPU — в пределах SLA
- Качество модели: точность ≥ 90 % или RMSE ≤ 0.1

## Набор данных
Исторические логи ассессорских замеров за последний год

## План экспериментов
1. Дообучить картиночную тушку (например, EfficientNet/CLIP)
2. Дообучить VLM-модель

Важно: на вход будут приходить изображения в размере 1024x1024. Важно, чтобы модель умела НАТИВНО поддерживать такой формат

## Метрика
- Accurac и другие стандартные метрики бинарной классификации. Поскольку модель работает как сиамская голова, можно использовать Accuracy вместо Balanced Accuracy
- Macro-Accuracy. Берем сбс, которые статзначимо прокрасились. Смотрим, совпадает ли вердикт сбс на ассессорах и сбс поверх обученной метрики

## Бейзлайны
- Запромпченные gpt-5.1, qwen-3-30b

## Запуск

### Установка зависимостей
```bash
uv sync
```

### DVC Пайплайн
Воспроизведение всего пайплайна (подготовка данных, обучение, валидация):
```bash
uv run dvc repro
```

### MLflow
Результаты экспериментов логируются в папку `mlruns`.
Чтобы посмотреть UI:
```bash
uv run mlflow ui
```

### Тестирование
```bash
uv run pytest tests
```

### Docker (Offline Inference)
Сборка образа:
```bash
docker build -t ml-app:v1 .
```

Запуск инференса:
```bash
# Предполагается, что данные лежат в папке ./data_input на хосте
docker run --rm \
  -v $(pwd)/data_input:/app/input \
  -v $(pwd)/data_output:/app/output \
  ml-app:v1 --input_path /app/input --output_path /app/output/preds.csv
```

### TorchServe (Online Inference)
Сборка образа для TorchServe:
1. Создайте `.mar` архив (если модель обновилась):
```bash
mkdir -p model_store
uv run torch-model-archiver --model-name pointwise_model \
  --version 1.0 \
  --model-file src/models/classic_cv_model.py \
  --serialized-file best_model.pth \
  --handler src/handler.py \
  --export-path model_store
```

2. Соберите Docker-образ:
```bash
docker build -f Dockerfile.torchserve -t pointwise-serve:v1 .
```

3. Запустите сервис:
```bash
docker run --rm -p 8080:8080 -p 8081:8081 pointwise-serve:v1
```

4. Проверьте работу (API):
```bash
# Тест с curl (отправка изображения)
curl -X POST http://localhost:8080/predictions/pointwise -T test_input/test.jpg
```
