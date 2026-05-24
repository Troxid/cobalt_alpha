# Cobalt|Alpha

AI агент, схожий по функционалу на Wolfram|Alpha - консультирует и делает научные, инженерные, и математические расчёты.  
Проект создан исключительно в образовательных целях.


# Инициализация проекта

Используемый в проекте пакетный менеджер - `uv` ([Установка uv](https://docs.astral.sh/uv/getting-started/installation/))

Необходимо создать и заполнить своими параметрами `.env`.  
Пример доступных параметров в `.env.example`.  


```shell
# создание venv
uv venv

# активация venv
source .venv/bin/activate

# скачивание зависимостей
uv sync --dev
```

# Запуск агента с локальной LLM

В случае использования локальной ollama:
```shell
ollama serve &
ollama run gemma4:e4b & 
ollama run gemma4:26b &
```

Запуск веб-сервера
```
uv run cobalt-alpha
```

# Запуск платформы трейсинга (Arize Phoenix)

```
docker compose -f docker-compose.phoenix.yml up -d
```

# Ограничения 

Проблемы с безопасностью - использует питоновский eval для запуска скриптов расчета
