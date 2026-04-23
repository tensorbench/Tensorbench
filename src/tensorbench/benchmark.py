"""
Модуль бенчмарка: скачивание моделей, запуск инференса и замер метрик.
"""
import os
import time
import sys
import urllib.request
from rich.console import Console
from rich.progress import Progress

# Импортируем движок ИИ
try:
    from llama_cpp import Llama
except ImportError:
    print("❌ Ошибка: llama-cpp-python не установлен.")
    print("👉 Выполни: pip install llama-cpp-python")
    sys.exit(1)

console = Console()

# Конфигурация моделей (размер -> URL и имя файла)
MODEL_URLS = {
    "0.5b": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "name": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "desc": "Лёгкая модель (быстрая проверка)"
    },
    "3b": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "name": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "desc": "Средняя модель (баланс)"
    },
    "7b": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
        "name": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "desc": "Тяжёлая модель (стресс-тест VRAM)"
    }
}

MODEL_DIR = "models"


def download_model(size: str = "0.5b"):
    """Скачивает модель, если её нет в папке models"""
    if size not in MODEL_URLS:
        console.print(f"[red]❌ Неизвестный размер модели: {size}. Доступны: {list(MODEL_URLS.keys())}[/red]")
        sys.exit(1)

    model_cfg = MODEL_URLS[size]
    model_path = os.path.join(MODEL_DIR, model_cfg["name"])

    # Проверяем, есть ли файл уже
    if os.path.exists(model_path):
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        console.print(f"[green]✅[/green] Модель найдена: [bold]{model_cfg['name']}[/bold] ({file_size_mb:.1f} MB)")
        return model_path

    console.print(f"\n📥 [bold]Скачивание модели ({model_cfg['desc']})[/bold]")
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        # Используем Rich Progress для красивого бара загрузки
        with Progress() as progress:
            task = progress.add_task("[cyan]Загрузка...", total=None)
            
            def reporthook(block_num, block_size, total_size):
                progress.update(task, advance=block_size)

            urllib.request.urlretrieve(model_cfg["url"], model_path, reporthook=reporthook)
            
            progress.update(task, completed=total_size, total=total_size)
            progress.refresh()

        console.print("[green]✅[/green] Скачивание завершено!\n")
        return model_path

    except Exception as e:
        console.print(f"[red]❌ Ошибка скачивания: {e}[/red]")
        sys.exit(1)


def run_benchmark(model_path: str):
    """Запускает генерацию текста и замеряет производительность"""
    console.print(f"[bold]🚀 Запуск бенчмарка...[/bold]")
    console.print("[dim]Инициализация модели (может занять время)...[/dim]")

    try:
        # Инициализация модели
        # n_gpu_layers=-1 означает использование видеокарты по максимуму (если доступна)
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  
            n_ctx=2048,       # Контекстное окно
            verbose=False     # Скрываем логи движка
        )

        prompt = "Explain the theory of relativity in 3 sentences."
        console.print(f"\n📝 Промпт: [italic]'{prompt}'[/italic]")
        console.print("\n🤖 [bold]Генерация ответа:[/bold]")
        
        # Замер времени
        start_time = time.time()
        token_count = 0
        
        # Генерация в потоковом режиме
        output = llm(prompt, max_tokens=100, stream=True)
        for chunk in output:
            token = chunk['choices'][0]['text']
            # Печатаем токен без переноса строки и буферизации
            console.print(token, end="", style="cyan")
            token_count += 1
            
        end_time = time.time()
        print("\n") # Отступ после генерации
        
        # Расчет метрик
        duration = end_time - start_time
        tps = token_count / duration if duration > 0 else 0
        
        return {
            "tokens": token_count,
            "duration": round(duration, 2),
            "tps": round(tps, 2)
        }

    except Exception as e:
        console.print(f"[red]❌ Ошибка при запуске модели: {e}[/red]")
        console.print("[dim]💡 Возможно, не хватает оперативной памяти или видеопамяти.[/dim]")
        return None


def print_results(results: dict):
    """Выводит итоговую таблицу с метриками"""
    if not results:
        return
    
    console.print("\n📊 [bold]Результаты бенчмарка:[/bold]")
    console.rule()
    console.print(f" Сгенерировано токенов : {results['tokens']}")
    console.print(f" Затраченное время    : {results['duration']} сек")
    console.print(f" Скорость (TPS)       : [bold green]{results['tps']} ток/сек[/bold green]")
    console.rule()
    
    # Вывод оценки скорости
    tps = results['tps']
    if tps > 50:
        console.print(f"[bold green]🚀 Отличная скорость! ({tps} TPS)[/bold green]")
    elif tps > 20:
        console.print(f"[bold yellow]✅ Хорошая скорость ({tps} TPS)[/bold yellow]")
    else:
        console.print(f"[bold red]🐢 Медленная скорость ({tps} TPS)[/bold red]")
