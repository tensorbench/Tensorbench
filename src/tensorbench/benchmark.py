"""Модуль бенчмарка: запуск ИИ и замер скорости (Phase 1)"""
import os
import time
import urllib.request
import sys
from rich.console import Console
from rich.progress import Progress

# Импортируем библиотеку ИИ
try:
    from llama_cpp import Llama
    import pynvml
except ImportError:
    print("❌ Ошибка: llama-cpp-python или pynvml не установлены. Запусти 'pip install -e .'")
    sys.exit(1)

console = Console()

MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_NAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_DIR = "models"

def download_model():
    """Скачивает модель, если её нет"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    
    if os.path.exists(model_path):
        console.print(f"[green]✅[/green] Модель найдена локально: {MODEL_NAME}")
        return model_path

    console.print(f"[blue]⬇️[/blue] Скачиваю модель {MODEL_NAME} (~400MB)...")
    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Загрузка...", total=None)
            def report(block_num, block_size, total_size):
                progress.update(task, advance=block_size)
            
            urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=report)
        console.print("[green]✅[/green] Скачивание завершено!")
        return model_path
    except Exception as e:
        console.print(f"[red]❌ Ошибка скачивания: {e}[/red]")
        sys.exit(1)

def run_benchmark(model_path):
    """Запускает тест генерации и замеряет TPS"""
    console.print("\n [bold]Запуск бенчмарка...[/bold]")
    console.print("[dim]Модель загружается в память (это может занять 10-20 сек)...[/dim]")

    try:
        # Загрузка модели (n_gpu_layers=-1 означает использование всей GPU)
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            verbose=False
        )

        prompt = "Explain quantum physics in 3 sentences."
        console.print(f"\n📝 Промпт: [italic]'{prompt}'[/italic]")
        console.print("\n🤖 Генерация ответа:")
        
        start_time = time.time()
        token_count = 0
        
        # Генерация
        output = llm(prompt, max_tokens=100, stream=True)
        for chunk in output:
            token = chunk['choices'][0]['text']
            sys.stdout.write(token)
            sys.stdout.flush()
            token_count += 1
            
        end_time = time.time()
        print("\n") # Отступ
        
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
        return None

def print_results(results):
    if not results:
        return
    
    console.print("\n📊 [bold]Результаты бенчмарка:[/bold]")
    console.rule()
    console.print(f" Всего токенов : {results['tokens']}")
    console.print(f" Время работы  : {results['duration']} сек")
    console.print(f" Скорость (TPS): [bold green]{results['tps']} ток/сек[/bold green]")
    console.rule()
    console.print("💡 TPS (Tokens Per Second) — главная метрика скорости ИИ.")
    console.print(f"   Результат {results['tps']} TPS считается ", end="")
    if results['tps'] > 20:
        console.print("[bold green]отличным![/bold green]")
    elif results['tps'] > 10:
        console.print("[bold yellow]хорошим.[/bold yellow]")
    else:
        console.print("[bold red]медленным.[/bold red]")
