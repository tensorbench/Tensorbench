"""Модуль аналитики: интерпретация метрик и рекомендации по апгрейду"""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Базовые пороги производительности (для 0.5B Q4 модели)
THRESHOLDS = {
    "excellent": 100,
    "good": 50,
    "average": 20,
    "slow": 10
}

# Ориентировочные цены на апгрейды (USD)
UPGRADE_COSTS = {
    "gpu_4060ti_16gb": 450,
    "gpu_3060_12gb": 280,
    "ram_32gb_ddr4": 60,
    "ram_64gb_ddr4": 110,
    "nvme_1tb_gen4": 80
}

def analyze_performance(hw_info: dict, bench_result: dict) -> dict:
    """Анализирует результаты и формирует отчёт"""
    tps = bench_result["tps"]
    vram = hw_info.get("vram_total_gb")
    ram = hw_info["ram_total_gb"]
    gpu = hw_info["gpu"]
    
    # 1. Оценка скорости
    if tps >= THRESHOLDS["excellent"]:
        perf_rating = " Отлично"
        perf_color = "green"
    elif tps >= THRESHOLDS["good"]:
        perf_rating = "✅ Хорошо"
        perf_color = "cyan"
    elif tps >= THRESHOLDS["average"]:
        perf_rating = "⚠️ Средне"
        perf_color = "yellow"
    else:
        perf_rating = "🐢 Медленно"
        perf_color = "red"

    # 2. Определение bottleneck (упрощённо для MVP)
    bottlenecks = []
    recommendations = []

    if "NVIDIA" in gpu.upper():
        if vram and vram < 12:
            bottlenecks.append("VRAM")
            recommendations.append(f"🔹 Замени GPU на RTX 3060 12GB / 4060 Ti 16GB (~${UPGRADE_COSTS['gpu_3060_12gb']}-${UPGRADE_COSTS['gpu_4060ti_16gb']})")
        if tps < 50 and vram and vram >= 12:
            bottlenecks.append("CUDA/Драйверы")
            recommendations.append("🔹 Установи CUDA Toolkit 12.4 + обновите драйверы NVIDIA Studio/Game Ready")
    else:
        bottlenecks.append("GPU")
        recommendations.append(f"🔹 Добавь дискретную GPU (RTX 3060 12GB от ${UPGRADE_COSTS['gpu_3060_12gb']}) для ускорения в 3-5 раз")

    if ram < 32:
        bottlenecks.append("RAM")
        recommendations.append(f"🔹 Добавь ОЗУ до 32/64 ГБ (~${UPGRADE_COSTS['ram_32gb_ddr4']}) для стабильной работы RAG/агентов")

    # 3. Расчёт ROI (условный)
    roi_note = "💡 ROI = стоимость апгрейда / ожидаемый прирост TPS. Чем ниже, тем выгоднее."
    
    return {
        "tps": tps,
        "rating": perf_rating,
        "color": perf_color,
        "bottlenecks": bottlenecks if bottlenecks else ["Нет явных ограничений"],
        "recommendations": recommendations if recommendations else ["Конфигурация сбалансирована для текущих задач"],
        "roi_note": roi_note
    }

def print_analysis(report: dict):
    """Выводит красивый отчёт в консоль"""
    console.print(f"\n📊 [bold]Аналитика TensorBench[/bold]")
    console.rule()
    
    # Таблица метрик
    table = Table(show_header=False, box=None)
    table.add_row("Скорость генерации:", f"[bold {report['color']}]{report['tps']} TPS[/bold {report['color']}]")
    table.add_row("Оценка:", report["rating"])
    console.print(table)
    
    console.print("\n[bold]🔍 Узкие места:[/bold]")
    for b in report["bottlenecks"]:
        console.print(f"  • {b}")
        
    console.print("\n[bold]💡 Рекомендации по апгрейду:[/bold]")
    for r in report["recommendations"]:
        console.print(f"  {r}")
        
    console.print(f"\n[dim]{report['roi_note']}[/dim]")
    console.rule()
