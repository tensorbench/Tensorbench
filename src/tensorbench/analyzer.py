"""Модуль аналитики: сценарные рекомендации по апгрейду (v2)"""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# ===== СЦЕНАРИИ И ТРЕБОВАНИЯ =====
SCENARIOS = {
    "llm_7b": {
        "name": "🤖 LLM 7B (Llama/Mistral)",
        "min_vram": 6,      # GB для Q4
        "rec_vram": 8,      # GB для Q4 + контекст
        "min_ram": 16,
        "rec_ram": 32,
        "gpu_preferred": True,
        "desc": "Чат-боты, код, перевод"
    },
    "llm_13b": {
        "name": "🤖 LLM 13B (Llama2/Mixtral)",
        "min_vram": 8,
        "rec_vram": 12,
        "min_ram": 24,
        "rec_ram": 32,
        "gpu_preferred": True,
        "desc": "Сложные задачи, агенты, RAG"
    },
    "llm_30b": {
        "name": "🤖 LLM 30B+ (Command R, Yi)",
        "min_vram": 16,
        "rec_vram": 24,
        "min_ram": 32,
        "rec_ram": 64,
        "gpu_preferred": True,
        "desc": "Профессиональные задачи"
    },
    "image_sd15": {
        "name": "🎨 Stable Diffusion 1.5",
        "min_vram": 4,
        "rec_vram": 6,
        "min_ram": 8,
        "rec_ram": 16,
        "gpu_preferred": True,
        "desc": "Генерация 512x512, ~5-10 сек/изобр"
    },
    "image_sdxl": {
        "name": "🎨 Stable Diffusion XL",
        "min_vram": 8,
        "rec_vram": 12,
        "min_ram": 16,
        "rec_ram": 32,
        "gpu_preferred": True,
        "desc": "Генерация 1024x1024, ~10-20 сек/изобр"
    },
    "image_flux": {
        "name": "🎨 FLUX.1 (топ качество)",
        "min_vram": 12,
        "rec_vram": 16,
        "min_ram": 24,
        "rec_ram": 32,
        "gpu_preferred": True,
        "desc": "Профессиональная генерация"
    }
}

# ===== АПГРЕЙДЫ И ЦЕНЫ =====
UPGRADES = {
    "gpu_3060_12gb": {
        "name": "RTX 3060 12GB",
        "price": 280,
        "vram": 12,
        "perf_multiplier": 3.5,  # vs CPU
        "best_for": ["llm_13b", "image_sdxl"]
    },
    "gpu_4060ti_16gb": {
        "name": "RTX 4060 Ti 16GB",
        "price": 450,
        "vram": 16,
        "perf_multiplier": 5.0,
        "best_for": ["llm_30b", "image_flux"]
    },
    "gpu_4070_12gb": {
        "name": "RTX 4070 12GB",
        "price": 550,
        "vram": 12,
        "perf_multiplier": 6.5,
        "best_for": ["llm_13b", "image_sdxl"]
    },
    "ram_32gb": {
        "name": "ОЗУ 32GB DDR4",
        "price": 60,
        "vram": None,
        "perf_multiplier": 1.2,
        "best_for": ["llm_7b", "llm_13b"]
    },
    "ram_64gb": {
        "name": "ОЗУ 64GB DDR4",
        "price": 110,
        "vram": None,
        "perf_multiplier": 1.3,
        "best_for": ["llm_30b"]
    }
}

def check_scenario(hw_info: dict, scenario_key: str) -> dict:
    """Проверяет, потянет ли железо конкретный сценарий"""
    scenario = SCENARIOS[scenario_key]
    vram = hw_info.get("vram_total_gb") or 0
    ram = hw_info["ram_total_gb"]
    gpu = hw_info.get("gpu", "").lower()
    has_gpu = "nvidia" in gpu or "amd" in gpu or "intel" in gpu
    
    # Оценка готовности
    can_run = True
    limitations = []
    
    if scenario["gpu_preferred"] and not has_gpu:
        can_run = False
        limitations.append("❌ Нет дискретной GPU")
    elif scenario["gpu_preferred"] and has_gpu:
        if vram < scenario["min_vram"]:
            can_run = False
            limitations.append(f"❌ VRAM {vram}GB < {scenario['min_vram']}GB")
        elif vram < scenario["rec_vram"]:
            limitations.append(f"⚠️ VRAM {vram}GB (рекомендуется {scenario['rec_vram']}GB)")
    
    if ram < scenario["min_ram"]:
        can_run = False
        limitations.append(f"❌ RAM {ram}GB < {scenario['min_ram']}GB")
    elif ram < scenario["rec_ram"]:
        limitations.append(f"⚠️ RAM {ram}GB (рекомендуется {scenario['rec_ram']}GB)")
    
    return {
        "scenario": scenario,
        "can_run": can_run,
        "limitations": limitations,
        "ready": len(limitations) == 0
    }

def recommend_upgrades(hw_info: dict, target_scenarios: list) -> list:
    """Рекомендует апгрейды для целевых сценариев"""
    vram = hw_info.get("vram_total_gb") or 0
    ram = hw_info["ram_total_gb"]
    
    recommendations = []
    
    # Проверяем каждый апгрейд
    for upgrade_key, upgrade in UPGRADES.items():
        # Считаем, сколько сценариев станут доступны
        newly_available = 0
        for scenario_key in target_scenarios:
            scenario = SCENARIOS[scenario_key]
            
            # Проверяем, поможет ли этот апгрейд
            if upgrade["vram"] and upgrade["vram"] >= scenario["min_vram"]:
                if vram < scenario["min_vram"]:
                    newly_available += 1
            elif not upgrade["vram"] and upgrade_key.startswith("ram"):
                if ram < scenario["min_ram"] and upgrade["name"].split()[1].replace("GB", "") >= str(scenario["min_ram"]):
                    newly_available += 1
        
        if newly_available > 0:
            roi = upgrade["price"] / newly_available  # $ за новый сценарий
            recommendations.append({
                "upgrade": upgrade,
                "new_scenarios": newly_available,
                "roi": round(roi, 0),
                "priority": "high" if newly_available >= 2 else "medium"
            })
    
    # Сортируем по ROI (дешевле за сценарий = выше приоритет)
    recommendations.sort(key=lambda x: x["roi"])
    return recommendations

def analyze_performance(hw_info: dict, bench_result: dict) -> dict:
    """Основная функция аналитики (v2)"""
    tps = bench_result["tps"]
    vram = hw_info.get("vram_total_gb")
    ram = hw_info["ram_total_gb"]
    gpu = hw_info.get("gpu", "Unknown")
    
    # 1. Базовая оценка скорости
    if tps >= 100:
        perf_rating = "🚀 Отлично"
        perf_color = "green"
    elif tps >= 50:
        perf_rating = "✅ Хорошо"
        perf_color = "cyan"
    elif tps >= 20:
        perf_rating = "⚠️ Средне"
        perf_color = "yellow"
    else:
        perf_rating = "🐢 Медленно"
        perf_color = "red"
    
    # 2. Проверяем все сценарии
    scenario_results = {}
    for key in SCENARIOS.keys():
        scenario_results[key] = check_scenario(hw_info, key)
    
    # 3. Рекомендации для недоступных сценариев
    unavailable = [k for k, v in scenario_results.items() if not v["can_run"]]
    upgrade_recommendations = recommend_upgrades(hw_info, unavailable) if unavailable else []
    
    return {
        "tps": tps,
        "rating": perf_rating,
        "color": perf_color,
        "scenarios": scenario_results,
        "upgrade_recommendations": upgrade_recommendations[:3],  # Топ-3
        "current_tps": tps,
        "gpu": gpu,
        "vram": vram,
        "ram": ram
    }

def print_analysis(report: dict):
    """Выводит детальный отчёт (v2)"""
    console.print(f"\n[bold]📊 TensorBench: Полный отчёт[/bold]")
    console.rule()
    
    # Базовые метрики
    console.print(f"[bold]Текущая скорость:[/bold] {report['current_tps']} TPS ({report['rating']})")
    console.print(f"[bold]Конфигурация:[/bold] GPU: {report['gpu']}, VRAM: {report['vram']}GB, RAM: {report['ram']}GB")
    
    # Таблица сценариев
    console.print(f"\n[bold]🎯 Что можно запустить:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Задача", style="cyan")
    table.add_column("Статус", justify="center")
    table.add_column("Ограничения", style="yellow")
    
    for key, scenario in report["scenarios"].items():
        name = scenario["scenario"]["name"]
        status = "✅" if scenario["ready"] else ("⚠️" if scenario["can_run"] else "❌")
        limitations = ", ".join(scenario["limitations"]) if scenario["limitations"] else "Готово"
        table.add_row(name, status, limitations)
    
    console.print(table)
    
    # Рекомендации по апгрейду
    if report["upgrade_recommendations"]:
        console.print(f"\n[bold]💡 Топ-3 апгрейда для новых задач:[/bold]")
        for i, rec in enumerate(report["upgrade_recommendations"], 1):
            upgrade = rec["upgrade"]
            console.print(f"{i}. [bold]{upgrade['name']}[/bold] (~${upgrade['price']})")
            console.print(f"   → Откроет {rec['new_scenarios']} новых сценариев")
            console.print(f"   → ROI: ${rec['roi']} за сценарий")
    else:
        console.print(f"\n[bold green]✅ Ваша система готова ко всем популярным задачам![/bold green]")
    
    console.rule()
