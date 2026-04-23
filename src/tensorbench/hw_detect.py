"""Модуль сбора информации о железе (v0.1)"""
import platform
import psutil
import os

def get_system_info() -> dict:
    """Возвращает базовую информацию о системе"""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "Unknown",
        "cpu_cores": os.cpu_count() or 0,
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "gpu": "Detection pending (Phase 1)",
        "vram_total_gb": None,
    }
    return info

def print_report(info: dict) -> None:
    """Выводит отчёт в консоль"""
    print("\n🖥️  TensorBench Hardware Report")
    print("=" * 40)
    print(f"OS        : {info['os']}")
    print(f"CPU       : {info['cpu']} ({info['cpu_cores']} cores)")
    print(f"RAM       : {info['ram_total_gb']} GB total / {info['ram_available_gb']} GB available")
    print(f"GPU       : {info['gpu']}")
    print("=" * 40)
    print("️  GPU detection will be added in next update.\n")
