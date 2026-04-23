"""Модуль сбора информации о железе (v0.2)"""
import platform
import psutil
import os
import subprocess

def _run_cmd(cmd: str) -> str:
    """Запускает системную команду и возвращает вывод"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""

def get_gpu_info() -> dict:
    """Определяет GPU и VRAM (кроссплатформенно)"""
    gpu_name = "Unknown"
    vram_gb = None

    # 1. NVIDIA (nvidia-smi)
    out = _run_cmd("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if out:
        parts = out.split(",")
        gpu_name = parts[0].strip()
        try:
            vram_mb = float(parts[1].strip().replace(" MiB", ""))
            vram_gb = round(vram_mb / 1024, 2)
        except (IndexError, ValueError):
            pass
        return {"name": gpu_name, "vram_gb": vram_gb}

    # 2. Windows (WMIC / PowerShell)
    if platform.system() == "Windows":
        out = _run_cmd('powershell -Command "(Get-CimInstance Win32_VideoController).Name"')
        if out:
            gpu_name = out.split("\n")[0].strip()
            # Детект VRAM на Windows требует WMI, оставим на потом
            return {"name": gpu_name, "vram_gb": vram_gb}

    # 3. Linux / macOS fallback
    if platform.system() == "Linux":
        out = _run_cmd("lspci | grep -i vga")
    elif platform.system() == "Darwin":
        out = _run_cmd("system_profiler SPDisplaysDataType | grep 'Chipset Model'")
    else:
        out = ""

    if out:
        gpu_name = out.split(":")[-1].strip() if ":" in out else out.strip()

    return {"name": gpu_name, "vram_gb": vram_gb}

def get_system_info() -> dict:
    """Возвращает полную информацию о системе"""
    gpu_data = get_gpu_info()
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "Unknown",
        "cpu_cores": os.cpu_count() or 0,
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "gpu": gpu_data["name"],
        "vram_total_gb": gpu_data["vram_gb"],
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
    if info['vram_total_gb']:
        print(f"VRAM      : {info['vram_total_gb']} GB")
    print("=" * 40)
    print("✅ Hardware detected. Ready for Phase 1: AI Inference Benchmark.\n")
