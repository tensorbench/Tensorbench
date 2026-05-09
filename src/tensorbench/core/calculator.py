"""
Performance calculator for TensorBench.
Supports multiple workload profiles (Image, LLM).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from .hardware_db import (
    WORKLOAD_PRESETS,
    find_gpu, find_cpu, find_ram_by_name
)

@dataclass
class PerformancePrediction:
    model_name: str
    predicted_speed: float
    vram_usage_gb: float
    status: str
    relative_speed: float
    unit: str = "it/s"

@dataclass
class SystemScore:
    overall: float
    gpu_score: float
    cpu_score: float
    ram_score: float
    balance_factor: float
    tier_label: str

def calculate_sys_score(gpu_name: str, cpu_name: str, ram_name: str) -> SystemScore:
    gpu = find_gpu(gpu_name)
    cpu = find_cpu(cpu_name)
    ram = find_ram_by_name(ram_name)
    
    gpu_score = gpu.tier_score * 10.0
    cpu_score = cpu.tier_score * 10.0
    ram_score = ram.tier_score * 10.0
    
    max_tier = max(gpu.tier_score, cpu.tier_score, ram.tier_score)
    min_tier = min(gpu.tier_score, cpu.tier_score, ram.tier_score)
    balance_factor = max(0.7, min_tier / max(max_tier, 0.1))
    
    overall = (gpu_score * 0.5 + cpu_score * 0.3 + ram_score * 0.2) * balance_factor
    overall = min(10.0, max(0.5, overall))
    
    if overall >= 9.0: tier = "S"
    elif overall >= 7.5: tier = "A"
    elif overall >= 5.5: tier = "B"
    elif overall >= 3.5: tier = "C"
    else: tier = "D"
    
    return SystemScore(
        overall=round(overall, 1),
        gpu_score=round(gpu_score, 1),
        cpu_score=round(cpu_score, 1),
        ram_score=round(ram_score, 1),
        balance_factor=round(balance_factor, 2),
        tier_label=tier
    )

def predict_performance(gpu_name: str, ram_name: str, workload: str = "image") -> List[PerformancePrediction]:
    gpu = find_gpu(gpu_name)
    preset = WORKLOAD_PRESETS.get(workload, WORKLOAD_PRESETS["image"])
    base_perf = preset["base_perf_per_tier"] * gpu.tier_score
    unit = preset["unit"]
    
    predictions = []
    for model_name, specs in preset["models"].items():
        predicted_perf = base_perf * specs["relative_speed"]
        vram_needed = specs["vram_gb"]
        
        if vram_needed > gpu.vram_gb:
            predicted_perf *= 0.15
            status = "SUB-OPTIMAL"
        elif vram_needed > gpu.vram_gb * 0.85:
            predicted_perf *= 0.85
            status = "MARGINAL"
        else:
            status = "OPTIMAL"
            
        relative_speed = min(1.0, predicted_perf / base_perf)
        
        predictions.append(PerformancePrediction(
            model_name=model_name,
            predicted_speed=round(predicted_perf, 1),
            vram_usage_gb=vram_needed,
            status=status,
            relative_speed=round(relative_speed, 2),
            unit=unit
        ))
    return predictions

def generate_recommendations(gpu_name: str, cpu_name: str, ram_name: str, workload: str = "image") -> List[str]:
    gpu = find_gpu(gpu_name)
    cpu = find_cpu(cpu_name)
    recs = []
    
    preset = WORKLOAD_PRESETS.get(workload, WORKLOAD_PRESETS["image"])
    
    if gpu.vram_gb < 8:
        recs.append(f"⚠️ VRAM {gpu.vram_gb}GB ограничивает запуск тяжелых моделей. Рекомендуется 12GB+")
    elif gpu.vram_gb < 12 and workload == "image":
        recs.append(f"💡 VRAM {gpu.vram_gb}GB достаточно для большинства моделей. Для SDXL/Flux рекомендуется 16GB+")
    elif gpu.vram_gb < 16 and workload == "llm":
        recs.append(f"💡 Для LLM >14B рекомендуется 16GB+ VRAM")
    
    gpu_display_name = gpu.name.split("GeForce ")[-1] if "GeForce" in gpu.name else gpu.name
    
    if gpu.tier_score < 0.25:
        recs.append(f"🔧 {gpu_display_name} имеет ограниченную производительность для AI. Рассмотрите RTX 4060 Ti+")
    elif gpu.tier_score < 0.35:
        recs.append(f"💡 {gpu_display_name} подходит для базовых задач. Для тяжелых моделей лучше RTX 4070+")
    
    try:
        cap = int(ram_name.split("GB")[0])
        if cap < 16:
            recs.append(f"⚠️ ОЗУ {cap}GB критически мало. Рекомендуется 32GB+")
        elif cap < 32 and workload in ["llm", "video"]:
            recs.append(f"💡 Для {preset['name']} рекомендуется 64GB+ ОЗУ")
    except: pass
    
    good_cards = ["RTX 4060 Ti", "RTX 4070", "RTX 4070 Ti", "RTX 4080", "RTX 4090", "RTX 3080", "RTX 3090"]
    is_good_card = any(card in gpu.name for card in good_cards)
    
    if is_good_card and gpu.vram_gb >= 12:
        recs.append(f"✅ {gpu_display_name} ({gpu.vram_gb}GB) — отличная карта для {preset['name'].lower()}!")
    
    if not recs:
        recs.append(f"✅ Система подходит для {preset['name'].lower()}")
    
    return recs