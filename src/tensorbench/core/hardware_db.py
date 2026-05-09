"""
Hardware database for TensorBench.
Contains specs, tiers, and workload presets for Image and LLM modes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import re

@dataclass
class GPU:
    name: str
    vram_gb: int
    vram_type: str
    bandwidth_gbps: float
    tflops_fp16: float
    tflops_fp8: float
    architecture: str
    tier_score: float
    price_tier: int

@dataclass
class CPU:
    name: str
    cores: int
    threads: int
    base_clock_ghz: float
    boost_clock_ghz: float
    pcie_gen: int
    tier_score: float

@dataclass
class RAM:
    name: str
    capacity_gb: int
    type: str
    speed_mhz: int
    channels: int
    tier_score: float

# =============================================================================
# DATABASES
# =============================================================================

GPU_DB: Dict[str, GPU] = {
    "NVIDIA GeForce RTX 4090 (24GB)": GPU("NVIDIA GeForce RTX 4090 (24GB)", 24, "GDDR6X", 1008.0, 82.6, 165.2, "Ada Lovelace", 1.0, 5),
    "NVIDIA GeForce RTX 4080 (16GB)": GPU("NVIDIA GeForce RTX 4080 (16GB)", 16, "GDDR6X", 716.8, 48.7, 97.4, "Ada Lovelace", 0.78, 4),
    "NVIDIA GeForce RTX 4070 Ti SUPER (16GB)": GPU("NVIDIA GeForce RTX 4070 Ti SUPER (16GB)", 16, "GDDR6X", 672.0, 40.1, 80.2, "Ada Lovelace", 0.68, 4),
    "NVIDIA GeForce RTX 4070 Ti (12GB)": GPU("NVIDIA GeForce RTX 4070 Ti (12GB)", 12, "GDDR6X", 504.2, 40.1, 80.2, "Ada Lovelace", 0.62, 3),
    "NVIDIA GeForce RTX 4070 SUPER (12GB)": GPU("NVIDIA GeForce RTX 4070 SUPER (12GB)", 12, "GDDR6X", 480.0, 35.5, 71.0, "Ada Lovelace", 0.55, 3),
    "NVIDIA GeForce RTX 4070 (12GB)": GPU("NVIDIA GeForce RTX 4070 (12GB)", 12, "GDDR6X", 504.2, 29.1, 58.2, "Ada Lovelace", 0.48, 3),
    "NVIDIA GeForce RTX 4060 Ti (16GB)": GPU("NVIDIA GeForce RTX 4060 Ti (16GB)", 16, "GDDR6", 288.0, 22.1, 44.2, "Ada Lovelace", 0.38, 2),
    "NVIDIA GeForce RTX 4060 Ti 8GB (8GB)": GPU("NVIDIA GeForce RTX 4060 Ti 8GB (8GB)", 8, "GDDR6", 288.0, 22.1, 44.2, "Ada Lovelace", 0.35, 2),
    "NVIDIA GeForce RTX 4060 (8GB)": GPU("NVIDIA GeForce RTX 4060 (8GB)", 8, "GDDR6", 272.0, 15.1, 30.2, "Ada Lovelace", 0.28, 1),
    "NVIDIA GeForce RTX 3090 Ti (24GB)": GPU("NVIDIA GeForce RTX 3090 Ti (24GB)", 24, "GDDR6X", 1008.0, 40.0, 80.0, "Ampere", 0.65, 4),
    "NVIDIA GeForce RTX 3090 (24GB)": GPU("NVIDIA GeForce RTX 3090 (24GB)", 24, "GDDR6X", 936.2, 35.6, 71.2, "Ampere", 0.60, 4),
    "NVIDIA GeForce RTX 3080 Ti (12GB)": GPU("NVIDIA GeForce RTX 3080 Ti (12GB)", 12, "GDDR6X", 912.4, 34.1, 68.2, "Ampere", 0.55, 3),
    "NVIDIA GeForce RTX 3080 (10GB)": GPU("NVIDIA GeForce RTX 3080 (10GB)", 10, "GDDR6X", 760.3, 29.8, 59.6, "Ampere", 0.48, 3),
    "NVIDIA GeForce RTX 3070 Ti (8GB)": GPU("NVIDIA GeForce RTX 3070 Ti (8GB)", 8, "GDDR6X", 608.3, 21.8, 43.6, "Ampere", 0.38, 2),
    "NVIDIA GeForce RTX 3070 (8GB)": GPU("NVIDIA GeForce RTX 3070 (8GB)", 8, "GDDR6", 448.0, 20.3, 40.6, "Ampere", 0.35, 2),
    "NVIDIA GeForce RTX 3060 Ti (8GB)": GPU("NVIDIA GeForce RTX 3060 Ti (8GB)", 8, "GDDR6", 448.0, 16.2, 32.4, "Ampere", 0.30, 2),
    "NVIDIA GeForce RTX 3060 (12GB)": GPU("NVIDIA GeForce RTX 3060 (12GB)", 12, "GDDR6", 360.0, 12.7, 25.4, "Ampere", 0.25, 1),
    "NVIDIA GeForce RTX 3050 (8GB)": GPU("NVIDIA GeForce RTX 3050 (8GB)", 8, "GDDR6", 224.0, 9.1, 18.2, "Ampere", 0.18, 1),
}

CPU_DB: Dict[str, CPU] = {
    "AMD Ryzen 9 7950X": CPU("AMD Ryzen 9 7950X", 16, 32, 4.5, 5.7, 5, 1.0),
    "AMD Ryzen 9 7900X": CPU("AMD Ryzen 9 7900X", 12, 24, 4.7, 5.6, 5, 0.85),
    "AMD Ryzen 7 7800X3D": CPU("AMD Ryzen 7 7800X3D", 8, 16, 4.2, 5.0, 5, 0.82),
    "AMD Ryzen 5 7600X": CPU("AMD Ryzen 5 7600X", 6, 12, 4.7, 5.3, 5, 0.60),
    "Intel Core i9-14900K": CPU("Intel Core i9-14900K", 24, 32, 3.2, 6.0, 5, 0.95),
    "Intel Core i7-14700K": CPU("Intel Core i7-14700K", 20, 28, 3.4, 5.6, 5, 0.80),
    "Intel Core i5-14600K": CPU("Intel Core i5-14600K", 14, 20, 3.5, 5.3, 5, 0.65),
    "Intel Core i9-13900K": CPU("Intel Core i9-13900K", 24, 32, 3.0, 5.8, 5, 0.90),
    "Intel Core i7-13700K": CPU("Intel Core i7-13700K", 16, 24, 3.4, 5.4, 5, 0.75),
    "Intel Core i5-13600K": CPU("Intel Core i5-13600K", 14, 20, 3.5, 5.1, 5, 0.60),
    "12th Gen Intel(R) Core(TM) i5-12400F": CPU("12th Gen Intel(R) Core(TM) i5-12400F", 6, 12, 2.5, 4.4, 5, 0.45),
    "Intel Core i5-12400F": CPU("Intel Core i5-12400F", 6, 12, 2.5, 4.4, 5, 0.45),
    "AMD Ryzen 5 5600X": CPU("AMD Ryzen 5 5600X", 6, 12, 3.7, 4.6, 4, 0.45),
}

RAM_DB: Dict[str, RAM] = {
    "128GB DDR5-6000": RAM("128GB DDR5-6000", 128, "DDR5", 6000, 2, 1.0),
    "64GB DDR5-6000": RAM("64GB DDR5-6000", 64, "DDR5", 6000, 2, 0.85),
    "64GB DDR5-5600": RAM("64GB DDR5-5600", 64, "DDR5", 5600, 2, 0.80),
    "64GB DDR4-3600": RAM("64GB DDR4-3600", 64, "DDR4", 3600, 2, 0.55),
    "64GB DDR4-3200": RAM("64GB DDR4-3200", 64, "DDR4", 3200, 2, 0.50),
    "32GB DDR5-6000": RAM("32GB DDR5-6000", 32, "DDR5", 6000, 2, 0.65),
    "32GB DDR5-5600": RAM("32GB DDR5-5600", 32, "DDR5", 5600, 2, 0.60),
    "32GB DDR4-3600": RAM("32GB DDR4-3600", 32, "DDR4", 3600, 2, 0.45),
    "32GB DDR4-3200": RAM("32GB DDR4-3200", 32, "DDR4", 3200, 2, 0.40),
    "16GB DDR4-3200": RAM("16GB DDR4-3200", 16, "DDR4", 3200, 2, 0.25),
    "16GB DDR4-2666": RAM("16GB DDR4-2666", 16, "DDR4", 2666, 2, 0.20),
}

# =============================================================================
# WORKLOAD PRESETS
# =============================================================================

WORKLOAD_PRESETS = {
    "image": {
        "name": "Image Synthesis",
        "icon": "🎨",
        "unit": "it/s",
        "base_perf_per_tier": 45.0,
        "models": {
            "SD v1.5 (FP16)": {"vram_gb": 4.2, "relative_speed": 1.0, "resolution": "512x512"},
            "SDXL (FP16)": {"vram_gb": 12.5, "relative_speed": 0.28, "resolution": "1024x1024"},
            "Flux.1-Dev (FP8)": {"vram_gb": 18.2, "relative_speed": 0.08, "resolution": "1024x1024"},
            "SD Cascade": {"vram_gb": 14.8, "relative_speed": 0.18, "resolution": "1024x1024"},
            "PixArt-alpha": {"vram_gb": 16.0, "relative_speed": 0.15, "resolution": "1024x1024"},
            "HunyuanDiT": {"vram_gb": 22.4, "relative_speed": 0.03, "resolution": "1024x1024"},
        }
    },
    "llm": {
        "name": "LLM Inference",
        "icon": "📝",
        "unit": "tok/s",
        "base_perf_per_tier": 35.0,
        "models": {
            "Qwen2.5-0.5B (Q4)": {"vram_gb": 0.5, "relative_speed": 1.0, "params": "0.5B"},
            "Mistral-7B (Q4)": {"vram_gb": 4.2, "relative_speed": 0.15, "params": "7B"},
            "Llama-3-8B (Q4)": {"vram_gb": 4.8, "relative_speed": 0.13, "params": "8B"},
            "Qwen2.5-14B (Q4)": {"vram_gb": 8.5, "relative_speed": 0.07, "params": "14B"},
            "Command-R-35B (Q4)": {"vram_gb": 20.0, "relative_speed": 0.03, "params": "35B"},
            "Llama-3-70B (Q4)": {"vram_gb": 40.0, "relative_speed": 0.015, "params": "70B"},
        }
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_gpu_list() -> List[str]: return sorted(GPU_DB.keys())
def get_cpu_list() -> List[str]: return sorted(CPU_DB.keys())
def get_ram_list() -> List[str]: return sorted(RAM_DB.keys())

def find_gpu(name: str) -> GPU:
    if not name: return list(GPU_DB.values())[-1]
    clean_name = re.sub(r'\s*\(\d+GB\)', '', name).lower()
    for db_name, gpu in GPU_DB.items():
        clean_db = re.sub(r'\s*\(\d+GB\)', '', db_name).lower()
        if clean_db in clean_name or clean_name in clean_db:
            return gpu
    return list(GPU_DB.values())[-1]

def find_cpu(name: str) -> CPU:
    if not name: return list(CPU_DB.values())[-1]
    name_lower = name.lower()
    for db_name, cpu in CPU_DB.items():
        if db_name.lower() in name_lower or name_lower in db_name.lower():
            return cpu
    return list(CPU_DB.values())[-1]

def find_ram(capacity: int, ram_type: str, speed: int) -> RAM:
    if not ram_type: ram_type = "DDR4"
    target = f"{capacity}GB {ram_type}-{speed}"
    if target in RAM_DB: return RAM_DB[target]
    best = list(RAM_DB.values())[0]
    min_diff = 1000
    for r in RAM_DB.values():
        if r.type == ram_type:
            diff = abs(r.capacity_gb - capacity)
            if diff < min_diff:
                min_diff = diff
                best = r
    return best

def find_ram_by_name(name: str) -> RAM:
    if name in RAM_DB:
        return RAM_DB[name]
    try:
        parts = name.split()
        cap = int(parts[0].replace("GB", ""))
        rtype = parts[1].split("-")[0]
        speed = int(parts[1].split("-")[1])
        return find_ram(cap, rtype, speed)
    except:
        return RAM_DB["16GB DDR4-3200"]