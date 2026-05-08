"""
Core data models for TensorBench.
All models are Pydantic v2 compatible for validation and serialization.
"""
from __future__ import annotations

from enum import Enum, StrEnum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================

class OSPlatform(StrEnum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"


class GPUVendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


class ScenarioType(StrEnum):
    LLM = "llm"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class ReadinessStatus(StrEnum):
    READY = "ready"           # ✅ Потянет отлично
    LIMITED = "limited"       # ⚠️ Потянет с ограничениями
    NOT_READY = "not_ready"   # ❌ Не потянет


# =============================================================================
# HARDWARE MODELS
# =============================================================================

class CPUInfo(BaseModel):
    name: str
    cores: int
    threads: int
    base_freq_ghz: Optional[float] = None
    has_avx2: bool = True
    has_avx512: bool = False


class GPUInfo(BaseModel):
    name: str
    vendor: GPUVendor = GPUVendor.UNKNOWN
    vram_total_gb: Optional[float] = None
    vram_available_gb: Optional[float] = None
    cuda_compute_capability: Optional[str] = None  # e.g. "8.6"
    supports_fp16: bool = False
    supports_int8: bool = False
    driver_version: Optional[str] = None
    
    @field_validator('cuda_compute_capability')
    @classmethod
    def validate_cc(cls, v: Optional[str]) -> Optional[str]:
        """Validate CUDA compute capability format (e.g. '7.5')"""
        if v is None:
            return None
        parts = v.split('.')
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise ValueError(f"Invalid CUDA CC format: {v}")
        return v


class RAMInfo(BaseModel):
    total_gb: float
    available_gb: float
    speed_mhz: Optional[int] = None
    type: Optional[str] = None  # DDR4, DDR5, LPDDR4X


class StorageInfo(BaseModel):
    path: str
    total_gb: float
    free_gb: float
    type: Optional[str] = None  # SSD, NVMe, HDD
    read_speed_mbs: Optional[int] = None


class HardwareInfo(BaseModel):
    """Complete hardware snapshot of the user's system"""
    os: OSPlatform
    os_version: str
    
    cpu: CPUInfo
    gpu: GPUInfo
    ram: RAMInfo
    storage: StorageInfo
    
    # Optional metadata
    detected_at: Optional[str] = None  # ISO timestamp
    
    def get_bottleneck(self) -> Optional[str]:
        """Quick heuristic: return the most limiting component"""
        if self.gpu.vram_total_gb and self.gpu.vram_total_gb < 4:
            return "vram"
        if self.ram.total_gb < 8:
            return "ram"
        if self.storage.free_gb < 10:
            return "storage"
        return None


# =============================================================================
# SCENARIO & ANALYSIS MODELS
# =============================================================================

class ScenarioRequirements(BaseModel):
    """Requirements for a single AI scenario/model"""
    vram_min_gb: Optional[float] = None
    vram_recommended_gb: Optional[float] = None
    ram_min_gb: Optional[float] = None
    ram_recommended_gb: Optional[float] = None
    cuda_cc_min: Optional[str] = None
    requires_fp16: bool = False
    storage_min_gb: Optional[float] = None
    cpu_cores_min: Optional[int] = None
    notes: Optional[str] = None  # Human-readable hints


class ScenarioResult(BaseModel):
    """Result of checking hardware against a scenario"""
    scenario_id: str
    scenario_name: str
    scenario_type: ScenarioType
    
    status: ReadinessStatus
    score: float = Field(ge=0, le=100)  # 0-100 compatibility score
    
    metrics: dict[str, float] = Field(default_factory=dict)
    # e.g. {"vram_usage_pct": 85, "estimated_tps": 24.5}
    
    limitations: list[str] = Field(default_factory=list)
    # e.g. ["VRAM ограничивает размер контекста", "Только CPU-режим"]
    
    recommendations: list[str] = Field(default_factory=list)
    # e.g. ["Включить квантование Q4", "Уменьшить разрешение до 512x512"]


class ProfileResult(BaseModel):
    """Aggregated result for a whole profile (LLM, Image, etc.)"""
    profile_type: ScenarioType
    profile_name: str  # e.g. "Large Language Models"
    
    overall_status: ReadinessStatus
    overall_score: float = Field(ge=0, le=100)
    
    scenarios: list[ScenarioResult] = Field(default_factory=list)
    
    summary: Optional[str] = None  # One-liner for UI
    # e.g. "Потянет ~80% популярных LLM моделей до 13B параметров"


# =============================================================================
# UPGRADE & CONFIGURATOR MODELS
# =============================================================================

class UpgradeOption(BaseModel):
    """Single upgrade recommendation"""
    component: str  # "gpu", "ram", "storage"
    current: str    # e.g. "RTX 3060 12GB"
    suggested: str  # e.g. "RTX 4070 12GB"
    
    benefit: str    # e.g. "+40% скорости инференса"
    estimated_cost_rub: Optional[int] = None
    priority: int = Field(ge=1, le=3)  # 1 = high, 3 = low
    
    notes: Optional[str] = None


class HardwareCatalogEntry(BaseModel):
    """Entry in the hardware database for configurator"""
    id: str  # e.g. "nvidia_rtx_4070"
    name: str  # e.g. "NVIDIA GeForce RTX 4070"
    component_type: str  # "gpu", "cpu", "ram_module"
    
    specs: dict[str, str | int | float | bool]
    # e.g. {"vram_gb": 12, "cuda_cores": 5888, "tdp_w": 200}
    
    price_range_rub: Optional[tuple[int, int]] = None
    release_year: Optional[int] = None
    
    # For search/autocomplete
    aliases: list[str] = Field(default_factory=list)
    # e.g. ["4070", "RTX4070", "GeForce RTX 4070"]