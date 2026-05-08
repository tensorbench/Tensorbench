"""
Analysis engine for TensorBench.
Compares hardware against AI model requirements and produces verdicts.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..core.models import (
    HardwareInfo,
    ScenarioRequirements,
    ScenarioResult,
    ProfileResult,
    ReadinessStatus,
    ScenarioType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SCENARIO CHECKER
# =============================================================================

def check_scenario(
    hardware: HardwareInfo,
    scenario_id: str,
    requirements: ScenarioRequirements
) -> ScenarioResult:
    """
    Check if hardware meets requirements for a specific AI scenario.
    Returns typed result with status, score, limitations and recommendations.
    """
    limitations = []
    recommendations = []
    metrics = {}
    
    # =====================================================================
    # VRAM CHECK (most critical for GPU inference)
    # =====================================================================
    vram_score = 100
    if requirements.vram_min_gb and hardware.gpu.vram_total_gb:
        vram_ratio = hardware.gpu.vram_total_gb / requirements.vram_min_gb
        
        if vram_ratio < 0.5:
            limitations.append(f"VRAM {hardware.gpu.vram_total_gb} ГБ < {requirements.vram_min_gb} ГБ минимума")
            recommendations.append("Включить квантование (Q4) или уменьшить размер модели")
            vram_score = 20
        elif vram_ratio < 1.0:
            limitations.append(f"VRAM {hardware.gpu.vram_total_gb} ГБ — на грани минимума")
            recommendations.append("Использовать CPU offload или снизить разрешение/контекст")
            vram_score = 50
        elif vram_ratio < 1.5:
            limitations.append("VRAM хватает, но без запаса")
            recommendations.append("Закрыть другие приложения перед запуском")
            vram_score = 75
        else:
            vram_score = 100
        
        metrics["vram_ratio"] = round(vram_ratio, 2)
    
    # =====================================================================
    # RAM CHECK (critical for CPU inference / model loading)
    # =====================================================================
    ram_score = 100
    if requirements.ram_min_gb:
        ram_ratio = hardware.ram.total_gb / requirements.ram_min_gb
        
        if ram_ratio < 0.7:
            limitations.append(f"ОЗУ {hardware.ram.total_gb} ГБ < {requirements.ram_min_gb} ГБ")
            recommendations.append("Добавить оперативную память или использовать swap-файл")
            ram_score = 30
        elif ram_ratio < 1.0:
            limitations.append("ОЗУ на грани минимума — возможны тормоза")
            recommendations.append("Закрыть браузер и фоновые приложения")
            ram_score = 60
        elif ram_ratio < 1.5:
            ram_score = 85
        else:
            ram_score = 100
        
        metrics["ram_ratio"] = round(ram_ratio, 2)
    
    # =====================================================================
    # CUDA COMPUTE CAPABILITY CHECK
    # =====================================================================
    cc_score = 100
    if requirements.cuda_cc_min and hardware.gpu.cuda_compute_capability:
        hw_major = int(hardware.gpu.cuda_compute_capability.split(".")[0])
        req_major = int(requirements.cuda_cc_min.split(".")[0])
        
        if hw_major < req_major:
            limitations.append(f"GPU не поддерживает вычисления уровня {requirements.cuda_cc_min}")
            recommendations.append("Запустить в режиме CPU (медленнее) или обновить GPU")
            cc_score = 40
        metrics["cuda_cc"] = hardware.gpu.cuda_compute_capability
    
    # =====================================================================
    # FP16 SUPPORT CHECK
    # =====================================================================
    if requirements.requires_fp16 and not hardware.gpu.supports_fp16:
        limitations.append("GPU не поддерживает FP16 — будет использован медленный FP32")
        recommendations.append("Включить квантование или использовать CPU-режим")
    
    # =====================================================================
    # CPU CORES CHECK (for CPU inference)
    # =====================================================================
    cpu_score = 100
    if requirements.cpu_cores_min:
        if hardware.cpu.cores < requirements.cpu_cores_min:
            limitations.append(f"CPU: {hardware.cpu.cores} ядер < {requirements.cpu_cores_min} рекомендуемых")
            recommendations.append("Использовать более лёгкую модель или обновить CPU")
            cpu_score = 50
        metrics["cpu_cores"] = hardware.cpu.cores
    
    # =====================================================================
    # STORAGE CHECK
    # =====================================================================
    storage_score = 100
    if requirements.storage_min_gb and hardware.storage.free_gb < requirements.storage_min_gb:
        limitations.append(f"Свободно {hardware.storage.free_gb} ГБ < {requirements.storage_min_gb} ГБ для модели")
        recommendations.append("Освободить место на диске")
        storage_score = 40
    
    # =====================================================================
    # CALCULATE FINAL SCORE AND STATUS
    # =====================================================================
    # Weighted average: VRAM is most important for AI workloads
    weights = {"vram": 0.4, "ram": 0.25, "cc": 0.15, "cpu": 0.1, "storage": 0.1}
    final_score = (
        vram_score * weights["vram"] +
        ram_score * weights["ram"] +
        cc_score * weights["cc"] +
        cpu_score * weights["cpu"] +
        storage_score * weights["storage"]
    )
    final_score = round(min(100, max(0, final_score)), 1)
    
    # Determine status
    if final_score >= 80 and not limitations:
        status = ReadinessStatus.READY
    elif final_score >= 50:
        status = ReadinessStatus.LIMITED
    else:
        status = ReadinessStatus.NOT_READY
    
    # Add generic recommendations if none specific
    if not recommendations and status != ReadinessStatus.READY:
        recommendations.append("Попробовать более лёгкую версию модели или квантование")
    
    # Generate human-readable scenario name from ID
    scenario_name = _format_scenario_name(scenario_id)
    
    return ScenarioResult(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        scenario_type=_infer_scenario_type(scenario_id),
        status=status,
        score=final_score,
        metrics=metrics,
        limitations=limitations,
        recommendations=recommendations
    )


# =============================================================================
# PROFILE ANALYZER (aggregate multiple scenarios)
# =============================================================================

def analyze_profile(
    hardware: HardwareInfo,
    profile_type: ScenarioType,
    scenarios: dict[str, ScenarioRequirements]
) -> ProfileResult:
    """
    Analyze hardware against all scenarios of a given profile type.
    Returns aggregated result with overall score and summary.
    """
    # Filter scenarios by profile type
    profile_scenarios = {
        sid: req for sid, req in scenarios.items()
        if _infer_scenario_type(sid) == profile_type
    }
    
    if not profile_scenarios:
        return ProfileResult(
            profile_type=profile_type,
            profile_name=_format_profile_name(profile_type),
            overall_status=ReadinessStatus.NOT_READY,
            overall_score=0,
            summary="Нет данных для этого профиля"
        )
    
    # Check each scenario
    results = [
        check_scenario(hardware, sid, req)
        for sid, req in profile_scenarios.items()
    ]
    
    # Calculate aggregated metrics
    scores = [r.score for r in results]
    avg_score = sum(scores) / len(scores)
    ready_count = sum(1 for r in results if r.status == ReadinessStatus.READY)
    
    # Determine overall status
    if avg_score >= 80 and ready_count >= len(results) * 0.7:
        overall_status = ReadinessStatus.READY
    elif avg_score >= 50:
        overall_status = ReadinessStatus.LIMITED
    else:
        overall_status = ReadinessStatus.NOT_READY
    
    # Generate summary
    summary = _generate_profile_summary(
        profile_type, avg_score, ready_count, len(results)
    )
    
    return ProfileResult(
        profile_type=profile_type,
        profile_name=_format_profile_name(profile_type),
        overall_status=overall_status,
        overall_score=round(avg_score, 1),
        scenarios=results,
        summary=summary
    )


# =============================================================================
# HELPERS
# =============================================================================

def _format_scenario_name(scenario_id: str) -> str:
    """Convert scenario_id to human-readable name"""
    mapping = {
        "llm_7b_q4": "Llama-3 / Mistral 7B (Q4)",
        "llm_13b_q4": "Llama-2 13B / Mixtral (Q4)",
        "llm_30b_q4": "Large LLM 30B+ (Q4)",
        "image_sd15": "Stable Diffusion 1.5",
        "image_sdxl": "Stable Diffusion XL",
        "image_flux": "Flux.1 Dev",
    }
    return mapping.get(scenario_id, scenario_id.replace("_", " ").title())


def _infer_scenario_type(scenario_id: str) -> ScenarioType:
    """Infer scenario type from ID prefix"""
    if scenario_id.startswith("llm"):
        return ScenarioType.LLM
    elif scenario_id.startswith("image"):
        return ScenarioType.IMAGE
    elif scenario_id.startswith("audio"):
        return ScenarioType.AUDIO
    elif scenario_id.startswith("video"):
        return ScenarioType.VIDEO
    return ScenarioType.LLM  # fallback


def _format_profile_name(profile_type: ScenarioType) -> str:
    """Convert profile type to display name"""
    mapping = {
        ScenarioType.LLM: " Текстовые модели (LLM)",
        ScenarioType.IMAGE: " Изображения",
        ScenarioType.AUDIO: "🎙️ Голос и аудио",
        ScenarioType.VIDEO: "🎥 Видео и апскейл",
    }
    return mapping.get(profile_type, profile_type.value)


def _generate_profile_summary(
    profile_type: ScenarioType,
    avg_score: float,
    ready_count: int,
    total_count: int
) -> str:
    """Generate one-liner summary for profile result"""
    pct = round(ready_count / total_count * 100) if total_count > 0 else 0
    
    if avg_score >= 80:
        return f"✅ Потянет ~{pct}% популярных моделей этого типа"
    elif avg_score >= 50:
        return f"⚠️ Потянет лёгкие модели, тяжёлые — с ограничениями"
    else:
        return f"❌ Рекомендуется апгрейд для комфортной работы"