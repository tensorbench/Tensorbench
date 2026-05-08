"""
Real-world benchmark runner for TensorBench.
Downloads a small test model, measures actual TPS, and extrapolates predictions.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Optional dependency: llama-cpp-python for actual inference
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

from ..core.models import HardwareInfo, GPUVendor

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run"""
    model_url: str = "https://huggingface.co/TensorBench/tiny-llama-1.1b-chat-q4_k_m/resolve/main/tiny-llama-1.1b-chat-q4_k_m.gguf"
    model_path: Optional[Path] = None  # Auto-downloaded if None
    cache_dir: Path = Path.home() / ".tensorbench" / "models"
    
    # Test parameters
    prompt: str = "Explain quantum physics in 3 sentences."
    max_tokens: int = 100
    n_ctx: int = 512  # Small context for fast test
    
    # Hardware hints (auto-detected if not provided)
    force_cpu: bool = False


@dataclass
class BenchmarkResult:
    """Results of a real benchmark run"""
    # Measured metrics
    tokens_generated: int
    duration_seconds: float
    tps: float  # Tokens per second
    
    # Hardware context
    model_name: str
    model_size_mb: float
    hardware_snapshot: dict  # Simplified: gpu_name, vram_gb, cpu_cores
    
    # Extrapolation data
    baseline_factor: float = 1.0  # 1.0 = reference hardware
    extrapolation_notes: list[str]


@dataclass
class PredictionResult:
    """Predicted performance for a target model"""
    target_model: str  # e.g. "llama-3-8b-q4"
    predicted_tps: float
    confidence: str  # "high", "medium", "low"
    
    # Breakdown
    vram_ok: bool
    compute_bottleneck: Optional[str]  # "gpu", "cpu", "ram", None
    
    # Upgrade hint
    upgrade_potential: Optional[str]  # e.g. "RTX 4070: +45% TPS"


class BenchmarkRunner:
    """
    Main class for running benchmarks and generating predictions.
    Designed for async use in GUI (non-blocking).
    """
    
    # Reference TPS for TinyLlama-1.1B-Q4 on baseline hardware (RTX 3060 laptop)
    REFERENCE_TPS = 45.0
    
    # Approximate compute multipliers by GPU tier (for extrapolation)
    GPU_TIERS = {
        "RTX 4090": 3.5, "RTX 4080": 3.0, "RTX 4070": 2.4, "RTX 4060": 1.9,
        "RTX 3090": 2.8, "RTX 3080": 2.5, "RTX 3070": 2.0, "RTX 3060": 1.0,
        "RTX 2080": 1.6, "RTX 2070": 1.4, "RTX 2060": 1.2,
        "GTX 1660": 0.8, "GTX 1060": 0.5,
    }
    
    # Model size ratios (relative to TinyLlama-1.1B)
    MODEL_SIZE_RATIOS = {
        "llm_7b_q4": 7.0 / 1.1,
        "llm_13b_q4": 13.0 / 1.1,
        "llm_30b_q4": 30.0 / 1.1,
        "image_sd15": 4.0,  # Approximate compute ratio
        "image_sdxl": 12.0,
    }
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self._model: Optional[Llama] = None
        self._progress_cb: Optional[Callable[[float, str], None]] = None
    
    def set_progress_callback(self, cb: Callable[[float, str], None]):
        """Set callback for GUI progress updates (0.0-1.0, status text)"""
        self._progress_cb = cb
    
    def _progress(self, value: float, text: str):
        """Internal helper to call progress callback safely"""
        if self._progress_cb:
            try:
                self._progress_cb(value, text)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    async def run_async(self, hardware: Optional[HardwareInfo] = None) -> BenchmarkResult:
        """
        Run benchmark asynchronously (for GUI).
        Returns measured TPS and metadata.
        """
        if not LLAMA_AVAILABLE:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise RuntimeError("Benchmark requires llama-cpp-python. See README for installation.")
        
        loop = asyncio.get_event_loop()
        
        # Step 1: Download model (if needed)
        self._progress(0.1, "Проверка модели...")
        model_path = await loop.run_in_executor(
            None, self._ensure_model_downloaded
        )
        
        # Step 2: Load model
        self._progress(0.3, "Загрузка модели в память...")
        await loop.run_in_executor(
            None, lambda: self._load_model(model_path, hardware)
        )
        
        # Step 3: Run inference benchmark
        self._progress(0.5, "Запуск теста (генерация ответа)...")
        result = await loop.run_in_executor(
            None, self._run_inference_test
        )
        
        # Step 4: Add hardware context
        if hardware:
            result.hardware_snapshot = {
                "gpu_name": hardware.gpu.name,
                "gpu_vendor": hardware.gpu.vendor.value,
                "vram_gb": hardware.gpu.vram_total_gb,
                "cpu_cores": hardware.cpu.cores,
                "ram_gb": hardware.ram.total_gb,
            }
        
        # Step 5: Calculate baseline factor
        result.baseline_factor = self._calc_baseline_factor(result, hardware)
        result.extrapolation_notes = self._generate_extrapolation_notes(hardware)
        
        self._progress(1.0, "Готово!")
        return result
    
    def _ensure_model_downloaded(self) -> Path:
        """Download test model if not cached. Simplified for MVP."""
        cache_dir = self.config.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # For MVP: if model not found, raise with instructions
        # Real implementation: use huggingface_hub.snapshot_download()
        expected_path = cache_dir / "tiny-llama-1.1b-chat-q4_k_m.gguf"
        
        if not expected_path.exists():
            logger.info(f"Model not found. Download from: {self.config.model_url}")
            # TODO: Implement actual download with progress
            raise FileNotFoundError(
                f"Test model not found. Please download manually:\n"
                f"{self.config.model_url}\n"
                f"and save to: {expected_path}"
            )
        
        return expected_path
    
    def _load_model(self, model_path: Path, hardware: Optional[HardwareInfo]):
        """Load the GGUF model with appropriate settings"""
        # Auto-detect GPU offload
        n_gpu_layers = 0
        if hardware and hardware.gpu.vendor == GPUVendor.NVIDIA and not self.config.force_cpu:
            # Offload as many layers as VRAM allows (heuristic)
            if hardware.gpu.vram_total_gb and hardware.gpu.vram_total_gb >= 6:
                n_gpu_layers = -1  # All layers on GPU
        
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=self.config.n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
    
    def _run_inference_test(self) -> BenchmarkResult:
        """Run the actual inference and measure TPS"""
        if not self._model:
            raise RuntimeError("Model not loaded")
        
        start_time = time.perf_counter()
        
        # Run generation
        output = self._model(
            self.config.prompt,
            max_tokens=self.config.max_tokens,
            stop=["\n\n", "User:", "###"],
            echo=False,
        )
        
        end_time = time.perf_counter()
        
        # Count generated tokens (simplified)
        text = output["choices"][0]["text"]
        tokens = len(text.split())  # Approximate
        
        duration = end_time - start_time
        tps = tokens / duration if duration > 0 else 0
        
        return BenchmarkResult(
            tokens_generated=tokens,
            duration_seconds=round(duration, 2),
            tps=round(tps, 2),
            model_name="TinyLlama-1.1B-Q4",
            model_size_mb=650,  # Approximate
            hardware_snapshot={},
            extrapolation_notes=[]
        )
    
    def _calc_baseline_factor(self, result: BenchmarkResult, hardware: Optional[HardwareInfo]) -> float:
        """
        Calculate how user's hardware compares to reference.
        Factor > 1.0 = faster than reference, < 1.0 = slower.
        """
        if not hardware:
            return 1.0
        
        # Start with measured TPS vs reference
        factor = result.tps / self.REFERENCE_TPS
        
        # Adjust for GPU tier if known
        if hardware.gpu.name and hardware.gpu.vendor == GPUVendor.NVIDIA:
            for gpu_pattern, tier_multiplier in self.GPU_TIERS.items():
                if gpu_pattern.lower() in hardware.gpu.name.lower():
                    # If user's GPU is in our table, use tier as additional signal
                    expected_tier_tps = self.REFERENCE_TPS * tier_multiplier
                    # Blend measured vs expected
                    factor = factor * 0.7 + (result.tps / expected_tier_tps) * 0.3
                    break
        
        return round(factor, 2)
    
    def _generate_extrapolation_notes(self, hardware: Optional[HardwareInfo]) -> list[str]:
        """Generate human-readable notes for extrapolation"""
        notes = []
        if hardware:
            if hardware.gpu.vram_total_gb and hardware.gpu.vram_total_gb < 8:
                notes.append("Мало VRAM: большие модели могут не поместиться")
            if not hardware.gpu.supports_fp16:
                notes.append("GPU не поддерживает FP16: скорость может быть ниже")
            if hardware.cpu.cores < 6:
                notes.append("Мало ядер CPU: CPU-режим будет медленным")
        return notes
    
    def predict_for_model(self, benchmark: BenchmarkResult, target_model_id: str) -> PredictionResult:
        """
        Extrapolate predicted TPS for a target model based on benchmark results.
        """
        # Get size ratio
        size_ratio = self.MODEL_SIZE_RATIOS.get(target_model_id, 10.0)  # Default: 10x larger
        
        # Basic extrapolation: TPS inversely proportional to model size
        # Apply quantization factor (Q4 is ~2x faster than FP16 for same params)
        quant_factor = 2.0 if "q4" in target_model_id.lower() else 1.0
        
        base_prediction = (benchmark.tps / size_ratio) * quant_factor
        
        # Adjust for hardware bottlenecks
        hw = benchmark.hardware_snapshot
        vram_ok = True
        bottleneck = None
        
        # VRAM check (heuristic: 0.5 GB per 1B params for Q4)
        estimated_vram_needed = float(target_model_id.split('_')[1].replace('b','')) * 0.5
        if hw.get("vram_gb") and hw["vram_gb"] < estimated_vram_needed:
            vram_ok = False
            bottleneck = "vram"
            # If VRAM insufficient, assume CPU offload → much slower
            base_prediction *= 0.3
        
        # Compute bottleneck check
        elif hw.get("gpu_vendor") == "unknown" or not hw.get("gpu_name"):
            bottleneck = "cpu"
            base_prediction *= 0.5  # CPU is slower
        
        # Confidence based on how close we are to limits
        if vram_ok and bottleneck is None and benchmark.tps > 30:
            confidence = "high"
        elif vram_ok and benchmark.tps > 10:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Generate upgrade hint
        upgrade_hint = None
        if bottleneck == "vram" and hw.get("gpu_vendor") == "nvidia":
            upgrade_hint = "RTX 4070 12GB: +50-80% TPS, вмещает модели до 13B"
        elif bottleneck == "cpu":
            upgrade_hint = "Добавить дискретную GPU: +2-4× скорость"
        
        return PredictionResult(
            target_model=target_model_id,
            predicted_tps=round(max(0.1, base_prediction), 1),
            confidence=confidence,
            vram_ok=vram_ok,
            compute_bottleneck=bottleneck,
            upgrade_potential=upgrade_hint
        )
    
    def cleanup(self):
        """Free model resources"""
        if self._model:
            del self._model
            self._model = None