"""
Hardware detection module for TensorBench.
Cross-platform system information collector.
"""
from __future__ import annotations

import logging
import platform
import subprocess
from datetime import datetime, timezone
from typing import Optional

import cpuinfo
import psutil
from pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName, nvmlInit, nvmlShutdown, NVMLError

from ..core.models import (
    CPUInfo, GPUInfo, GPUVendor, HardwareInfo, OSPlatform, RAMInfo, StorageInfo
)

logger = logging.getLogger(__name__)


# =============================================================================
# PLATFORM DETECTION
# =============================================================================

def detect_os_platform() -> OSPlatform:
    """Detect current OS platform"""
    system = platform.system().lower()
    if system == "windows":
        return OSPlatform.WINDOWS
    elif system == "linux":
        return OSPlatform.LINUX
    elif system == "darwin":
        return OSPlatform.MACOS
    return OSPlatform.UNKNOWN  # type: ignore


def get_os_version() -> str:
    """Get OS version string"""
    if platform.system() == "Windows":
        return platform.version()
    return platform.release()


# =============================================================================
# CPU DETECTION
# =============================================================================

def detect_cpu() -> CPUInfo:
    """Detect CPU information"""
    try:
        info = cpuinfo.get_cpu_info()
        name = info.get("brand_raw", info.get("brand", "Unknown CPU"))
        
        # Get core/thread count
        cores = psutil.cpu_count(logical=False) or 1
        threads = psutil.cpu_count(logical=True) or cores
        
        # Get base frequency (in GHz)
        freq = psutil.cpu_freq()
        base_freq = freq.current / 1000 if freq else None
        
        # Check for AVX/AVX2/AVX512 (simplified heuristic for Windows)
        has_avx2 = "avx2" in info.get("flags", []) or platform.system() != "Windows"
        has_avx512 = "avx512" in info.get("flags", [])
        
        return CPUInfo(
            name=name,
            cores=cores,
            threads=threads,
            base_freq_ghz=round(base_freq, 2) if base_freq else None,
            has_avx2=has_avx2,
            has_avx512=has_avx512
        )
    except Exception as e:
        logger.warning(f"CPU detection failed: {e}")
        return CPUInfo(name="Unknown CPU", cores=1, threads=1)


# =============================================================================
# GPU DETECTION (Windows + NVIDIA focused for MVP)
# =============================================================================

def _detect_nvidia_gpu() -> Optional[GPUInfo]:
    """Detect NVIDIA GPU using NVML"""
    try:
        nvmlInit()
        count = nvmlDeviceGetCount()
        if count == 0:
            nvmlShutdown()
            return None
        
        # Use first GPU (for MVP; multi-GPU support later)
        handle = nvmlDeviceGetHandleByIndex(0)
        name = nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        vram_total = mem_info.total / (1024 ** 3)  # bytes → GB
        
        # Get compute capability (simplified: parse from name or use heuristic)
        cc = _guess_cuda_cc(name)
        
        nvmlShutdown()
        
        return GPUInfo(
            name=name,
            vendor=GPUVendor.NVIDIA,
            vram_total_gb=round(vram_total, 1),
            vram_available_gb=round(vram_total * 0.95, 1),  # heuristic: 95% available
            cuda_compute_capability=cc,
            supports_fp16=_supports_fp16(cc),
            supports_int8=_supports_int8(cc)
        )
    except NVMLError as e:
        logger.warning(f"NVML error (NVIDIA driver may not be installed): {e}")
        return None
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")
        return None


def _guess_cuda_cc(gpu_name: str) -> Optional[str]:
    """Guess CUDA compute capability from GPU name (heuristic table)"""
    cc_map = {
        "RTX 4090": "8.9", "RTX 4080": "8.9", "RTX 4070": "8.9", "RTX 4060": "8.9",
        "RTX 3090": "8.6", "RTX 3080": "8.6", "RTX 3070": "8.6", "RTX 3060": "8.6",
        "RTX 2080": "7.5", "RTX 2070": "7.5", "RTX 2060": "7.5",
        "GTX 1660": "7.5", "GTX 1650": "7.5",
        "GTX 1080": "6.1", "GTX 1070": "6.1", "GTX 1060": "6.1",
    }
    for pattern, cc in cc_map.items():
        if pattern.lower() in gpu_name.lower():
            return cc
    return None  # Unknown


def _supports_fp16(cuda_cc: Optional[str]) -> bool:
    """Check if GPU supports FP16 (Tensor Cores or native)"""
    if not cuda_cc:
        return False
    major = int(cuda_cc.split(".")[0])
    return major >= 7  # Volta+ has Tensor Cores


def _supports_int8(cuda_cc: Optional[str]) -> bool:
    """Check if GPU supports INT8 inference"""
    if not cuda_cc:
        return False
    major = int(cuda_cc.split(".")[0])
    return major >= 6  # Pascal+ has decent INT8


def detect_gpu() -> GPUInfo:
    """Main GPU detection entry point"""
    # Try NVIDIA first (priority for AI workloads)
    gpu = _detect_nvidia_gpu()
    if gpu:
        return gpu
    
    # TODO: Add AMD (ROCm) and Intel GPU detection
    # TODO: Add macOS Metal detection
    
    # Fallback
    return GPUInfo(name="Unknown GPU", vendor=GPUVendor.UNKNOWN)


# =============================================================================
# RAM & STORAGE DETECTION
# =============================================================================

def detect_ram() -> RAMInfo:
    """Detect RAM information"""
    mem = psutil.virtual_memory()
    return RAMInfo(
        total_gb=round(mem.total / (1024 ** 3), 1),
        available_gb=round(mem.available / (1024 ** 3), 1),
        # Speed and type require platform-specific tools (WMI on Windows)
        speed_mhz=None,
        type=None
    )


def detect_storage() -> StorageInfo:
    """Detect primary storage (system drive)"""
    # Get system drive
    if platform.system() == "Windows":
        path = "C:\\"
    else:
        path = "/"
    
    usage = psutil.disk_usage(path)
    
    # Try to detect SSD vs HDD (simplified)
    disk_type = None
    try:
        # On Windows, could use WMI to check MediaType
        # For MVP, leave as None
        pass
    except Exception:
        pass
    
    return StorageInfo(
        path=path,
        total_gb=round(usage.total / (1024 ** 3), 1),
        free_gb=round(usage.free / (1024 ** 3), 1),
        type=disk_type,
        read_speed_mbs=None  # Would require benchmark to measure
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def detect_hardware() -> HardwareInfo:
    """
    Main function: collect all hardware info and return typed model.
    Safe fallbacks: if any detection fails, returns partial info.
    """
    return HardwareInfo(
        os=detect_os_platform(),
        os_version=get_os_version(),
        cpu=detect_cpu(),
        gpu=detect_gpu(),
        ram=detect_ram(),
        storage=detect_storage(),
        detected_at=datetime.now(timezone.utc).isoformat()
    )


# =============================================================================
# CLI TEST (for development)
# =============================================================================

if __name__ == "__main__":
    # Quick test: run this file directly to see detected hardware
    import json
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Detecting hardware...")
    hw = detect_hardware()
    print("\n✅ Detected:")
    print(json.dumps(hw.model_dump(), indent=2, ensure_ascii=False))