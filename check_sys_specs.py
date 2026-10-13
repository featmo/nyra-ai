"""
==================================================
SYSTEM INFORMATION CHECKER
==================================================

Platform‑agnostic reporting of CPU, GPU and RAM usage.
Works on Windows, Linux and macOS.
"""
import platform
import subprocess
import sys
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import nvidia_smi
    HAS_NVIDIA_SMI = True
except ImportError:
    HAS_NVIDIA_SMI = False

try:
    import gpuinfo
    HAS_GPUINFO = True
except ImportError:
    HAS_GPUINFO = False


def get_os_info():
    """Return OS name and hostname."""
    os_name = platform.system()
    os_name = os_name.lower()
    if os_name == "windows":
        hostname = platform.node()
    elif os_name == "linux":
        hostname = platform.uname().nodename
    elif os_name == "darwin":
        hostname = platform.machine()
    else:
        hostname = "unknown"
    return os_name, hostname


def get_cpu_info():
    """Collect CPU information (core count, frequency, processor)."""
    info = {}
    if HAS_PSUTIL:
        # Number of logical CPUs
        info["logical_cores"] = psutil.cpu_count(logical=True)
        # Frequency (Hz) if available
        freq = psutil.cpu_freq()
        if freq:
            info["max_frequency_hz"] = freq.max
            info["current_frequency_hz"] = freq.current
        else:
            info["max_frequency_hz"] = None
            info["current_frequency_hz"] = None
    else:
        # Fallback: try platform.processor() (often returns CPU name)
        try:
            info["processor"] = platform.processor()
        except Exception:
            info["processor"] = "unknown"
    return info


def get_memory_info():
    """Collect RAM information (total, free, percentage used)."""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        info = {
            "total_gb": round(mem.total / (2**30), 2),
            "free_gb": round(mem.free / (2**30), 2),
            "percent_used": round(mem.percent, 2),
        }
    else:
        # Minimal fallback: try reading /proc/meminfo on Linux
        if sys.platform.startswith("linux"):
            try:
                with open("/proc/meminfo", "r") as f:
                    lines = f.readlines()
                    total = None
                    free = None
                    for line in lines:
                        if line.startswith("MemTotal:"):
                            total = int(line.split()[1])
                        elif line.startswith("MemFree:"):
                            free = int(line.split()[1])
                    if total:
                        info = {
                            "total_gb": round(total / (2**30), 2),
                            "free_gb": round(free / (2**30), 2) if free else None,
                            "percent_used": None,
                        }
                    else:
                        info = {}
            except Exception:
                info = {}
        else:
            info = {}
    return info


def get_gpu_info():
    """Try to collect GPU information (NVIDIA)."""
    if HAS_GPUINFO:
        try:
            # gpuinfo API varies by version - try several possibilities
            get_func = getattr(gpuinfo, "get_gpu_info", None)
            if callable(get_func):
                gpus = get_func()
                for g in gpus:
                    return {"vendor": g.vendor, "model": g.model, "memory_gb": g.memory_allocated()}
        except Exception:
            pass
    if HAS_NVIDIA_SMI:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu", "--format=csv", "-l 1"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split(",\n")
                return {
                    "vendor": lines[0].split("=")[1].strip() if lines else "unknown",
                    "memory_gb": lines[1].split("=")[1].strip() if len(lines) > 1 else "N/A",
                    "utilization_percent": lines[2].split("=")[1].strip() if len(lines) > 2 else "N/A",
                }
        except Exception:
            pass
    return {}


def main():
    os_name, hostname = get_os_info()
    cpu = get_cpu_info()
    mem = get_memory_info()
    gpu = get_gpu_info()

    print(f"OS: {os_name} | Host: {hostname}")
    print(f"CPU: {cpu.get('processor', 'unknown')} | Cores: {cpu.get('logical_cores', 'N/A')} | Max freq: {cpu.get('max_frequency_hz', 'N/A')} Hz")
    if mem:
        print(f"RAM: {mem.get('total_gb', 'N/A')} GB total | {mem.get('free_gb', 'N/A')} GB free | {mem.get('percent_used', 'N/A')}% used")
    else:
        print("RAM: not available")
    if gpu:
        g = gpu
        print(f"GPU: {g.get('vendor', 'unknown')} | Mem: {g.get('memory_gb', 'N/A')} GB | Util: {g.get('utilization_percent', 'N/A')}%")
    else:
        print("GPU: not detected")


if __name__ == "__main__":
    main()
