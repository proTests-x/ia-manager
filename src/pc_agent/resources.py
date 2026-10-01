from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil


@dataclass(slots=True)
class GPUInfo:
    name: str
    total_bytes: int
    free_bytes: int
    backend: str = "CUDA"


@dataclass(slots=True)
class ResourceSnapshot:
    cpu: str
    physical_cores: int
    logical_threads: int
    ram_total: int
    ram_available: int
    disk_free: int
    gpus: list[GPUInfo] = field(default_factory=list)
    backends: list[str] = field(default_factory=list)

    @property
    def vram_total(self) -> int:
        return sum(gpu.total_bytes for gpu in self.gpus)

    @property
    def vram_free(self) -> int:
        return sum(gpu.free_bytes for gpu in self.gpus)

    def to_dict(self) -> dict:
        return asdict(self)


class ResourceDetector:
    def __init__(self, workspace: Path):
        self.workspace = workspace

    def detect(self) -> ResourceSnapshot:
        virtual = psutil.virtual_memory()
        disk = shutil.disk_usage(self.workspace)
        gpus = self._nvidia_gpus()
        backends = ["CPU"]
        if gpus:
            backends.append("CUDA")
        if shutil.which("ollama"):
            backends.append("Ollama")
        return ResourceSnapshot(
            cpu=platform.processor() or platform.machine() or "desconocida",
            physical_cores=psutil.cpu_count(logical=False) or 1,
            logical_threads=psutil.cpu_count(logical=True) or 1,
            ram_total=int(virtual.total),
            ram_available=int(virtual.available),
            disk_free=int(disk.free),
            gpus=gpus,
            backends=backends,
        )

    @staticmethod
    def _nvidia_gpus() -> list[GPUInfo]:
        executable = shutil.which("nvidia-smi")
        if not executable:
            return []
        try:
            completed = subprocess.run(
                [executable, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        result = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                result.append(GPUInfo(parts[0], int(float(parts[1]) * 1024**2), int(float(parts[2]) * 1024**2)))
            except ValueError:
                continue
        return result


def human_bytes(value: int) -> str:
    number = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(number) < 1024 or unit == "TB":
            return f"{number:.1f} {unit}"
        number /= 1024
    return f"{number:.1f} TB"

