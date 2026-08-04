import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class NCIWorkPacket:
    attempt: int
    device: str
    batch_size: int
    eig_batch_size: int
    cpu_threads: Optional[int]
    route_reason: str

    def to_dict(self) -> dict:
        return {
            "attempt": int(self.attempt),
            "device": self.device,
            "batch_size": int(self.batch_size),
            "eig_batch_size": int(self.eig_batch_size),
            "cpu_threads": self.cpu_threads,
            "route_reason": self.route_reason,
        }


class NCIDeviceRouter:
    def __init__(
        self,
        gpu_min_free_vram_mb: int = 768,
        gpu_retry_cooldown_jobs: int = 2,
        max_gpu_packet_retries: int = 3,
        min_gpu_batch_size: int = 20000,
        min_gpu_eig_batch_size: int = 15000,
    ):
        self.gpu_min_free_vram_mb = max(0, int(gpu_min_free_vram_mb))
        self.gpu_retry_cooldown_jobs = max(1, int(gpu_retry_cooldown_jobs))
        self.max_gpu_packet_retries = max(1, int(max_gpu_packet_retries))
        self.min_gpu_batch_size = max(1, int(min_gpu_batch_size))
        self.min_gpu_eig_batch_size = max(1, int(min_gpu_eig_batch_size))

        self._lock = threading.Lock()
        self._cooldown_jobs_remaining = 0
        self._consecutive_gpu_oom = 0
        self._safe_gpu_batch_size: Optional[int] = None
        self._safe_gpu_eig_batch_size: Optional[int] = None
        self._last_route_reason = "initial"
        self._last_oom_error = ""
        self._last_oom_at_epoch = 0.0

    def _cpu_threads(self) -> int:
        return max(1, int(os.cpu_count() or 1))

    def _requested_gpu_device(self, requested_device: str) -> Optional[str]:
        normalized = (requested_device or "auto").strip().lower()
        if normalized.startswith("cuda"):
            return normalized
        if normalized == "auto":
            return "cuda"
        return None

    def _free_vram_mb(self, cuda_device: str) -> Optional[float]:
        if not torch.cuda.is_available():
            return None
        device_index = 0
        if ":" in cuda_device:
            try:
                device_index = int(cuda_device.split(":", 1)[1])
            except Exception:
                device_index = 0
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device_index)
        except Exception:
            return None
        return float(free_bytes) / (1024.0 * 1024.0)

    def _downshift_gpu_sizes(self, batch_size: int, eig_batch_size: int) -> tuple[int, int]:
        next_batch = max(self.min_gpu_batch_size, int(batch_size) // 2)
        next_eig = max(self.min_gpu_eig_batch_size, int(eig_batch_size) // 2)
        return next_batch, next_eig

    def build_packets(
        self,
        requested_device: str,
        batch_size: int,
        eig_batch_size: int,
    ) -> list[NCIWorkPacket]:
        normalized = (requested_device or "auto").strip().lower()
        base_batch = max(1, int(batch_size))
        base_eig = max(1, int(eig_batch_size))

        with self._lock:
            if normalized == "cpu":
                self._last_route_reason = "user_forced_cpu"
                return [
                    NCIWorkPacket(
                        attempt=1,
                        device="cpu",
                        batch_size=base_batch,
                        eig_batch_size=base_eig,
                        cpu_threads=self._cpu_threads(),
                        route_reason=self._last_route_reason,
                    )
                ]

            gpu_device = self._requested_gpu_device(normalized)
            if gpu_device is None:
                self._last_route_reason = "non_cuda_device_passthrough"
                return [
                    NCIWorkPacket(
                        attempt=1,
                        device=normalized,
                        batch_size=base_batch,
                        eig_batch_size=base_eig,
                        cpu_threads=None,
                        route_reason=self._last_route_reason,
                    )
                ]

            if not torch.cuda.is_available():
                self._last_route_reason = "cuda_unavailable_routed_to_cpu"
                return [
                    NCIWorkPacket(
                        attempt=1,
                        device="cpu",
                        batch_size=base_batch,
                        eig_batch_size=base_eig,
                        cpu_threads=self._cpu_threads(),
                        route_reason=self._last_route_reason,
                    )
                ]

            if self._cooldown_jobs_remaining > 0:
                self._cooldown_jobs_remaining -= 1
                self._last_route_reason = (
                    f"gpu_cooldown_routed_to_cpu_{self._cooldown_jobs_remaining}_jobs_remaining"
                )
                return [
                    NCIWorkPacket(
                        attempt=1,
                        device="cpu",
                        batch_size=base_batch,
                        eig_batch_size=base_eig,
                        cpu_threads=self._cpu_threads(),
                        route_reason=self._last_route_reason,
                    )
                ]

            free_mb = self._free_vram_mb(gpu_device)
            if free_mb is not None and free_mb < float(self.gpu_min_free_vram_mb):
                self._cooldown_jobs_remaining = max(1, self.gpu_retry_cooldown_jobs - 1)
                self._last_route_reason = (
                    f"low_vram_{free_mb:.0f}mb_routed_to_cpu_threshold_{self.gpu_min_free_vram_mb}mb"
                )
                return [
                    NCIWorkPacket(
                        attempt=1,
                        device="cpu",
                        batch_size=base_batch,
                        eig_batch_size=base_eig,
                        cpu_threads=self._cpu_threads(),
                        route_reason=self._last_route_reason,
                    )
                ]

            seed_batch = min(base_batch, self._safe_gpu_batch_size or base_batch)
            seed_eig = min(base_eig, self._safe_gpu_eig_batch_size or base_eig)

            packets: list[NCIWorkPacket] = []
            seen = set()
            cur_batch = max(self.min_gpu_batch_size, seed_batch)
            cur_eig = max(self.min_gpu_eig_batch_size, seed_eig)
            for attempt in range(1, self.max_gpu_packet_retries + 1):
                key = (cur_batch, cur_eig)
                if key in seen:
                    break
                seen.add(key)
                reason = "gpu_primary_packet" if attempt == 1 else f"gpu_downshift_packet_{attempt}"
                packets.append(
                    NCIWorkPacket(
                        attempt=attempt,
                        device=gpu_device,
                        batch_size=cur_batch,
                        eig_batch_size=cur_eig,
                        cpu_threads=None,
                        route_reason=reason,
                    )
                )
                next_batch, next_eig = self._downshift_gpu_sizes(cur_batch, cur_eig)
                if next_batch == cur_batch and next_eig == cur_eig:
                    break
                cur_batch, cur_eig = next_batch, next_eig

            packets.append(
                NCIWorkPacket(
                    attempt=len(packets) + 1,
                    device="cpu",
                    batch_size=base_batch,
                    eig_batch_size=base_eig,
                    cpu_threads=self._cpu_threads(),
                    route_reason="cpu_tail_fallback_packet",
                )
            )
            self._last_route_reason = "gpu_packet_plan_with_cpu_tail_fallback"
            return packets

    def report_success(self, packet: NCIWorkPacket) -> None:
        with self._lock:
            device = (packet.device or "").strip().lower()
            if device.startswith("cuda"):
                self._consecutive_gpu_oom = 0
                self._cooldown_jobs_remaining = 0
                self._safe_gpu_batch_size = int(packet.batch_size)
                self._safe_gpu_eig_batch_size = int(packet.eig_batch_size)
                self._last_route_reason = "gpu_success"
            elif device == "cpu":
                self._last_route_reason = "cpu_success"

    def report_cuda_oom(self, packet: NCIWorkPacket, error: BaseException) -> None:
        with self._lock:
            self._consecutive_gpu_oom += 1
            self._cooldown_jobs_remaining = min(
                8,
                max(1, self.gpu_retry_cooldown_jobs + self._consecutive_gpu_oom - 1),
            )
            self._safe_gpu_batch_size, self._safe_gpu_eig_batch_size = self._downshift_gpu_sizes(
                int(packet.batch_size),
                int(packet.eig_batch_size),
            )
            self._last_route_reason = "cuda_oom"
            self._last_oom_error = str(error)
            self._last_oom_at_epoch = float(time.time())

    def state_snapshot(self) -> dict:
        with self._lock:
            return {
                "cooldown_jobs_remaining": int(self._cooldown_jobs_remaining),
                "consecutive_gpu_oom": int(self._consecutive_gpu_oom),
                "safe_gpu_batch_size": self._safe_gpu_batch_size,
                "safe_gpu_eig_batch_size": self._safe_gpu_eig_batch_size,
                "last_route_reason": self._last_route_reason,
                "last_oom_error": self._last_oom_error,
                "last_oom_at_epoch": self._last_oom_at_epoch,
                "gpu_min_free_vram_mb": int(self.gpu_min_free_vram_mb),
                "gpu_retry_cooldown_jobs": int(self.gpu_retry_cooldown_jobs),
                "max_gpu_packet_retries": int(self.max_gpu_packet_retries),
                "min_gpu_batch_size": int(self.min_gpu_batch_size),
                "min_gpu_eig_batch_size": int(self.min_gpu_eig_batch_size),
            }


_DEFAULT_NCI_ROUTER = NCIDeviceRouter()


def get_nci_router() -> NCIDeviceRouter:
    return _DEFAULT_NCI_ROUTER
