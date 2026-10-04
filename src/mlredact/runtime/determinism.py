"""Process-level determinism controls (plan §13).

Call :func:`apply_process_settings` at the start of every process that runs numeric code (the CLI
parent and every worker initialiser).  Environment variables that libraries read at import or
thread-pool creation time are also exported to child processes via :func:`determinism_env`.
"""

from __future__ import annotations

import os
from typing import Any

from mlredact.config.schema import ThreadsConfig

# Never contact model hubs or telemetry endpoints from a processing process.
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    "TOKENIZERS_PARALLELISM": "false",
}


def determinism_env(threads: ThreadsConfig) -> dict[str, str]:
    blas = str(threads.blas)
    env = {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": blas,
        "OMP_THREAD_LIMIT": "1",  # Tesseract: one thread per process
        "OPENBLAS_NUM_THREADS": blas,
        "MKL_NUM_THREADS": blas,
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": blas,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OPENCV_OPENCL_RUNTIME": "disabled",
    }
    env.update(_OFFLINE_ENV)
    return env


def apply_process_settings(threads: ThreadsConfig) -> None:
    os.environ.update(determinism_env(threads))
    import cv2  # local import: cv2 must see the environment first

    cv2.setNumThreads(threads.opencv)
    cv2.ocl.setUseOpenCL(False)


def ort_session_options(threads: ThreadsConfig) -> Any:
    """ONNX Runtime options with fixed threads, sequential execution and deterministic kernels."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads.ort_intra_op
    so.inter_op_num_threads = threads.ort_inter_op
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.use_deterministic_compute = True
    so.enable_cpu_mem_arena = False
    so.log_severity_level = 3
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return so


def cpu_providers() -> list[str]:
    return ["CPUExecutionProvider"]
