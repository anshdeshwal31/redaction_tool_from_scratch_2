"""Runtime profile detection.

Bit-identical output is guaranteed per *runtime profile*: OS + architecture + CPU feature level
(+ GPU model for GPU tiers) + pinned native library versions.  Production configs pin the expected
profile id and workers refuse to start on a mismatch (plan §13).
"""

from __future__ import annotations

import contextlib
import platform
import sys
from dataclasses import dataclass, field
from functools import cache

from mlredact.config.schema import AppConfig
from mlredact.core.errors import EnvironmentErrorMl, ReasonCode

_V2 = ("SSE3", "SSSE3", "SSE41", "SSE42", "POPCNT")
_V3 = (*_V2, "AVX", "AVX2", "F16C", "FMA3")
_V4 = (*_V3, "AVX512F", "AVX512CD", "AVX512BW", "AVX512DQ", "AVX512VL")


def _numpy_cpu_features() -> dict[str, bool]:
    """CPU features as detected by numpy's dispatcher (private module, looked up defensively)."""
    import importlib

    for name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        features = getattr(module, "__cpu_features__", None)
        if isinstance(features, dict):
            return {str(k): bool(v) for k, v in features.items()}
    return {}


def _x86_level(features: dict[str, bool]) -> str:
    if all(features.get(f, False) for f in _V4):
        return "v4"
    if all(features.get(f, False) for f in _V3):
        return "v3"
    if all(features.get(f, False) for f in _V2):
        return "v2"
    return "v1"


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import onnxruntime

        versions["onnxruntime"] = onnxruntime.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import pypdfium2.version as pdfium_version

        versions["pypdfium2"] = str(pdfium_version.PYPDFIUM_INFO)
        versions["pdfium"] = str(pdfium_version.PDFIUM_INFO)
    except (ImportError, AttributeError):  # pragma: no cover
        pass
    try:
        from mlredact.ocr.tesseract import binary_version

        tesseract = binary_version()
        if tesseract:
            versions["tesseract"] = tesseract.removeprefix("tesseract").strip()
    except ImportError:  # pragma: no cover
        pass
    try:
        import tokenizers

        versions["tokenizers"] = str(tokenizers.__version__)
    except ImportError:  # pragma: no cover
        pass
    try:
        import pikepdf

        versions["pikepdf"] = pikepdf.__version__
        versions["qpdf"] = str(pikepdf.__libqpdf_version__)
    except ImportError:  # pragma: no cover
        pass
    try:
        import PIL
        from PIL import features

        versions["pillow"] = PIL.__version__
        versions["libjpeg_turbo"] = str(features.version("libjpeg_turbo"))
        versions["zlib"] = str(features.version("zlib"))
        versions["freetype2"] = str(features.version("freetype2"))
    except ImportError:  # pragma: no cover
        pass
    try:
        import cv2

        versions["opencv"] = cv2.__version__
    except ImportError:  # pragma: no cover
        pass
    with contextlib.suppress(Exception):
        from importlib.metadata import version

        versions["rapidocr"] = version("rapidocr")
    return dict(sorted(versions.items()))


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    profile_id: str
    os: str
    machine: str
    cpu_level: str
    libraries: dict[str, str] = field(default_factory=dict)


@cache
def detect_runtime_profile() -> RuntimeProfile:
    os_name = sys.platform if sys.platform != "win32" else "windows"
    os_name = "linux" if os_name.startswith("linux") else os_name
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    level = _x86_level(_numpy_cpu_features()) if machine == "x86_64" else "base"
    return RuntimeProfile(
        profile_id=f"{os_name}-{machine}-{level}",
        os=os_name,
        machine=machine,
        cpu_level=level,
        libraries=_library_versions(),
    )


def check_runtime_profile(config: AppConfig) -> RuntimeProfile:
    profile = detect_runtime_profile()
    expected = config.runtime.expected_profile
    if expected is not None and expected != profile.profile_id:
        raise EnvironmentErrorMl(ReasonCode.RUNTIME_PROFILE_MISMATCH)
    return profile
