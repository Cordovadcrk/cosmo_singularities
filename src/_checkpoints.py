"""Identidad y escritura atómica de checkpoints; no contiene física."""

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd
import scipy

SIGNATURE_COLUMN = "_checkpoint_signature"


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        # Infinito puede ser un umbral válido; JSON estándar no lo representa.
        return {"float": str(value)}
    return value


def scan_signature(grid, configuration):
    """Huella de grilla, configuración efectiva, fuentes y entorno numérico."""
    core = Path(__file__).parent
    payload = {
        "schema": 1,
        "grid": grid.sort_values("point_id").to_dict(orient="records"),
        "configuration": configuration,
        "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(core.glob("*.py"))},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__},
    }
    encoded = json.dumps(_json_value(payload), sort_keys=True, allow_nan=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_checkpoint(rows, path, signature):
    """Reemplaza un CSV completo, sin exponer un archivo parcialmente escrito."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame[SIGNATURE_COLUMN] = signature
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_checkpoint(path, signature, grid, retry_failed=True):
    frame = pd.read_csv(path, float_precision="round_trip")
    if SIGNATURE_COLUMN not in frame or not frame[SIGNATURE_COLUMN].eq(signature).all():
        raise ValueError(
            "Checkpoint histórico sin firma o incompatible con grilla, configuración, "
            "core o entorno. Use un checkpoint_path nuevo para preservar los datos."
        )
    required = {"point_id", "status", "behavior_label", "error"}
    if not required.issubset(frame):
        raise ValueError("Checkpoint incompleto: faltan columnas de identificación/estado.")
    if frame.point_id.isna().any() or frame.point_id.duplicated().any():
        raise ValueError("Checkpoint con point_id ausentes o duplicados.")
    if not frame.point_id.isin(grid.point_id).all():
        raise ValueError("Checkpoint contiene puntos ajenos a la grilla.")
    if retry_failed:
        failed = (frame.error.fillna("").astype(str).str.strip().ne("")
                  | frame.status.eq("numerically_inconclusive")
                  | frame.behavior_label.eq("analysis_error"))
        frame = frame.loc[~failed]
    return frame.drop(columns=SIGNATURE_COLUMN).to_dict(orient="records")
