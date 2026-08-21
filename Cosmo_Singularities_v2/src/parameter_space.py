"""
Utilities for systematic exploration of the parameter space of

    p = omega * rho,
    Q = alpha * rho**beta,

in the barotropic diffusion model used by the project.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.models import BarotropicPowerDiffusionParams
from src.barotropic_analysis import analyze_barotropic_case


_REQUIRED_GRID_COLUMNS = {
    "point_id", "omega", "alpha", "beta", "rho0", "a0", "t0",
    "F0", "D0", "valid_initial_domain", "invalid_reason",
}


def build_parameter_grid(
    alpha_values: Iterable[float],
    beta_values: Iterable[float],
    omega_values: Iterable[float],
    *,
    rho0: float = 1.0,
    a0: float = 1.0,
    t0: float = 0.0,
    F0_min: float = 1e-10,
    D0_min: float = 1e-8,
) -> pd.DataFrame:
    """Construye una grilla cartesiana (omega, alpha, beta)."""
    if not math.isfinite(rho0) or rho0 <= 0.0:
        raise ValueError("rho0 debe ser positivo y finito.")
    if not math.isfinite(a0) or a0 <= 0.0:
        raise ValueError("a0 debe ser positivo y finito.")
    if not math.isfinite(t0):
        raise ValueError("t0 debe ser finito.")

    alphas = np.asarray(list(alpha_values), dtype=float)
    betas = np.asarray(list(beta_values), dtype=float)
    omegas = np.asarray(list(omega_values), dtype=float)

    for name, values in (("alpha_values", alphas), ("beta_values", betas), ("omega_values", omegas)):
        if values.size == 0:
            raise ValueError(f"{name} no puede estar vacío.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contiene valores no finitos.")

    rows = []
    point_id = 0
    for omega in omegas:
        for beta in betas:
            rho_power = rho0 ** (beta - 1.0)
            for alpha in alphas:
                F0 = 1.0 + alpha * rho_power
                D0 = 1.0 + alpha * beta * rho_power
                reasons = []
                if F0 <= F0_min:
                    reasons.append("F(rho0)<=0")
                if abs(D0) <= D0_min:
                    reasons.append("D(rho0)≈0")
                rows.append({
                    "point_id": point_id,
                    "omega": float(omega),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "rho0": float(rho0),
                    "a0": float(a0),
                    "t0": float(t0),
                    "F0": float(F0),
                    "D0": float(D0),
                    "valid_initial_domain": len(reasons) == 0,
                    "invalid_reason": "; ".join(reasons),
                })
                point_id += 1
    return pd.DataFrame(rows)


def _validate_grid(grid: pd.DataFrame) -> None:
    if not isinstance(grid, pd.DataFrame):
        raise TypeError("grid debe ser un pandas.DataFrame.")
    missing = _REQUIRED_GRID_COLUMNS - set(grid.columns)
    if missing:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(sorted(missing)))
    if grid.empty:
        raise ValueError("grid no puede estar vacío.")


def _checkpoint_write(rows, checkpoint_path):
    if checkpoint_path is None:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(checkpoint_path, index=False)


def analyze_parameter_grid(
    grid: pd.DataFrame,
    *,
    analysis_kwargs: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    checkpoint_every: int = 20,
    verbose: bool = True,
) -> pd.DataFrame:
    """Analiza la grilla, registra errores y permite reanudar desde CSV."""
    _validate_grid(grid)
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every debe ser >= 1.")

    kwargs = dict(analysis_kwargs or {})
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    completed_rows = []
    completed_ids = set()

    if resume and checkpoint is not None and checkpoint.exists():
        previous = pd.read_csv(checkpoint)
        if "point_id" in previous.columns:
            previous = previous.drop_duplicates(subset=["point_id"], keep="last")
            completed_rows = previous.to_dict(orient="records")
            completed_ids = {int(v) for v in previous["point_id"].dropna()}
            if verbose:
                print(f"Reanudando: {len(completed_ids)} puntos ya disponibles.")

    since_checkpoint = 0
    for _, point in grid.sort_values("point_id").iterrows():
        point_id = int(point["point_id"])
        if point_id in completed_ids:
            continue

        base = {
            "point_id": point_id,
            "omega": float(point["omega"]),
            "alpha": float(point["alpha"]),
            "beta": float(point["beta"]),
            "rho0": float(point["rho0"]),
            "a0": float(point["a0"]),
            "t0": float(point["t0"]),
            "F0": float(point["F0"]),
            "D0": float(point["D0"]),
            "valid_initial_domain": bool(point["valid_initial_domain"]),
            "invalid_reason": str(point.get("invalid_reason", "")),
        }

        if not base["valid_initial_domain"]:
            completed_rows.append({
                **base,
                "analysis_route": "excluded",
                "termination_event": None,
                "signed_termination_event": None,
                "standard_type": None,
                "behavior_label": "invalid_initial_domain",
                "status": "excluded",
                "is_singularity": None,
                "finite_time": None,
                "confidence": 0.0,
                "time_limit": "unknown",
                "rho_limit": "unknown",
                "p_limit": "unknown",
                "a_limit": "unknown",
                "H_limit": "unknown",
                "Hdot_limit": "unknown",
                "rho_dot_limit": "unknown",
                "Hddot_limit": "unknown",
                "rho_F": np.nan,
                "rho_D": np.nan,
                "estimated_limit_time": np.nan,
                "boundary_type": None,
                "turnaround_detected": False,
                "contracting_branch": False,
                "error": None,
            })
        else:
            params = BarotropicPowerDiffusionParams(
                omega=base["omega"], alpha=base["alpha"], beta=base["beta"],
                rho0=base["rho0"], a0=base["a0"], t0=base["t0"],
            )
            try:
                analysis = analyze_barotropic_case(
                    params=params,
                    case_id=f"grid_{point_id}",
                    **kwargs,
                )
                summary = analysis.summary_dict()
                classification = analysis.classification.as_dict()
                observed = classification.get("observed_behavior", {})
                completed_rows.append({
                    **base,
                    **summary,
                    "p_limit": observed.get("p", "unknown"),
                    "estimated_limit_time": analysis.evidence.get("estimated_limit_time"),
                    "boundary_type": analysis.evidence.get("boundary_type"),
                    "turnaround_detected": bool(
                        analysis.evidence.get(
                            "turnaround_detected",
                            analysis.diagnostics.get("signed_summary", {}).get("turnaround_detected", False),
                        )
                    ),
                    "contracting_branch": bool(analysis.evidence.get("contracting_branch", False)),
                    "error": None,
                })
            except Exception as error:
                completed_rows.append({
                    **base,
                    "analysis_route": "analysis_error",
                    "termination_event": None,
                    "signed_termination_event": None,
                    "standard_type": None,
                    "behavior_label": "analysis_error",
                    "status": "numerically_inconclusive",
                    "is_singularity": None,
                    "finite_time": None,
                    "confidence": 0.0,
                    "time_limit": "unknown",
                    "rho_limit": "unknown",
                    "p_limit": "unknown",
                    "a_limit": "unknown",
                    "H_limit": "unknown",
                    "Hdot_limit": "unknown",
                    "rho_dot_limit": "unknown",
                    "Hddot_limit": "unknown",
                    "rho_F": np.nan,
                    "rho_D": np.nan,
                    "estimated_limit_time": np.nan,
                    "boundary_type": None,
                    "turnaround_detected": False,
                    "contracting_branch": False,
                    "error": f"{type(error).__name__}: {error}",
                })

        since_checkpoint += 1
        if checkpoint is not None and since_checkpoint >= checkpoint_every:
            _checkpoint_write(completed_rows, checkpoint)
            since_checkpoint = 0
        if verbose and len(completed_rows) % 50 == 0:
            print(f"Puntos almacenados: {len(completed_rows)} / {len(grid)}")

    _checkpoint_write(completed_rows, checkpoint)
    result = pd.DataFrame(completed_rows)
    if not result.empty:
        result = result.drop_duplicates(subset=["point_id"], keep="last").sort_values("point_id").reset_index(drop=True)
    return result


def select_omega_slice(results: pd.DataFrame, omega: float, *, atol: float = 1e-12) -> pd.DataFrame:
    if "omega" not in results.columns:
        raise ValueError("results no contiene la columna omega.")
    mask = np.isclose(results["omega"].to_numpy(dtype=float), float(omega), atol=atol, rtol=0.0)
    return results.loc[mask].copy()


def behavior_matrix(results: pd.DataFrame, omega: float, *, value_column: str = "behavior_label") -> pd.DataFrame:
    sliced = select_omega_slice(results, omega)
    if sliced.empty:
        raise ValueError(f"No existen resultados para omega={omega}.")
    if value_column not in sliced.columns:
        raise ValueError(f"No existe la columna {value_column!r}.")
    return sliced.pivot_table(index="beta", columns="alpha", values=value_column, aggfunc="first").sort_index().sort_index(axis=1)


def encode_behavior_matrix(matrix: pd.DataFrame):
    labels = sorted({str(v) for v in matrix.to_numpy().ravel() if pd.notna(v)})
    label_to_code = {label: code for code, label in enumerate(labels)}
    codebook = {code: label for label, code in label_to_code.items()}
    encoded = np.full(matrix.shape, np.nan, dtype=float)
    values = matrix.to_numpy()
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if pd.notna(value):
                encoded[i, j] = label_to_code[str(value)]
    return encoded, codebook


def detect_behavior_boundaries(results: pd.DataFrame, omega: float, *, include_invalid: bool = False) -> pd.DataFrame:
    matrix = behavior_matrix(results, omega)
    alphas = matrix.columns.to_numpy(dtype=float)
    betas = matrix.index.to_numpy(dtype=float)
    labels = matrix.to_numpy(dtype=object)
    excluded = {"invalid_initial_domain", "analysis_error"}
    boundaries = []

    def valid_pair(a, b):
        if pd.isna(a) or pd.isna(b) or str(a) == str(b):
            return False
        if not include_invalid and (str(a) in excluded or str(b) in excluded):
            return False
        return True

    for i, beta in enumerate(betas):
        for j in range(len(alphas) - 1):
            left, right = labels[i, j], labels[i, j + 1]
            if valid_pair(left, right):
                boundaries.append({
                    "omega": float(omega), "orientation": "alpha",
                    "alpha_mid": float(0.5 * (alphas[j] + alphas[j + 1])),
                    "beta_mid": float(beta),
                    "label_1": str(left), "label_2": str(right),
                })

    for i in range(len(betas) - 1):
        for j, alpha in enumerate(alphas):
            lower, upper = labels[i, j], labels[i + 1, j]
            if valid_pair(lower, upper):
                boundaries.append({
                    "omega": float(omega), "orientation": "beta",
                    "alpha_mid": float(alpha),
                    "beta_mid": float(0.5 * (betas[i] + betas[i + 1])),
                    "label_1": str(lower), "label_2": str(upper),
                })
    return pd.DataFrame(boundaries)


def summarize_transitions(boundaries: pd.DataFrame) -> pd.DataFrame:
    if boundaries.empty:
        return pd.DataFrame(columns=["behavior_1", "behavior_2", "count"])
    pairs = [tuple(sorted((str(a), str(b)))) for a, b in zip(boundaries["label_1"], boundaries["label_2"])]
    counts = pd.Series(pairs).value_counts()
    rows = [{"behavior_1": pair[0], "behavior_2": pair[1], "count": int(count)} for pair, count in counts.items()]
    return pd.DataFrame(rows)


def detect_finite_time_beta_transition(results: pd.DataFrame, omega: float, *, require_positive_alpha: bool = True) -> pd.DataFrame:
    sliced = select_omega_slice(results, omega)
    sliced = sliced[sliced["valid_initial_domain"] & (sliced["rho_limit"] == "diverges")].copy()
    if require_positive_alpha:
        sliced = sliced[sliced["alpha"] > 0.0].copy()
    rows = []
    for alpha, group in sliced.groupby("alpha"):
        ordered = group.dropna(subset=["finite_time"]).sort_values("beta")
        if len(ordered) < 2:
            continue
        betas = ordered["beta"].to_numpy(dtype=float)
        finite = ordered["finite_time"].astype(bool).to_numpy()
        for i in range(len(betas) - 1):
            if finite[i] and not finite[i + 1]:
                rows.append({
                    "omega": float(omega), "alpha": float(alpha),
                    "beta_left": float(betas[i]), "beta_right": float(betas[i + 1]),
                    "beta_mid": float(0.5 * (betas[i] + betas[i + 1])),
                })
                break
    return pd.DataFrame(rows)


def compare_phantom_beta2_prediction(results: pd.DataFrame) -> pd.DataFrame:
    required = {"omega", "alpha", "beta", "rho_limit", "finite_time", "valid_initial_domain"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError("Faltan columnas para la comparación: " + ", ".join(sorted(missing)))
    selected = results[
        results["valid_initial_domain"]
        & (results["omega"] < -1.0)
        & (results["alpha"] > 0.0)
        & (results["rho_limit"] == "diverges")
        & (results["finite_time"].notna())
    ].copy()
    if selected.empty:
        return selected
    selected["asymptotic_finite_time_prediction"] = selected["beta"] < 2.0
    selected["finite_time_agreement"] = (
        selected["finite_time"].astype(bool)
        == selected["asymptotic_finite_time_prediction"].astype(bool)
    )
    return selected
