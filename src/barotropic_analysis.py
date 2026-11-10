"""
barotropic_analysis.py

Herramientas de análisis para el modelo cosmológico

    p = omega*rho,
    Q = alpha*rho**beta.

Este módulo no define las ecuaciones diferenciales del modelo.
Esas ecuaciones se encuentran en src.models.

Su objetivo es analizar las soluciones mediante:

- densidades críticas;
- tiempos asociados a densidades objetivo;
- barridos de umbrales;
- pruebas de convergencia;
- construcción automática de evidencia para clasificación.

La clasificación física final se realiza en src.classifiers.
"""

from __future__ import annotations

import math
import inspect

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from scipy.integrate import quad

from src.models import (
    BarotropicPowerDiffusionModel,
    BarotropicPowerDiffusionParams,
)

from src.classifiers import (
    classify_barotropic_power_diffusion,
)

@dataclass(frozen=True)
class CriticalDensities:
    """
    Densidades críticas del modelo barotrópico con difusión.

    Parameters
    ----------
    rho_F:
        Raíz positiva aislada de

            F(rho) = 1 + alpha*rho**(beta - 1) = 0.

        En esta frontera se cumple H=0.

    rho_D:
        Raíz positiva aislada de

            D(rho)
            =
            1 + alpha*beta*rho**(beta - 1)
            =
            0.

        En esta frontera puede divergir dot(rho).

    F_degenerate:
        True cuando F(rho)=0 para toda densidad, como puede ocurrir
        formalmente para beta=1 y alpha=-1.

    D_degenerate:
        True cuando D(rho)=0 para toda densidad.

    notes:
        Información adicional sobre casos especiales.
    """

    rho_F: float | None

    rho_D: float | None

    F_degenerate: bool = False

    D_degenerate: bool = False

    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        """
        Convierte el resultado a un diccionario.
        """

        return asdict(self)

def compute_critical_densities(
    params: BarotropicPowerDiffusionParams,
    *,
    tolerance: float = 1e-12,
) -> CriticalDensities:
    """
    Calcula las densidades críticas positivas del modelo.

    Las funciones relevantes son

        F(rho)
        =
        1 + alpha*rho**(beta - 1),

    y

        D(rho)
        =
        1 + alpha*beta*rho**(beta - 1).

    Para beta != 1:

        rho_F
        =
        (-1/alpha)**[1/(beta - 1)],

    cuando alpha < 0, y

        rho_D
        =
        (-1/(alpha*beta))**[1/(beta - 1)],

    cuando alpha*beta < 0.
    """

    if not isinstance(
        params,
        BarotropicPowerDiffusionParams,
    ):
        raise TypeError(
            "params debe ser BarotropicPowerDiffusionParams."
        )

    params.validate()

    if not math.isfinite(tolerance):
        raise ValueError(
            "tolerance debe ser finita."
        )

    if tolerance <= 0.0:
        raise ValueError(
            "tolerance debe ser positiva."
        )

    alpha = float(params.alpha)
    beta = float(params.beta)

    rho_F = None
    rho_D = None

    F_degenerate = False
    D_degenerate = False

    notes = []

    # ---------------------------------------------------------
    # Caso beta = 1
    # ---------------------------------------------------------

    if math.isclose(
        beta,
        1.0,
        abs_tol=tolerance,
        rel_tol=0.0,
    ):
        constant_F = 1.0 + alpha
        constant_D = 1.0 + alpha

        if math.isclose(
            constant_F,
            0.0,
            abs_tol=tolerance,
            rel_tol=0.0,
        ):
            F_degenerate = True

            notes.append(
                "Para beta=1 y alpha=-1, F(rho)=0 "
                "para toda rho."
            )

        if math.isclose(
            constant_D,
            0.0,
            abs_tol=tolerance,
            rel_tol=0.0,
        ):
            D_degenerate = True

            notes.append(
                "Para beta=1 y alpha=-1, D(rho)=0 "
                "para toda rho."
            )

        return CriticalDensities(
            rho_F=None,
            rho_D=None,
            F_degenerate=F_degenerate,
            D_degenerate=D_degenerate,
            notes=" ".join(notes),
        )

    # ---------------------------------------------------------
    # Frontera F(rho)=0
    # ---------------------------------------------------------

    if alpha < 0.0:
        base_F = -1.0 / alpha

        rho_F_candidate = (
            base_F
            ** (
                1.0
                / (beta - 1.0)
            )
        )

        if (
            math.isfinite(rho_F_candidate)
            and rho_F_candidate > 0.0
        ):
            rho_F = float(
                rho_F_candidate
            )

    # ---------------------------------------------------------
    # Frontera D(rho)=0
    # ---------------------------------------------------------

    alpha_beta = alpha * beta

    if alpha_beta < 0.0:
        base_D = (
            -1.0
            / alpha_beta
        )

        rho_D_candidate = (
            base_D
            ** (
                1.0
                / (beta - 1.0)
            )
        )

        if (
            math.isfinite(rho_D_candidate)
            and rho_D_candidate > 0.0
        ):
            rho_D = float(
                rho_D_candidate
            )

    return CriticalDensities(
        rho_F=rho_F,
        rho_D=rho_D,
        F_degenerate=False,
        D_degenerate=False,
        notes=" ".join(notes),
    )

def lies_between(
    value: float,
    endpoint_1: float,
    endpoint_2: float,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """
    Determina si value pertenece al intervalo cerrado definido
    por endpoint_1 y endpoint_2.
    """

    lower = min(
        endpoint_1,
        endpoint_2,
    )

    upper = max(
        endpoint_1,
        endpoint_2,
    )

    scale = max(
        abs(lower),
        abs(upper),
        1.0,
    )

    absolute_tolerance = (
        tolerance
        * scale
    )

    return bool(
        lower - absolute_tolerance
        <= value
        <= upper + absolute_tolerance
    )


def initial_density_direction(
    params: BarotropicPowerDiffusionParams,
) -> int:
    """
    Determina la dirección inicial de la evolución de rho.

    Returns
    -------
    int
        +1 si inicialmente dot(rho)>0.

        -1 si inicialmente dot(rho)<0.

         0 si inicialmente dot(rho)=0.
    """

    model = BarotropicPowerDiffusionModel(
        params
    )

    rho0 = float(params.rho0)

    F0 = model.F(rho0)
    D0 = model.D(rho0)

    if F0 < 0.0:
        raise ValueError(
            "La condición inicial está fuera del dominio F>=0."
        )

    if math.isclose(
        D0,
        0.0,
        abs_tol=1e-14,
        rel_tol=0.0,
    ):
        raise ValueError(
            "La condición inicial satisface D(rho0)=0."
        )

    H0 = model.signed_hubble(
        rho0,
        branch=1.0,
    )

    rho_dot_0 = (
        -3.0
        * H0
        * (1.0 + params.omega)
        * rho0
        / D0
    )

    if math.isclose(
        rho_dot_0,
        0.0,
        abs_tol=1e-14,
        rel_tol=0.0,
    ):
        return 0

    return int(
        np.sign(rho_dot_0)
    )

def state_at_density(
    params: BarotropicPowerDiffusionParams,
    rho_target: float,
    *,
    require_future: bool = True,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    integration_limit: int = 500,
    boundary_tolerance: float = 1e-12,
) -> dict[str, float]:
    """
    Calcula el estado cosmológico cuando rho alcanza rho_target.

    Se utiliza cosmológico cuando rho alcanza rho_target.

    Se utiliza la variable

        u = ln(rho),

    para la cual

        dt/du
        =
        -D(rho)
        /
        [3(1 + omega)H(rho)].

    La integración se realiza sobre la rama expansiva H>=0.

    Parameters
    ----------
    params:
        Parámetros del modelo.

    rho_target:
        Densidad positiva que se desea alcanzar.

    require_future:
        Si es True, se exige que rho_target se encuentre en la
        dirección futura de la solución desde rho0.

    Returns
    -------
    dict
        Estado cosmológico evaluado en rho_target.
    """

    if not isinstance(
        params,
        BarotropicPowerDiffusionParams,
    ):
        raise TypeError(
            "params debe ser BarotropicPowerDiffusionParams."
        )

    params.validate()

    if not math.isfinite(rho_target):
        raise ValueError(
            "rho_target debe ser finita."
        )

    if rho_target <= 0.0:
        raise ValueError(
            "rho_target debe ser positiva."
        )

    if epsabs <= 0.0 or epsrel <= 0.0:
        raise ValueError(
            "epsabs y epsrel deben ser positivos."
        )

    if integration_limit < 1:
        raise ValueError(
            "integration_limit debe ser positivo."
        )

    model = BarotropicPowerDiffusionModel(
        params
    )

    rho0 = float(params.rho0)
    omega = float(params.omega)

    if math.isclose(
        rho_target,
        rho0,
        rel_tol=1e-14,
        abs_tol=1e-14,
    ):
        F_value = model.F(rho0)
        D_value = model.D(rho0)

        H_value = model.signed_hubble(
            rho0,
            branch=1.0,
        )

        Hdot_value = (
            -0.5
            * (1.0 + omega)
            * rho0
        )

        rho_dot_value = (
            -3.0
            * H_value
            * (1.0 + omega)
            * rho0
            / D_value
        )

        Hddot_value = (
            -0.5
            * (1.0 + omega)
            * rho_dot_value
        )

        return {
            "t": float(params.t0),
            "rho": rho0,
            "log_a": float(
                math.log(params.a0)
            ),
            "a": float(params.a0),
            "F": float(F_value),
            "D": float(D_value),
            "H": float(H_value),
            "Hdot": float(Hdot_value),
            "rho_dot": float(
                rho_dot_value
            ),
            "Hddot": float(
                Hddot_value
            ),
            "quadrature_error": 0.0,
        }

    if math.isclose(
        1.0 + omega,
        0.0,
        abs_tol=1e-14,
        rel_tol=0.0,
    ):
        raise ValueError(
            "Para omega=-1, rho es constante y no puede "
            "alcanzar una densidad distinta de rho0."
        )

    # ---------------------------------------------------------
    # Comprobar que el objetivo pertenece a la rama futura
    # ---------------------------------------------------------

    future_direction = (
        initial_density_direction(
            params
        )
    )

    if require_future:
        if future_direction == 0:
            raise ValueError(
                "La densidad inicial es estacionaria; no existe "
                "una dirección futura distinta de rho0."
            )

        if (
            future_direction > 0
            and rho_target < rho0
        ):
            raise ValueError(
                "rho_target está en la dirección pasada. "
                "La densidad crece inicialmente."
            )

        if (
            future_direction < 0
            and rho_target > rho0
        ):
            raise ValueError(
                "rho_target está en la dirección pasada. "
                "La densidad decrece inicialmente."
            )

    # ---------------------------------------------------------
    # Verificar que no se crucen fronteras físicas
    # ---------------------------------------------------------

    critical = compute_critical_densities(
        params,
        tolerance=boundary_tolerance,
    )

    if critical.F_degenerate:
        raise ValueError(
            "F(rho)=0 en todo el dominio."
        )

    if critical.D_degenerate:
        raise ValueError(
            "D(rho)=0 en todo el dominio."
        )

    if (
        critical.rho_F is not None
        and lies_between(
            critical.rho_F,
            rho0,
            rho_target,
            tolerance=boundary_tolerance,
        )
    ):
        raise ValueError(
            "La trayectoria hacia rho_target alcanza o cruza "
            "la frontera F(rho)=0."
        )

    if (
        critical.rho_D is not None
        and lies_between(
            critical.rho_D,
            rho0,
            rho_target,
            tolerance=boundary_tolerance,
        )
    ):
        raise ValueError(
            "La trayectoria hacia rho_target alcanza o cruza "
            "la frontera D(rho)=0."
        )

    F_target = model.F(
        rho_target
    )

    D_target = model.D(
        rho_target
    )

    if F_target <= 0.0:
        raise ValueError(
            "rho_target está fuera del dominio físico F(rho)>0."
        )

    if math.isclose(
        D_target,
        0.0,
        abs_tol=boundary_tolerance,
        rel_tol=0.0,
    ):
        raise ValueError(
            "rho_target está demasiado cerca de D(rho)=0."
        )

    # ---------------------------------------------------------
    # Cuadratura temporal
    # ---------------------------------------------------------

    u0 = math.log(rho0)
    u_target = math.log(rho_target)

    def dt_du(
        log_rho: float,
    ) -> float:
        rho = math.exp(
            log_rho
        )

        F_value = model.F(rho)

        if F_value <= 0.0:
            raise ValueError(
                "La cuadratura salió del dominio F(rho)>0."
            )

        D_value = model.D(rho)

        H_value = model.signed_hubble(
            rho,
            branch=1.0,
        )

        if H_value <= 0.0:
            raise ValueError(
                "La cuadratura alcanzó H=0."
            )

        return float(
            -D_value
            / (
                3.0
                * (1.0 + omega)
                * H_value
            )
        )

    delta_t, quadrature_error = quad(
        dt_du,
        u0,
        u_target,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=integration_limit,
    )

    target_time = (
        float(params.t0)
        + float(delta_t)
    )

    if (
        require_future
        and target_time
        < params.t0 - 1e-10
    ):
        raise RuntimeError(
            "La cuadratura produjo un tiempo anterior a t0."
        )

    # ---------------------------------------------------------
    # Reconstrucción del resto de las variables
    # ---------------------------------------------------------

    log_a_value = float(
        model.log_a_of_rho(
            rho_target
        )
    )

    maximum_log_float = float(
        math.log(
            np.finfo(float).max
        )
    )

    if log_a_value <= maximum_log_float:
        a_value = float(
            math.exp(log_a_value)
        )
    else:
        a_value = float("inf")

    H_value = float(
        model.signed_hubble(
            rho_target,
            branch=1.0,
        )
    )

    Hdot_value = float(
        -0.5
        * (1.0 + omega)
        * rho_target
    )

    rho_dot_value = float(
        -3.0
        * H_value
        * (1.0 + omega)
        * rho_target
        / D_target
    )

    Hddot_value = float(
        -0.5
        * (1.0 + omega)
        * rho_dot_value
    )

    return {
        "t": target_time,
        "rho": float(rho_target),
        "log_a": log_a_value,
        "a": a_value,
        "F": float(F_target),
        "D": float(D_target),
        "H": H_value,
        "Hdot": Hdot_value,
        "rho_dot": rho_dot_value,
        "Hddot": Hddot_value,
        "quadrature_error": float(
            quadrature_error
        ),
    }

def scan_density_targets(
    params: BarotropicPowerDiffusionParams,
    rho_targets: Iterable[float],
    *,
    require_future: bool = True,
    strict: bool = False,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    integration_limit: int = 500,
) -> pd.DataFrame:
    """
    Evalúa una secuencia de densidades objetivo.

    Si strict=False, los objetivos no alcanzables se registran
    en el DataFrame sin interrumpir el barrido.
    """

    targets = list(
        rho_targets
    )

    if not targets:
        raise ValueError(
            "rho_targets no puede estar vacío."
        )

    rows = []

    for index, rho_target in enumerate(
        targets
    ):
        try:
            result = state_at_density(
                params=params,
                rho_target=float(
                    rho_target
                ),
                require_future=require_future,
                epsabs=epsabs,
                epsrel=epsrel,
                integration_limit=(
                    integration_limit
                ),
            )

            row = {
                "target_index": index,
                "success": True,
                "error": None,
                **result,
            }

        except (
            ValueError,
            RuntimeError,
            OverflowError,
            FloatingPointError,
        ) as error:
            if strict:
                raise

            row = {
                "target_index": index,
                "success": False,
                "error": str(error),
                "t": np.nan,
                "rho": float(
                    rho_target
                ),
                "log_a": np.nan,
                "a": np.nan,
                "F": np.nan,
                "D": np.nan,
                "H": np.nan,
                "Hdot": np.nan,
                "rho_dot": np.nan,
                "Hddot": np.nan,
                "quadrature_error": np.nan,
            }

        rows.append(row)

    result = pd.DataFrame(
        rows
    )

    result.attrs.update(
        {
            "omega": float(
                params.omega
            ),
            "alpha": float(
                params.alpha
            ),
            "beta": float(
                params.beta
            ),
            "rho0": float(
                params.rho0
            ),
            "a0": float(
                params.a0
            ),
            "t0": float(
                params.t0
            ),
            "future_density_direction": (
                initial_density_direction(
                    params
                )
            ),
            "critical_densities": (
                compute_critical_densities(
                    params
                ).as_dict()
            ),
        }
    )

    return result

def _successful_scan_rows(
    scan: pd.DataFrame,
) -> pd.DataFrame:
    """
    Selecciona las filas exitosas de un barrido y conserva
    su orden físico.
    """

    if not isinstance(
        scan,
        pd.DataFrame,
    ):
        raise TypeError(
            "scan debe ser un pandas.DataFrame."
        )

    if scan.empty:
        raise ValueError(
            "scan no puede estar vacío."
        )

    result = scan.copy()

    if "success" in result.columns:
        result = result[
            result["success"]
        ].copy()

    if "target_index" in result.columns:
        result = result.sort_values(
            "target_index"
        )

    if len(result) < 3:
        raise ValueError(
            "Se requieren al menos tres puntos "
            "exitosos para estudiar convergencia."
        )

    return result.reset_index(
        drop=True
    )

def _relative_spread(
    values,
) -> float:
    """
    Dispersión relativa de una secuencia finita.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return np.nan

    scale = max(
        float(
            np.max(
                np.abs(array)
            )
        ),
        1e-300,
    )

    return float(
        (
            np.max(array)
            - np.min(array)
        )
        / scale
    )


def _monotonic_fraction(
    values,
    direction: str,
) -> float:
    """
    Fracción de pasos compatibles con una tendencia
    creciente o decreciente.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    differences = np.diff(
        array
    )

    if differences.size == 0:
        return 0.0

    if direction == "increasing":
        return float(
            np.mean(
                differences >= 0.0
            )
        )

    if direction == "decreasing":
        return float(
            np.mean(
                differences <= 0.0
            )
        )

    raise ValueError(
        "direction debe ser 'increasing' "
        "o 'decreasing'."
    )

def _tail_log_slope(
    values,
    tail_points: int = 4,
) -> float:
    """
    Calcula la pendiente de log10(|y|) respecto al índice
    en el tramo final del barrido.

    Una pendiente positiva indica crecimiento sistemático.
    Una pendiente negativa indica decrecimiento.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(array)
        & (
            np.abs(array)
            > 0.0
        )
    )

    array = np.abs(
        array[valid]
    )

    if array.size < 3:
        return np.nan

    tail_size = min(
        max(
            3,
            tail_points,
        ),
        array.size,
    )

    tail = array[
        -tail_size:
    ]

    x_values = np.arange(
        tail_size,
        dtype=float,
    )

    y_values = np.log10(
        np.maximum(
            tail,
            1e-300,
        )
    )

    slope, _ = np.polyfit(
        x_values,
        y_values,
        1,
    )

    return float(slope)

def infer_time_limit_from_scan(
    scan: pd.DataFrame,
    *,
    finite_ratio_threshold: float = 0.85,
    infinite_ratio_threshold: float = 0.95,
) -> dict[str, Any]:
    """
    Determina si los tiempos requeridos para alcanzar
    densidades sucesivas convergen.

    Sean

        Delta t_n = |t_n - t_(n-1)|.

    Si Delta t_n decrece aproximadamente de forma
    geométrica, la suma puede converger a un tiempo finito.

    Si los incrementos permanecen constantes o crecen,
    el límite temporal es infinito.
    """

    successful = _successful_scan_rows(
        scan
    )

    t_values = successful[
        "t"
    ].to_numpy(
        dtype=float
    )

    increments = np.abs(
        np.diff(t_values)
    )

    increments = increments[
        np.isfinite(increments)
        & (
            increments > 0.0
        )
    ]

    if increments.size < 2:
        return {
            "time_limit": "unknown",
            "estimated_limit_time": np.nan,
            "time_increment_ratio": np.nan,
            "time_convergence_stable": False,
        }

    ratios = (
        increments[1:]
        / np.maximum(
            increments[:-1],
            1e-300,
        )
    )

    tail_size = min(
        3,
        ratios.size,
    )

    tail_ratio = float(
        np.median(
            ratios[-tail_size:]
        )
    )

    last_smaller_than_first = bool(
        increments[-1]
        < 0.7 * increments[0]
    )

    nondecreasing_fraction = (
        _monotonic_fraction(
            increments,
            "increasing",
        )
    )

    if (
        tail_ratio
        < finite_ratio_threshold
        and last_smaller_than_first
    ):
        time_limit = "finite"

        geometric_ratio = float(
            np.clip(
                tail_ratio,
                0.0,
                0.999999,
            )
        )

        estimated_limit_time = float(
            t_values[-1]
            + (
                increments[-1]
                * geometric_ratio
                / (
                    1.0
                    - geometric_ratio
                )
            )
        )

        stable = bool(
            ratios.size >= 2
            and np.max(
                ratios[-tail_size:]
            )
            < 0.95
        )

    elif (
        tail_ratio
        >= infinite_ratio_threshold
        or nondecreasing_fraction
        >= 0.75
    ):
        time_limit = "infinite"
        estimated_limit_time = float(
            "inf"
        )
        stable = True

    else:
        time_limit = "unknown"
        estimated_limit_time = np.nan
        stable = False

    return {
        "time_limit": time_limit,
        "estimated_limit_time": (
            estimated_limit_time
        ),
        "time_increment_ratio": (
            tail_ratio
        ),
        "time_convergence_stable": (
            stable
        ),
    }

def infer_series_limit(
    values,
    *,
    zero_ratio_threshold: float = 1e-6,
    divergence_ratio_threshold: float = 1e3,
    slope_threshold: float = 0.20,
    constant_spread_threshold: float = 1e-3,
) -> dict[str, Any]:
    """
    Infiere conservadoramente si una serie:

    - tiende a cero;
    - diverge;
    - tiende a una constante;
    - todavía no puede determinarse.

    La función no utiliza la clasificación esperada del
    modelo.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size < 3:
        return {
            "limit": "unknown",
            "ratio": np.nan,
            "log_slope": np.nan,
        }

    magnitudes = np.abs(
        array
    )

    initial_scale = max(
        float(
            magnitudes[0]
        ),
        1e-300,
    )

    final_scale = float(
        magnitudes[-1]
    )

    ratio = (
        final_scale
        / initial_scale
    )

    slope = _tail_log_slope(
        magnitudes
    )

    increasing_fraction = (
        _monotonic_fraction(
            magnitudes,
            "increasing",
        )
    )

    decreasing_fraction = (
        _monotonic_fraction(
            magnitudes,
            "decreasing",
        )
    )

    tail_size = min(
        4,
        magnitudes.size,
    )

    tail_spread = _relative_spread(
        magnitudes[-tail_size:]
    )

    if (
        ratio
        <= zero_ratio_threshold
        and decreasing_fraction
        >= 0.75
        and slope
        < -slope_threshold
    ):
        limit = "zero"

    elif (
        ratio
        >= divergence_ratio_threshold
        and increasing_fraction
        >= 0.75
        and slope
        > slope_threshold
    ):
        limit = "diverges"

    elif (
        tail_spread
        <= constant_spread_threshold
        and abs(slope) < 0.05
    ):
        limit = "constant"

    else:
        limit = "unknown"

    return {
        "limit": limit,
        "ratio": float(ratio),
        "log_slope": float(slope),
        "tail_relative_spread": float(
            tail_spread
        ),
        "increasing_fraction": float(
            increasing_fraction
        ),
        "decreasing_fraction": float(
            decreasing_fraction
        ),
    }

def infer_scale_factor_limit(
    log_a_values,
) -> dict[str, Any]:
    """
    Infiere el comportamiento de a a partir de ln(a).

    Esto evita overflow cuando a alcanza valores enormes.
    """

    log_a = np.asarray(
        log_a_values,
        dtype=float,
    )

    log_a = log_a[
        np.isfinite(log_a)
    ]

    if log_a.size < 3:
        return {
            "limit": "unknown",
            "step_ratio": np.nan,
        }

    changes = np.diff(
        log_a
    )

    absolute_steps = np.abs(
        changes
    )

    nonzero_steps = absolute_steps[
        absolute_steps > 1e-14
    ]

    total_change = float(
        log_a[-1]
        - log_a[0]
    )

    if nonzero_steps.size < 2:
        return {
            "limit": "constant",
            "step_ratio": 0.0,
            "total_log_change": (
                total_change
            ),
        }

    step_ratios = (
        nonzero_steps[1:]
        / np.maximum(
            nonzero_steps[:-1],
            1e-300,
        )
    )

    tail_size = min(
        3,
        step_ratios.size,
    )

    tail_ratio = float(
        np.median(
            step_ratios[-tail_size:]
        )
    )

    increasing_fraction = (
        _monotonic_fraction(
            log_a,
            "increasing",
        )
    )

    decreasing_fraction = (
        _monotonic_fraction(
            log_a,
            "decreasing",
        )
    )

    if (
        total_change > 10.0
        and increasing_fraction >= 0.75
        and (
            tail_ratio >= 0.85
            or nonzero_steps[-1] > 0.5
        )
    ):
        limit = "diverges"

    elif (
        total_change < -10.0
        and decreasing_fraction >= 0.75
        and (
            tail_ratio >= 0.85
            or nonzero_steps[-1] > 0.5
        )
    ):
        limit = "zero"

    elif (
        tail_ratio < 0.80
        and nonzero_steps[-1]
        < 0.2 * nonzero_steps[0]
    ):
        limit = "finite"

    else:
        limit = "unknown"

    return {
        "limit": limit,
        "step_ratio": tail_ratio,
        "total_log_change": (
            total_change
        ),
        "last_log_step": float(
            nonzero_steps[-1]
        ),
    }

def infer_limit_evidence(
    scan: pd.DataFrame,
) -> dict[str, Any]:
    """
    Convierte un barrido de densidades objetivo en evidencia
    compatible con classify_barotropic_power_diffusion.

    No asigna directamente una singularidad.
    Solo describe límites observados y convergencia.
    """

    successful = _successful_scan_rows(
        scan
    )

    rho_targets = successful[
        "rho"
    ].to_numpy(
        dtype=float
    )

    if (
        rho_targets[-1]
        > rho_targets[0]
    ):
        rho_limit = "diverges"
        target_direction = (
            "growth_to_infinity"
        )

    elif (
        rho_targets[-1]
        < rho_targets[0]
    ):
        rho_limit = "zero"
        target_direction = (
            "decay_to_zero"
        )

    else:
        rho_limit = "constant"
        target_direction = "constant"

    time_diagnostic = (
        infer_time_limit_from_scan(
            successful
        )
    )

    a_diagnostic = (
        infer_scale_factor_limit(
            successful[
                "log_a"
            ].to_numpy(
                dtype=float
            )
        )
    )

    H_diagnostic = infer_series_limit(
        successful[
            "H"
        ].to_numpy(
            dtype=float
        )
    )

    Hdot_diagnostic = (
        infer_series_limit(
            successful[
                "Hdot"
            ].to_numpy(
                dtype=float
            )
        )
    )

    rho_dot_diagnostic = (
        infer_series_limit(
            successful[
                "rho_dot"
            ].to_numpy(
                dtype=float
            )
        )
    )

    Hddot_diagnostic = (
        infer_series_limit(
            successful[
                "Hddot"
            ].to_numpy(
                dtype=float
            )
        )
    )

    diagnostic_limits = [
        a_diagnostic["limit"],
        H_diagnostic["limit"],
        Hdot_diagnostic["limit"],
        rho_dot_diagnostic["limit"],
        Hddot_diagnostic["limit"],
    ]

    determined_fraction = float(
        np.mean(
            [
                value != "unknown"
                for value
                in diagnostic_limits
            ]
        )
    )

    stable = bool(
        time_diagnostic[
            "time_convergence_stable"
        ]
        and determined_fraction >= 0.6
    )

    confidence = float(
        np.clip(
            0.45
            + 0.25 * float(stable)
            + 0.25 * determined_fraction,
            0.0,
            0.95,
        )
    )

    return {
        "time_limit": (
            time_diagnostic[
                "time_limit"
            ]
        ),
        "rho_limit": rho_limit,
        "a_limit": (
            a_diagnostic[
                "limit"
            ]
        ),
        "H_limit": (
            H_diagnostic[
                "limit"
            ]
        ),
        "Hdot_limit": (
            Hdot_diagnostic[
                "limit"
            ]
        ),
        "rho_dot_limit": (
            rho_dot_diagnostic[
                "limit"
            ]
        ),
        "Hddot_limit": (
            Hddot_diagnostic[
                "limit"
            ]
        ),
        "stable_under_threshold_changes": (
            stable
        ),
        "confidence": confidence,
        "target_direction": (
            target_direction
        ),
        "estimated_limit_time": (
            time_diagnostic[
                "estimated_limit_time"
            ]
        ),
        "diagnostics": {
            "time": (
                time_diagnostic
            ),
            "a": a_diagnostic,
            "H": H_diagnostic,
            "Hdot": (
                Hdot_diagnostic
            ),
            "rho_dot": (
                rho_dot_diagnostic
            ),
            "Hddot": (
                Hddot_diagnostic
            ),
            "determined_fraction": (
                determined_fraction
            ),
        },
    }

def scan_boundary_tolerances(
    params: BarotropicPowerDiffusionParams,
    *,
    t_end: float,
    boundary_tolerances: Iterable[float] = (
        1e-2,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
        1e-7,
        1e-8,
    ),
    n_points: int = 2000,
    rho_min: float = 1e-14,
    rho_max: float = 1e14,
    a_max: float = 1e100,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Estudia la convergencia hacia F(rho)=0 o D(rho)=0.

    Para cada valor de boundary_tol se ejecuta nuevamente
    solve_background_t y se almacena el estado final.

    Una frontera física real debe presentar convergencia de:

        t_final,
        rho_final,
        a_final,
        H_final,
        Hdot_final,

    cuando boundary_tol -> 0.

    En la frontera D(rho)=0 también se estudian:

        rho_dot,
        Hddot.

    Parameters
    ----------
    params:
        Parámetros del modelo.

    t_end:
        Tiempo máximo de integración. Debe ser suficientemente
        grande para que el integrador alcance la frontera.

    boundary_tolerances:
        Secuencia positiva y decreciente de tolerancias.

    strict:
        Si es True, cualquier error detiene el barrido.
        Si es False, el error queda registrado en el DataFrame.

    Returns
    -------
    pandas.DataFrame
        Tabla con el estado final para cada tolerancia.
    """

    if not isinstance(
        params,
        BarotropicPowerDiffusionParams,
    ):
        raise TypeError(
            "params debe ser BarotropicPowerDiffusionParams."
        )

    params.validate()

    if not math.isfinite(t_end):
        raise ValueError(
            "t_end debe ser finito."
        )

    if t_end <= params.t0:
        raise ValueError(
            "t_end debe ser mayor que t0."
        )

    tolerances = np.asarray(
        list(boundary_tolerances),
        dtype=float,
    )

    if tolerances.size < 3:
        raise ValueError(
            "Se requieren al menos tres tolerancias."
        )

    if not np.isfinite(
        tolerances
    ).all():
        raise ValueError(
            "Todas las tolerancias deben ser finitas."
        )

    if np.any(
        tolerances <= 0.0
    ):
        raise ValueError(
            "Todas las tolerancias deben ser positivas."
        )

    # Se ordenan desde la tolerancia más grande hasta
    # la más pequeña: acercamiento progresivo a la frontera.
    tolerances = np.sort(
        np.unique(tolerances)
    )[::-1]

    model = BarotropicPowerDiffusionModel(
        params
    )

    rows = []

    for index, boundary_tol in enumerate(
        tolerances
    ):
        try:
            background = model.solve_background_t(
                t_end=t_end,
                n_points=n_points,
                rho_min=rho_min,
                rho_max=rho_max,
                a_max=a_max,
                boundary_tol=float(
                    boundary_tol
                ),
                method=method,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )

            final = background.iloc[-1]

            rho_final = float(
                final["rho"]
            )

            a_final = float(
                final["a"]
            )

            H_final = float(
                final["H"]
            )

            Hdot_final = float(
                final["Hdot"]
            )

            F_final = float(
                model.F(
                    rho_final
                )
            )

            D_final = float(
                model.D(
                    rho_final
                )
            )

            if abs(D_final) > 0.0:
                rho_dot_final = float(
                    -3.0
                    * H_final
                    * (
                        1.0
                        + params.omega
                    )
                    * rho_final
                    / D_final
                )
            else:
                rho_dot_final = float(
                    "nan"
                )

            Hddot_final = float(
                -0.5
                * (
                    1.0
                    + params.omega
                )
                * rho_dot_final
            )

            termination_event = (
                background.attrs.get(
                    "termination_event"
                )
            )

            row = {
                "tolerance_index": index,
                "boundary_tol": float(
                    boundary_tol
                ),
                "success": True,
                "error": None,
                "termination_event": (
                    termination_event
                ),
                "t_final": float(
                    final["t"]
                ),
                "rho_final": rho_final,
                "log_a_final": float(
                    math.log(a_final)
                ),
                "a_final": a_final,
                "H_final": H_final,
                "Hdot_final": Hdot_final,
                "F_final": F_final,
                "D_final": D_final,
                "rho_dot_final": (
                    rho_dot_final
                ),
                "Hddot_final": (
                    Hddot_final
                ),
                "solver_success": bool(
                    background.attrs.get(
                        "solver_success",
                        False,
                    )
                ),
                "reached_t_end": bool(
                    background.attrs.get(
                        "reached_t_end",
                        False,
                    )
                ),
            }

        except (
            ValueError,
            RuntimeError,
            OverflowError,
            FloatingPointError,
        ) as error:
            if strict:
                raise

            row = {
                "tolerance_index": index,
                "boundary_tol": float(
                    boundary_tol
                ),
                "success": False,
                "error": str(error),
                "termination_event": None,
                "t_final": np.nan,
                "rho_final": np.nan,
                "log_a_final": np.nan,
                "a_final": np.nan,
                "H_final": np.nan,
                "Hdot_final": np.nan,
                "F_final": np.nan,
                "D_final": np.nan,
                "rho_dot_final": np.nan,
                "Hddot_final": np.nan,
                "solver_success": False,
                "reached_t_end": False,
            }

        rows.append(row)

    result = pd.DataFrame(
        rows
    )

    result.attrs.update(
        {
            "omega": float(
                params.omega
            ),
            "alpha": float(
                params.alpha
            ),
            "beta": float(
                params.beta
            ),
            "rho0": float(
                params.rho0
            ),
            "a0": float(
                params.a0
            ),
            "t0": float(
                params.t0
            ),
            "critical_densities": (
                compute_critical_densities(
                    params
                ).as_dict()
            ),
        }
    )

    return result

def identify_boundary_type(
    boundary_scan: pd.DataFrame,
) -> str:
    """
    Identifica la frontera dominante de un barrido.

    Returns
    -------
    str
        "F", "D", "mixed", "none" o "unknown".
    """

    if not isinstance(
        boundary_scan,
        pd.DataFrame,
    ):
        raise TypeError(
            "boundary_scan debe ser un pandas.DataFrame."
        )

    if boundary_scan.empty:
        return "unknown"

    successful = boundary_scan.copy()

    if "success" in successful.columns:
        successful = successful[
            successful["success"]
        ]

    if successful.empty:
        return "unknown"

    events = set(
        successful[
            "termination_event"
        ].dropna()
    )

    F_event = (
        "effective_density_boundary"
        in events
    )

    D_event = (
        "rho_dot_denominator_boundary"
        in events
    )

    if F_event and D_event:
        return "mixed"

    if F_event:
        return "F"

    if D_event:
        return "D"

    if not events:
        return "none"

    return "unknown"

def infer_finite_endpoint(
    values,
    *,
    relative_tolerance: float = 1e-3,
    tail_points: int = 3,
) -> dict[str, Any]:
    """
    Determina si una secuencia converge hacia un valor finito.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size < 3:
        return {
            "converges": False,
            "limit": np.nan,
            "relative_spread": np.nan,
        }

    tail_size = min(
        max(
            3,
            tail_points,
        ),
        array.size,
    )

    tail = array[-tail_size:]

    scale = max(
        float(
            np.max(
                np.abs(tail)
            )
        ),
        1.0,
    )

    relative_spread = float(
        (
            np.max(tail)
            - np.min(tail)
        )
        / scale
    )

    converges = bool(
        relative_spread
        <= relative_tolerance
    )

    return {
        "converges": converges,
        "limit": float(
            np.mean(tail)
        ),
        "relative_spread": (
            relative_spread
        ),
    }

def infer_boundary_divergence(
    boundary_tolerances,
    values,
    *,
    minimum_growth_factor: float = 10.0,
    minimum_log_slope: float = 0.25,
) -> dict[str, Any]:
    """
    Determina si |value| diverge cuando boundary_tol -> 0.

    Se ajusta:

        log10(|value|)
        =
        m log10(1 / boundary_tol) + b.

    Si m > 0, la cantidad crece al acercarse a la frontera.
    """

    tolerances = np.asarray(
        boundary_tolerances,
        dtype=float,
    )

    array = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(tolerances)
        & np.isfinite(array)
        & (
            tolerances > 0.0
        )
        & (
            np.abs(array) > 0.0
        )
    )

    tolerances = tolerances[
        valid
    ]

    magnitudes = np.abs(
        array[valid]
    )

    if magnitudes.size < 3:
        return {
            "diverges": False,
            "log_slope": np.nan,
            "growth_factor": np.nan,
        }

    x_values = np.log10(
        1.0
        / tolerances
    )

    y_values = np.log10(
        np.maximum(
            magnitudes,
            1e-300,
        )
    )

    slope, intercept = np.polyfit(
        x_values,
        y_values,
        1,
    )

    growth_factor = float(
        magnitudes[-1]
        / max(
            magnitudes[0],
            1e-300,
        )
    )

    increasing_fraction = (
        _monotonic_fraction(
            magnitudes,
            "increasing",
        )
    )

    diverges = bool(
        slope >= minimum_log_slope
        and growth_factor
        >= minimum_growth_factor
        and increasing_fraction >= 0.75
    )

    return {
        "diverges": diverges,
        "log_slope": float(slope),
        "intercept": float(intercept),
        "growth_factor": growth_factor,
        "increasing_fraction": float(
            increasing_fraction
        ),
    }

def infer_boundary_zero_limit(
    boundary_tolerances,
    values,
    *,
    maximum_decay_ratio: float = 0.1,
    maximum_log_slope: float = -0.20,
) -> dict[str, Any]:
    """
    Determina si una cantidad tiende a cero cuando
    boundary_tol -> 0.
    """

    tolerances = np.asarray(
        boundary_tolerances,
        dtype=float,
    )

    array = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(tolerances)
        & np.isfinite(array)
        & (
            tolerances > 0.0
        )
    )

    tolerances = tolerances[
        valid
    ]

    magnitudes = np.abs(
        array[valid]
    )

    if magnitudes.size < 3:
        return {
            "tends_to_zero": False,
            "log_slope": np.nan,
            "decay_ratio": np.nan,
        }

    safe_magnitudes = np.maximum(
        magnitudes,
        1e-300,
    )

    x_values = np.log10(
        1.0
        / tolerances
    )

    y_values = np.log10(
        safe_magnitudes
    )

    slope, intercept = np.polyfit(
        x_values,
        y_values,
        1,
    )

    decay_ratio = float(
        safe_magnitudes[-1]
        / safe_magnitudes[0]
    )

    decreasing_fraction = (
        _monotonic_fraction(
            magnitudes,
            "decreasing",
        )
    )

    tends_to_zero = bool(
        slope <= maximum_log_slope
        and decay_ratio
        <= maximum_decay_ratio
        and decreasing_fraction >= 0.75
    )

    return {
        "tends_to_zero": tends_to_zero,
        "log_slope": float(slope),
        "intercept": float(intercept),
        "decay_ratio": decay_ratio,
        "decreasing_fraction": float(
            decreasing_fraction
        ),
    }

def infer_boundary_evidence(
    boundary_scan: pd.DataFrame,
) -> dict[str, Any]:
    """
    Construye evidencia física a partir de un barrido
    de boundary_tol.

    La función distingue entre:

    - frontera F(rho)=0;
    - frontera D(rho)=0;
    - barrido inconcluso.
    """

    if not isinstance(
        boundary_scan,
        pd.DataFrame,
    ):
        raise TypeError(
            "boundary_scan debe ser un pandas.DataFrame."
        )

    successful = boundary_scan.copy()

    if "success" in successful.columns:
        successful = successful[
            successful["success"]
        ].copy()

    if len(successful) < 3:
        return {
            "time_limit": "unknown",
            "rho_limit": "unknown",
            "a_limit": "unknown",
            "H_limit": "unknown",
            "Hdot_limit": "unknown",
            "rho_dot_limit": "unknown",
            "Hddot_limit": "unknown",
            "stable_under_tolerance_changes": False,
            "confidence": 0.2,
            "boundary_type": "unknown",
            "diagnostics": {
                "reason": (
                    "Menos de tres integraciones exitosas."
                )
            },
        }

    successful = successful.sort_values(
        "boundary_tol",
        ascending=False,
    ).reset_index(
        drop=True
    )

    boundary_type = identify_boundary_type(
        successful
    )

    tolerances = successful[
        "boundary_tol"
    ].to_numpy(
        dtype=float
    )

    time_diagnostic = infer_finite_endpoint(
        successful[
            "t_final"
        ].to_numpy(
            dtype=float
        )
    )

    rho_diagnostic = infer_finite_endpoint(
        successful[
            "rho_final"
        ].to_numpy(
            dtype=float
        )
    )

    log_a_diagnostic = infer_finite_endpoint(
        successful[
            "log_a_final"
        ].to_numpy(
            dtype=float
        )
    )

    H_finite_diagnostic = infer_finite_endpoint(
        successful[
            "H_final"
        ].to_numpy(
            dtype=float
        )
    )

    Hdot_diagnostic = infer_finite_endpoint(
        successful[
            "Hdot_final"
        ].to_numpy(
            dtype=float
        )
    )

    H_zero_diagnostic = (
        infer_boundary_zero_limit(
            tolerances,
            successful[
                "H_final"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    rho_dot_divergence = (
        infer_boundary_divergence(
            tolerances,
            successful[
                "rho_dot_final"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    Hddot_divergence = (
        infer_boundary_divergence(
            tolerances,
            successful[
                "Hddot_final"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    rho_dot_zero = (
        infer_boundary_zero_limit(
            tolerances,
            successful[
                "rho_dot_final"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    Hddot_zero = (
        infer_boundary_zero_limit(
            tolerances,
            successful[
                "Hddot_final"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    # Las fronteras F y D ocurren en rho finita.
    rho_limit = (
        "finite"
        if rho_diagnostic["converges"]
        else "unknown"
    )

    a_limit = (
        "finite"
        if log_a_diagnostic["converges"]
        else "unknown"
    )

    Hdot_limit = (
        "finite"
        if Hdot_diagnostic["converges"]
        else "unknown"
    )

    time_limit = (
        "finite"
        if time_diagnostic["converges"]
        else "unknown"
    )

    if boundary_type == "F":
        if H_zero_diagnostic[
            "tends_to_zero"
        ]:
            H_limit = "zero"
        elif H_finite_diagnostic[
            "converges"
        ]:
            H_limit = "finite"
        else:
            H_limit = "unknown"

        rho_dot_limit = (
            "zero"
            if rho_dot_zero[
                "tends_to_zero"
            ]
            else "unknown"
        )

        Hddot_limit = (
            "zero"
            if Hddot_zero[
                "tends_to_zero"
            ]
            else "unknown"
        )

    elif boundary_type == "D":
        H_limit = (
            "finite"
            if H_finite_diagnostic[
                "converges"
            ]
            else "unknown"
        )

        rho_dot_limit = (
            "diverges"
            if rho_dot_divergence[
                "diverges"
            ]
            else "unknown"
        )

        Hddot_limit = (
            "diverges"
            if Hddot_divergence[
                "diverges"
            ]
            else "unknown"
        )

    else:
        H_limit = "unknown"
        rho_dot_limit = "unknown"
        Hddot_limit = "unknown"

    principal_convergence = all(
        diagnostic["converges"]
        for diagnostic in [
            time_diagnostic,
            rho_diagnostic,
            log_a_diagnostic,
            Hdot_diagnostic,
        ]
    )

    if boundary_type == "F":
        derivative_behavior_confirmed = bool(
            H_zero_diagnostic[
                "tends_to_zero"
            ]
        )

    elif boundary_type == "D":
        derivative_behavior_confirmed = bool(
            rho_dot_divergence[
                "diverges"
            ]
            or Hddot_divergence[
                "diverges"
            ]
        )

    else:
        derivative_behavior_confirmed = False

    stable = bool(
        boundary_type in {
            "F",
            "D",
        }
        and principal_convergence
        and derivative_behavior_confirmed
    )

    confidence = float(
        np.clip(
            0.40
            + 0.35
            * float(
                principal_convergence
            )
            + 0.20
            * float(
                derivative_behavior_confirmed
            ),
            0.0,
            0.95,
        )
    )

    return {
        "time_limit": time_limit,
        "rho_limit": rho_limit,
        "a_limit": a_limit,
        "H_limit": H_limit,
        "Hdot_limit": Hdot_limit,
        "rho_dot_limit": (
            rho_dot_limit
        ),
        "Hddot_limit": (
            Hddot_limit
        ),
        "stable_under_tolerance_changes": (
            stable
        ),
        "confidence": confidence,
        "boundary_type": boundary_type,
        "estimated_limit_time": (
            time_diagnostic["limit"]
        ),
        "estimated_boundary_density": (
            rho_diagnostic["limit"]
        ),
        "estimated_log_a": (
            log_a_diagnostic["limit"]
        ),
        "diagnostics": {
            "time": time_diagnostic,
            "rho": rho_diagnostic,
            "log_a": (
                log_a_diagnostic
            ),
            "H_finite": (
                H_finite_diagnostic
            ),
            "H_zero": (
                H_zero_diagnostic
            ),
            "Hdot": Hdot_diagnostic,
            "rho_dot_divergence": (
                rho_dot_divergence
            ),
            "Hddot_divergence": (
                Hddot_divergence
            ),
            "rho_dot_zero": (
                rho_dot_zero
            ),
            "Hddot_zero": (
                Hddot_zero
            ),
        },
    }

def solve_signed_background_compatible(
    params: BarotropicPowerDiffusionParams,
    **solver_kwargs,
) -> pd.DataFrame:
    """
    Ejecuta solve_signed_background_t usando únicamente los
    argumentos admitidos por la versión actual del método.

    Esto evita errores si posteriormente cambia ligeramente
    la firma del solucionador.

    Los argumentos ignorados se almacenan en result.attrs.
    """

    if not isinstance(
        params,
        BarotropicPowerDiffusionParams,
    ):
        raise TypeError(
            "params debe ser BarotropicPowerDiffusionParams."
        )

    params.validate()

    model = BarotropicPowerDiffusionModel(
        params
    )

    if not hasattr(
        model,
        "solve_signed_background_t",
    ):
        raise AttributeError(
            "BarotropicPowerDiffusionModel no contiene "
            "solve_signed_background_t."
        )

    signed_solver = (
        model.solve_signed_background_t
    )

    signature = inspect.signature(
        signed_solver
    )

    supported_names = set(
        signature.parameters
    )

    supported_kwargs = {
        name: value
        for name, value in solver_kwargs.items()
        if name in supported_names
    }

    ignored_kwargs = {
        name: value
        for name, value in solver_kwargs.items()
        if name not in supported_names
    }

    if "t_end" not in supported_kwargs:
        raise ValueError(
            "Debe entregarse t_end para la integración signada."
        )

    result = signed_solver(
        **supported_kwargs
    )

    if not isinstance(
        result,
        pd.DataFrame,
    ):
        raise TypeError(
            "solve_signed_background_t debe devolver "
            "un pandas.DataFrame."
        )

    result.attrs[
        "analysis_supported_kwargs"
    ] = supported_kwargs

    result.attrs[
        "analysis_ignored_kwargs"
    ] = ignored_kwargs

    return result

def summarize_signed_background(
    signed_background: pd.DataFrame,
    *,
    zero_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """
    Resume una evolución donde H puede cambiar de signo.

    Detecta:

    - cruce de expansión a contracción;
    - tiempo aproximado del turnaround;
    - máximo del factor de escala;
    - estado final;
    - residuo de la restricción de Friedmann.
    """

    required_columns = {
        "t",
        "rho",
        "a",
        "H",
        "Hdot",
    }

    if not isinstance(
        signed_background,
        pd.DataFrame,
    ):
        raise TypeError(
            "signed_background debe ser un pandas.DataFrame."
        )

    missing_columns = (
        required_columns
        - set(signed_background.columns)
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas en signed_background: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if len(signed_background) < 3:
        raise ValueError(
            "Se requieren al menos tres puntos."
        )

    t_values = signed_background[
        "t"
    ].to_numpy(
        dtype=float
    )

    H_values = signed_background[
        "H"
    ].to_numpy(
        dtype=float
    )

    a_values = signed_background[
        "a"
    ].to_numpy(
        dtype=float
    )

    rho_values = signed_background[
        "rho"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        np.column_stack(
            [
                t_values,
                H_values,
                a_values,
                rho_values,
            ]
        )
    ).all():
        raise ValueError(
            "La solución signada contiene valores no finitos."
        )

    negative_indices = np.flatnonzero(
        H_values < -zero_tolerance
    )

    positive_indices = np.flatnonzero(
        H_values > zero_tolerance
    )

    turnaround_detected = bool(
        positive_indices.size > 0
        and negative_indices.size > 0
        and negative_indices[0]
        > positive_indices[0]
    )

    turnaround_time = np.nan
    turnaround_density = np.nan
    turnaround_scale_factor = np.nan

    if turnaround_detected:
        first_negative = int(
            negative_indices[0]
        )

        previous_positive_candidates = (
            positive_indices[
                positive_indices
                < first_negative
            ]
        )

        if (
            previous_positive_candidates.size
            > 0
        ):
            left_index = int(
                previous_positive_candidates[-1]
            )

            right_index = (
                first_negative
            )

            t_left = t_values[
                left_index
            ]

            t_right = t_values[
                right_index
            ]

            H_left = H_values[
                left_index
            ]

            H_right = H_values[
                right_index
            ]

            denominator = (
                H_right
                - H_left
            )

            if abs(denominator) > 0.0:
                fraction = (
                    -H_left
                    / denominator
                )

                fraction = float(
                    np.clip(
                        fraction,
                        0.0,
                        1.0,
                    )
                )

                turnaround_time = float(
                    t_left
                    + fraction
                    * (
                        t_right
                        - t_left
                    )
                )

                turnaround_density = float(
                    rho_values[left_index]
                    + fraction
                    * (
                        rho_values[right_index]
                        - rho_values[left_index]
                    )
                )

                turnaround_scale_factor = float(
                    a_values[left_index]
                    + fraction
                    * (
                        a_values[right_index]
                        - a_values[left_index]
                    )
                )

    maximum_a_index = int(
        np.argmax(
            a_values
        )
    )

    final = signed_background.iloc[-1]

    return {
        "turnaround_detected": (
            turnaround_detected
        ),
        "turnaround_time": float(
            turnaround_time
        ),
        "turnaround_density": float(
            turnaround_density
        ),
        "turnaround_scale_factor": float(
            turnaround_scale_factor
        ),
        "maximum_a": float(
            a_values[
                maximum_a_index
            ]
        ),
        "time_at_maximum_a": float(
            t_values[
                maximum_a_index
            ]
        ),
        "initial_H": float(
            H_values[0]
        ),
        "final_H": float(
            final["H"]
        ),
        "final_rho": float(
            final["rho"]
        ),
        "final_a": float(
            final["a"]
        ),
        "final_Hdot": float(
            final["Hdot"]
        ),
        "contracting_at_end": bool(
            final["H"]
            < -zero_tolerance
        ),
        "termination_event": (
            signed_background.attrs.get(
                "termination_event"
            )
        ),
        "solver_success": bool(
            signed_background.attrs.get(
                "solver_success",
                False,
            )
        ),
        "reached_t_end": bool(
            signed_background.attrs.get(
                "reached_t_end",
                False,
            )
        ),
        "maximum_constraint_residual": (
            signed_background.attrs.get(
                "maximum_constraint_residual"
            )
        ),
    }

def scan_signed_collapse_thresholds(
    params: BarotropicPowerDiffusionParams,
    *,
    t_end: float,
    rho_max_values: Iterable[float] = (
        1e2,
        1e3,
        1e4,
        1e5,
        1e6,
        1e7,
    ),
    n_points: int = 4000,
    rho_min: float = 1e-12,
    a_min: float = 1e-12,
    a_max: float = 1e100,
    H_max: float = 1e100,
    boundary_tol: float = 1e-8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Estudia una posible fase contractiva aumentando rho_max.

    La función requiere que solve_signed_background_t admita
    rho_max como argumento.
    """

    thresholds = np.asarray(
        list(rho_max_values),
        dtype=float,
    )

    if thresholds.size < 3:
        raise ValueError(
            "Se requieren al menos tres valores de rho_max."
        )

    if not np.isfinite(
        thresholds
    ).all():
        raise ValueError(
            "rho_max_values debe contener valores finitos."
        )

    if np.any(
        thresholds <= params.rho0
    ):
        raise ValueError(
            "Cada rho_max debe ser mayor que rho0."
        )

    thresholds = np.sort(
        np.unique(
            thresholds
        )
    )

    test_model = (
        BarotropicPowerDiffusionModel(
            params
        )
    )

    signed_signature = inspect.signature(
        test_model.solve_signed_background_t
    )

    if (
        "rho_max"
        not in signed_signature.parameters
    ):
        raise ValueError(
            "La versión actual de solve_signed_background_t "
            "no admite rho_max. Debe agregarse ese evento "
            "antes de realizar este barrido."
        )

    rows = []

    for index, rho_max in enumerate(
        thresholds
    ):
        try:
            signed_background = (
                solve_signed_background_compatible(
                    params,
                    t_end=t_end,
                    n_points=n_points,
                    rho_min=rho_min,
                    rho_max=float(
                        rho_max
                    ),
                    a_min=a_min,
                    a_max=a_max,
                    H_max=H_max,
                    boundary_tol=(
                        boundary_tol
                    ),
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step,
                )
            )

            summary = (
                summarize_signed_background(
                    signed_background
                )
            )

            final = (
                signed_background.iloc[-1]
            )

            rho_final = float(
                final["rho"]
            )

            H_final = float(
                final["H"]
            )

            D_final = float(
                test_model.D(
                    rho_final
                )
            )

            if abs(D_final) > 0.0:
                rho_dot_final = float(
                    -3.0
                    * H_final
                    * (
                        1.0
                        + params.omega
                    )
                    * rho_final
                    / D_final
                )
            else:
                rho_dot_final = np.nan

            Hddot_final = float(
                -0.5
                * (
                    1.0
                    + params.omega
                )
                * rho_dot_final
            )

            row = {
                "threshold_index": index,
                "rho_max": float(
                    rho_max
                ),
                "success": True,
                "error": None,
                "turnaround_detected": (
                    summary[
                        "turnaround_detected"
                    ]
                ),
                "turnaround_time": (
                    summary[
                        "turnaround_time"
                    ]
                ),
                "turnaround_density": (
                    summary[
                        "turnaround_density"
                    ]
                ),
                "maximum_a": (
                    summary[
                        "maximum_a"
                    ]
                ),
                "termination_event": (
                    summary[
                        "termination_event"
                    ]
                ),
                "t_final": float(
                    final["t"]
                ),
                "rho_final": rho_final,
                "a_final": float(
                    final["a"]
                ),
                "H_final": H_final,
                "Hdot_final": float(
                    final["Hdot"]
                ),
                "rho_dot_final": (
                    rho_dot_final
                ),
                "Hddot_final": (
                    Hddot_final
                ),
                "contracting_at_end": (
                    summary[
                        "contracting_at_end"
                    ]
                ),
                "maximum_constraint_residual": (
                    summary[
                        "maximum_constraint_residual"
                    ]
                ),
            }

        except (
            ValueError,
            RuntimeError,
            OverflowError,
            FloatingPointError,
        ) as error:
            if strict:
                raise

            row = {
                "threshold_index": index,
                "rho_max": float(
                    rho_max
                ),
                "success": False,
                "error": str(error),
                "turnaround_detected": False,
                "turnaround_time": np.nan,
                "turnaround_density": np.nan,
                "maximum_a": np.nan,
                "termination_event": None,
                "t_final": np.nan,
                "rho_final": np.nan,
                "a_final": np.nan,
                "H_final": np.nan,
                "Hdot_final": np.nan,
                "rho_dot_final": np.nan,
                "Hddot_final": np.nan,
                "contracting_at_end": False,
                "maximum_constraint_residual": np.nan,
            }

        rows.append(row)

    result = pd.DataFrame(
        rows
    )

    result.attrs[
        "critical_densities"
    ] = compute_critical_densities(
        params
    ).as_dict()

    result.attrs["omega"] = float(
        params.omega
    )

    result.attrs["alpha"] = float(
        params.alpha
    )

    result.attrs["beta"] = float(
        params.beta
    )

    return result

def infer_signed_collapse_evidence(
    collapse_scan: pd.DataFrame,
) -> dict[str, Any]:
    """
    Infiere si un barrido signado es compatible con:

        turnaround -> Big Crunch.

    No basta con detectar H<0. También se exige convergencia
    temporal y tendencias consistentes de rho, a y H.
    """

    if not isinstance(
        collapse_scan,
        pd.DataFrame,
    ):
        raise TypeError(
            "collapse_scan debe ser un pandas.DataFrame."
        )

    successful = collapse_scan.copy()

    if "success" in successful.columns:
        successful = successful[
            successful["success"]
        ].copy()

    if len(successful) < 3:
        return {
            "time_limit": "unknown",
            "rho_limit": "unknown",
            "a_limit": "unknown",
            "H_limit": "unknown",
            "Hdot_limit": "unknown",
            "rho_dot_limit": "unknown",
            "Hddot_limit": "unknown",
            "stable_under_threshold_changes": False,
            "confidence": 0.2,
            "diagnostics": {
                "reason": (
                    "Menos de tres ejecuciones exitosas."
                )
            },
        }

    successful = successful.sort_values(
        "rho_max"
    ).reset_index(
        drop=True
    )

    temporary_time_scan = pd.DataFrame(
        {
            "target_index": np.arange(
                len(successful)
            ),
            "success": True,
            "t": successful[
                "t_final"
            ].to_numpy(
                dtype=float
            ),
        }
    )

    time_diagnostic = (
        infer_time_limit_from_scan(
            temporary_time_scan
        )
    )

    inverse_thresholds = (
        1.0
        / successful[
            "rho_max"
        ].to_numpy(
            dtype=float
        )
    )

    rho_diagnostic = (
        infer_boundary_divergence(
            inverse_thresholds,
            successful[
                "rho_final"
            ].to_numpy(
                dtype=float
            ),
            minimum_growth_factor=10.0,
            minimum_log_slope=0.25,
        )
    )

    H_diagnostic = (
        infer_boundary_divergence(
            inverse_thresholds,
            np.abs(
                successful[
                    "H_final"
                ].to_numpy(
                    dtype=float
                )
            ),
            minimum_growth_factor=5.0,
            minimum_log_slope=0.15,
        )
    )

    Hdot_diagnostic = (
        infer_boundary_divergence(
            inverse_thresholds,
            successful[
                "Hdot_final"
            ].to_numpy(
                dtype=float
            ),
            minimum_growth_factor=5.0,
            minimum_log_slope=0.15,
        )
    )

    rho_dot_diagnostic = (
        infer_boundary_divergence(
            inverse_thresholds,
            successful[
                "rho_dot_final"
            ].to_numpy(
                dtype=float
            ),
            minimum_growth_factor=5.0,
            minimum_log_slope=0.15,
        )
    )

    Hddot_diagnostic = (
        infer_boundary_divergence(
            inverse_thresholds,
            successful[
                "Hddot_final"
            ].to_numpy(
                dtype=float
            ),
            minimum_growth_factor=5.0,
            minimum_log_slope=0.15,
        )
    )

    a_zero_diagnostic = (
        infer_boundary_zero_limit(
            inverse_thresholds,
            successful[
                "a_final"
            ].to_numpy(
                dtype=float
            ),
            maximum_decay_ratio=0.5,
            maximum_log_slope=-0.05,
        )
    )

    turnaround_fraction = float(
        successful[
            "turnaround_detected"
        ].mean()
    )

    contraction_fraction = float(
        successful[
            "contracting_at_end"
        ].mean()
    )

    finite_time = (
        time_diagnostic[
            "time_limit"
        ]
        == "finite"
    )

    collapse_confirmed = bool(
        turnaround_fraction >= 0.8
        and contraction_fraction >= 0.8
        and finite_time
        and rho_diagnostic[
            "diverges"
        ]
        and H_diagnostic[
            "diverges"
        ]
        and a_zero_diagnostic[
            "tends_to_zero"
        ]
    )

    confidence = float(
        np.clip(
            0.35
            + 0.15
            * turnaround_fraction
            + 0.15
            * contraction_fraction
            + 0.15
            * float(finite_time)
            + 0.20
            * float(
                collapse_confirmed
            ),
            0.0,
            0.95,
        )
    )

    return {
        "time_limit": (
            "finite"
            if finite_time
            else time_diagnostic[
                "time_limit"
            ]
        ),
        "rho_limit": (
            "diverges"
            if rho_diagnostic[
                "diverges"
            ]
            else "unknown"
        ),
        "a_limit": (
            "zero"
            if a_zero_diagnostic[
                "tends_to_zero"
            ]
            else "unknown"
        ),
        "H_limit": (
            "diverges"
            if H_diagnostic[
                "diverges"
            ]
            else "unknown"
        ),
        "Hdot_limit": (
            "diverges"
            if Hdot_diagnostic[
                "diverges"
            ]
            else "unknown"
        ),
        "rho_dot_limit": (
            "diverges"
            if rho_dot_diagnostic[
                "diverges"
            ]
            else "unknown"
        ),
        "Hddot_limit": (
            "diverges"
            if Hddot_diagnostic[
                "diverges"
            ]
            else "unknown"
        ),
        "stable_under_threshold_changes": (
            collapse_confirmed
        ),
        "confidence": confidence,
        "turnaround_detected": bool(
            turnaround_fraction >= 0.8
        ),
        "contracting_branch": bool(
            contraction_fraction >= 0.8
        ),
        "estimated_limit_time": (
            time_diagnostic[
                "estimated_limit_time"
            ]
        ),
        "diagnostics": {
            "time": time_diagnostic,
            "rho": rho_diagnostic,
            "a_zero": (
                a_zero_diagnostic
            ),
            "H": H_diagnostic,
            "Hdot": (
                Hdot_diagnostic
            ),
            "rho_dot": (
                rho_dot_diagnostic
            ),
            "Hddot": (
                Hddot_diagnostic
            ),
            "turnaround_fraction": (
                turnaround_fraction
            ),
            "contraction_fraction": (
                contraction_fraction
            ),
            "collapse_confirmed": (
                collapse_confirmed
            ),
        },
    }

@dataclass
class BarotropicCaseAnalysis:
    """
    Resultado completo del análisis de un caso barotrópico.
    """

    case_id: str

    params: BarotropicPowerDiffusionParams

    critical_densities: CriticalDensities

    background: pd.DataFrame

    signed_background: pd.DataFrame | None

    density_scan: pd.DataFrame | None

    boundary_scan: pd.DataFrame | None

    collapse_scan: pd.DataFrame | None

    evidence: dict[str, Any]

    classification: Any

    diagnostics: dict[str, Any]

    def summary_dict(
        self,
    ) -> dict[str, Any]:
        """
        Resume el resultado en una fila apta para un DataFrame.
        """

        classification_dict = (
            self.classification.as_dict()
        )

        return {
            "classification_basis": (
                "analytic_with_numerical_diagnostics"
                if self.classification.evidence.get("analytic_basis")
                else "numerical"
            ),
            "case_id": self.case_id,
            "omega": float(
                self.params.omega
            ),
            "alpha": float(
                self.params.alpha
            ),
            "beta": float(
                self.params.beta
            ),
            "rho0": float(
                self.params.rho0
            ),
            "rho_F": (
                self.critical_densities.rho_F
            ),
            "rho_D": (
                self.critical_densities.rho_D
            ),
            "initial_density_direction": (
                self.diagnostics.get(
                    "initial_density_direction"
                )
            ),
            "analysis_route": (
                self.diagnostics.get(
                    "analysis_route"
                )
            ),
            "termination_event": (
                self.background.attrs.get(
                    "termination_event"
                )
            ),
            "signed_termination_event": (
                None
                if self.signed_background is None
                else self.signed_background.attrs.get(
                    "termination_event"
                )
            ),
            "standard_type": (
                classification_dict[
                    "standard_type"
                ]
            ),
            "behavior_label": (
                classification_dict[
                    "behavior_label"
                ]
            ),
            "status": (
                classification_dict[
                    "status"
                ]
            ),
            "is_singularity": (
                classification_dict[
                    "is_singularity"
                ]
            ),
            "finite_time": (
                classification_dict[
                    "finite_time"
                ]
            ),
            "confidence": (
                classification_dict[
                    "confidence"
                ]
            ),
            "time_limit": (
                self.evidence.get(
                    "time_limit"
                )
            ),
            "rho_limit": (
                self.evidence.get(
                    "rho_limit"
                )
            ),
            "a_limit": (
                self.evidence.get(
                    "a_limit"
                )
            ),
            "H_limit": (
                self.evidence.get(
                    "H_limit"
                )
            ),
            "Hdot_limit": (
                self.evidence.get(
                    "Hdot_limit"
                )
            ),
            "rho_dot_limit": (
                self.evidence.get(
                    "rho_dot_limit"
                )
            ),
            "Hddot_limit": (
                self.evidence.get(
                    "Hddot_limit"
                )
            ),
        }

def _first_future_boundary(
    params: BarotropicPowerDiffusionParams,
    critical: CriticalDensities,
    direction: int,
) -> tuple[str | None, float | None]:
    """
    Encuentra la primera frontera F=0 o D=0 situada en
    la dirección futura de rho.
    """

    if direction == 0:
        return None, None

    rho0 = float(
        params.rho0
    )

    candidates = []

    for boundary_type, density in [
        ("F", critical.rho_F),
        ("D", critical.rho_D),
    ]:
        if density is None:
            continue

        density = float(
            density
        )

        lies_in_future = bool(
            (
                direction > 0
                and density > rho0
            )
            or (
                direction < 0
                and density < rho0
            )
        )

        if not lies_in_future:
            continue

        logarithmic_distance = abs(
            math.log(
                density
                / rho0
            )
        )

        candidates.append(
            (
                logarithmic_distance,
                boundary_type,
                density,
            )
        )

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda item: item[0]
    )

    _, boundary_type, density = (
        candidates[0]
    )

    return (
        boundary_type,
        float(density),
    )

def _make_density_targets(
    params: BarotropicPowerDiffusionParams,
    direction: int,
    density_decades: int,
) -> list[float]:
    """
    Genera umbrales logarítmicos hacia rho=0 o rho=infinito.
    """

    if density_decades < 3:
        raise ValueError(
            "density_decades debe ser al menos 3."
        )

    rho0 = float(
        params.rho0
    )

    if direction > 0:
        return [
            rho0
            * 10.0**index
            for index in range(
                1,
                density_decades + 1,
            )
        ]

    if direction < 0:
        return [
            rho0
            * 10.0**(-index)
            for index in range(
                1,
                density_decades + 1,
            )
        ]

    return []

def analyze_barotropic_case(
    params: BarotropicPowerDiffusionParams,
    *,
    case_id: str = "unnamed_case",
    t_end: float = 20.0,
    n_points: int = 2000,
    max_step: float | None = 0.01,
    density_decades: int = 6,
    boundary_tolerances: Iterable[float] = (
        1e-2,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
    ),
    collapse_rho_max_values: Iterable[float] = (
        1e2,
        1e3,
        1e4,
        1e5,
        1e6,
    ),
    rho_min: float = 1e-12,
    rho_max: float = 1e12,
    a_min: float = 1e-12,
    a_max: float = 1e100,
    H_max: float = 1e100,
    boundary_tol: float = 1e-8,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> BarotropicCaseAnalysis:
    """
    Ejecuta automáticamente el análisis completo del modelo

        p = omega*rho,
        Q = alpha*rho**beta.

    La ruta se selecciona a partir de la evolución y las
    densidades críticas, no a partir de una clasificación
    esperada.
    """

    if not isinstance(
        params,
        BarotropicPowerDiffusionParams,
    ):
        raise TypeError(
            "params debe ser BarotropicPowerDiffusionParams."
        )

    params.validate()

    if not isinstance(
        case_id,
        str,
    ):
        raise TypeError(
            "case_id debe ser un string."
        )

    model = BarotropicPowerDiffusionModel(
        params
    )

    critical = compute_critical_densities(
        params
    )

    try:
        density_direction = (
            initial_density_direction(
                params
            )
        )
    except ValueError:
        density_direction = 0

    # Para la integración diagnóstica general no necesitamos
    # acercarnos extremadamente a D=0 o F=0. El análisis fino
    # será realizado posteriormente mediante un barrido de
    # tolerancias.
    diagnostic_boundary_tol = max(
    float(boundary_tol),
    1e-6,
    )

    if max_step is None:
        diagnostic_max_step = 0.001
    else:
        diagnostic_max_step = min(
        float(max_step),
        0.001,
    )

    try:
        background = model.solve_background_t(
            t_end=t_end,
            n_points=n_points,
            rho_min=rho_min,
            rho_max=rho_max,
            a_max=a_max,
            boundary_tol=diagnostic_boundary_tol,
            rtol=rtol,
            atol=atol,
            max_step=diagnostic_max_step,
        )

    except RuntimeError as error:
        error_message = str(error)

        if (
            "Required step size is less than spacing"
            not in error_message
        ):
            raise

        # Segundo intento más conservador. Se detiene todavía
        # antes de la frontera singular.
        diagnostic_boundary_tol = max(
            diagnostic_boundary_tol,
            1e-5,
        )

        background = model.solve_background_t(
            t_end=t_end,
            n_points=n_points,
            rho_min=rho_min,
            rho_max=rho_max,
            a_max=a_max,
            boundary_tol=diagnostic_boundary_tol,
            rtol=max(
                rtol,
                1e-9,
            ),
            atol=max(
                atol,
                1e-11,
            ),
            max_step=min(
                diagnostic_max_step,
                5e-4,
            ),
        )

    signed_background = None
    density_scan = None
    boundary_scan = None
    collapse_scan = None

    evidence: dict[str, Any] = {}

    diagnostics: dict[str, Any] = {
        "initial_density_direction": (
            density_direction
        ),
        "background_termination_event": (
            background.attrs.get(
                "termination_event"
            )
        ),
    }

    # Mayor resolución cerca de las fronteras.
    if max_step is None:
        fine_max_step = 0.001
    else:
        fine_max_step = min(
            float(max_step),
            0.001,
        )

    # ---------------------------------------------------------
    # 1. Caso estacionario: omega=-1
    # ---------------------------------------------------------

    if (
        math.isclose(
            float(params.omega),
            -1.0,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or density_direction == 0
    ):
        diagnostics[
            "analysis_route"
        ] = "stationary_density"

        evidence = {
            "stable_under_threshold_changes": True,
            "confidence": 0.95,
        }

    else:
        future_boundary_type, future_boundary_density = (
            _first_future_boundary(
                params=params,
                critical=critical,
                direction=density_direction,
            )
        )

        diagnostics[
            "first_future_boundary"
        ] = future_boundary_type

        diagnostics[
            "first_future_boundary_density"
        ] = future_boundary_density

        background_event = (
            background.attrs.get(
                "termination_event"
            )
        )

        # La información analítica de las raíces solo se usa para
        # decidir qué barrido realizar. La clasificación final
        # sigue dependiendo de la convergencia numérica.
        if future_boundary_type is not None:
            selected_route = (
                f"boundary_{future_boundary_type}"
            )

        elif (
            background_event
            == "rho_dot_denominator_boundary"
        ):
            selected_route = "boundary_D"

        elif (
            background_event
            == "effective_density_boundary"
        ):
            selected_route = "boundary_F"

        else:
            selected_route = "density_limit"

        diagnostics[
            "analysis_route"
        ] = selected_route

        # -----------------------------------------------------
        # 2. Frontera D(rho)=0
        # -----------------------------------------------------

        if selected_route == "boundary_D":
            boundary_scan = (
                scan_boundary_tolerances(
                    params=params,
                    t_end=t_end,
                    boundary_tolerances=(
                        boundary_tolerances
                    ),
                    n_points=n_points,
                    rho_min=rho_min,
                    rho_max=rho_max,
                    a_max=a_max,
                    rtol=rtol,
                    atol=atol,
                    max_step=fine_max_step,
                    strict=False,
                )
            )

            evidence = (
                infer_boundary_evidence(
                    boundary_scan
                )
            )

        # -----------------------------------------------------
        # 3. Frontera F(rho)=0 y continuación signada
        # -----------------------------------------------------

        elif selected_route == "boundary_F":
            boundary_scan = (
                scan_boundary_tolerances(
                    params=params,
                    t_end=t_end,
                    boundary_tolerances=(
                        boundary_tolerances
                    ),
                    n_points=n_points,
                    rho_min=rho_min,
                    rho_max=rho_max,
                    a_max=a_max,
                    rtol=rtol,
                    atol=atol,
                    max_step=fine_max_step,
                    strict=False,
                )
            )

            boundary_evidence = (
                infer_boundary_evidence(
                    boundary_scan
                )
            )

            collapse_thresholds = np.asarray(
                list(
                    collapse_rho_max_values
                ),
                dtype=float,
            )

            if collapse_thresholds.size == 0:
                collapse_thresholds = (
                    np.asarray(
                        [1e2, 1e3, 1e4],
                        dtype=float,
                    )
                )

            signed_rho_max = max(
                float(
                    np.max(
                        collapse_thresholds
                    )
                ),
                rho_max,
            )

            try:
                signed_background = (
                    solve_signed_background_compatible(
                        params,
                        t_end=t_end,
                        n_points=max(
                            n_points,
                            2500,
                        ),
                        rho_min=rho_min,
                        rho_max=signed_rho_max,
                        a_min=a_min,
                        a_max=a_max,
                        H_max=H_max,
                        boundary_tol=(
                            boundary_tol
                        ),
                        rtol=rtol,
                        atol=atol,
                        max_step=fine_max_step,
                    )
                )

                signed_summary = (
                    summarize_signed_background(
                        signed_background
                    )
                )

                diagnostics[
                    "signed_summary"
                ] = signed_summary

                if signed_summary[
                    "turnaround_detected"
                ]:
                    try:
                        collapse_scan = (
                            scan_signed_collapse_thresholds(
                                params=params,
                                t_end=t_end,
                                rho_max_values=(
                                    collapse_thresholds
                                ),
                                n_points=max(
                                    n_points,
                                    2500,
                                ),
                                rho_min=rho_min,
                                a_min=a_min,
                                a_max=a_max,
                                H_max=H_max,
                                boundary_tol=(
                                    boundary_tol
                                ),
                                rtol=rtol,
                                atol=atol,
                                max_step=(
                                    fine_max_step
                                ),
                                strict=False,
                            )
                        )

                        collapse_evidence = (
                            infer_signed_collapse_evidence(
                                collapse_scan
                            )
                        )

                        if collapse_evidence.get(
                            "turnaround_detected",
                            False,
                        ):
                            evidence = (
                                collapse_evidence
                            )
                        else:
                            evidence = (
                                boundary_evidence
                            )

                    except (
                        ValueError,
                        RuntimeError,
                        AttributeError,
                    ) as error:
                        diagnostics[
                            "collapse_scan_error"
                        ] = str(error)

                        evidence = (
                            boundary_evidence
                        )

                else:
                    evidence = (
                        boundary_evidence
                    )

            except (
                ValueError,
                RuntimeError,
                AttributeError,
            ) as error:
                diagnostics[
                    "signed_solver_error"
                ] = str(error)

                evidence = (
                    boundary_evidence
                )

        # -----------------------------------------------------
        # 4. Límite rho -> 0 o rho -> infinito
        # -----------------------------------------------------

        else:
            density_targets = (
                _make_density_targets(
                    params=params,
                    direction=(
                        density_direction
                    ),
                    density_decades=(
                        density_decades
                    ),
                )
            )

            if density_targets:
                density_scan = (
                    scan_density_targets(
                        params=params,
                        rho_targets=(
                            density_targets
                        ),
                        require_future=True,
                        strict=False,
                        epsabs=atol,
                        epsrel=rtol,
                    )
                )

                successful_targets = int(
                    density_scan[
                        "success"
                    ].sum()
                )

                diagnostics[
                    "successful_density_targets"
                ] = successful_targets

                if successful_targets >= 3:
                    evidence = (
                        infer_limit_evidence(
                            density_scan
                        )
                    )

                else:
                    evidence = {
                        "time_limit": "unknown",
                        "rho_limit": "unknown",
                        "a_limit": "unknown",
                        "H_limit": "unknown",
                        "Hdot_limit": "unknown",
                        "rho_dot_limit": "unknown",
                        "Hddot_limit": "unknown",
                        "stable_under_threshold_changes": False,
                        "confidence": 0.2,
                    }

            else:
                evidence = {
                    "stable_under_threshold_changes": False,
                    "confidence": 0.2,
                }

    # Datos físicos iniciales del core para controles analíticos del clasificador.
    evidence["initial_D"] = model.D(params.rho0)
    classification = (
        classify_barotropic_power_diffusion(
            background=background,
            params=params,
            signed_background=(
                signed_background
            ),
            limit_evidence=evidence,
        )
    )

    return BarotropicCaseAnalysis(
        case_id=case_id,
        params=params,
        critical_densities=critical,
        background=background,
        signed_background=(
            signed_background
        ),
        density_scan=density_scan,
        boundary_scan=boundary_scan,
        collapse_scan=collapse_scan,
        evidence=evidence,
        classification=classification,
        diagnostics=diagnostics,
    )

@dataclass
class BarotropicCatalogAnalysis:
    """
    Resultado de analizar un conjunto de modelos barotrópicos.
    """

    summary: pd.DataFrame
    analyses: dict[str, BarotropicCaseAnalysis]
    errors: dict[str, str]

    @property
    def n_cases(self) -> int:
        return len(
            self.analyses
        ) + len(
            self.errors
        )

    @property
    def n_successful(self) -> int:
        return len(
            self.analyses
        )

    @property
    def n_failed(self) -> int:
        return len(
            self.errors
        )

def analyze_parameter_catalog(
    cases: Mapping[
        str,
        BarotropicPowerDiffusionParams,
    ],
    *,
    continue_on_error: bool = True,
    **analysis_kwargs,
) -> BarotropicCatalogAnalysis:
    """
    Analiza automáticamente un catálogo de parámetros.

    Parameters
    ----------
    cases:
        Diccionario de la forma

            {
                "B1": params_B1,
                "B2": params_B2,
                ...
            }

    continue_on_error:
        Si es True, un caso numéricamente problemático se
        registra como error y el catálogo continúa.

    **analysis_kwargs:
        Argumentos enviados a analyze_barotropic_case.

    Returns
    -------
    BarotropicCatalogAnalysis
        Contiene:

        - summary:
            tabla consolidada;

        - analyses:
            resultados completos de cada caso exitoso;

        - errors:
            errores de los casos no completados.
    """

    if not isinstance(
        cases,
        Mapping,
    ):
        raise TypeError(
            "cases debe ser un Mapping."
        )

    if len(cases) == 0:
        raise ValueError(
            "cases no puede estar vacío."
        )

    analyses = {}
    errors = {}

    rows = []

    for case_id, params in cases.items():

        if not isinstance(
            case_id,
            str,
        ):
            raise TypeError(
                "Todos los identificadores deben ser strings."
            )

        if not isinstance(
            params,
            BarotropicPowerDiffusionParams,
        ):
            raise TypeError(
                f"{case_id}: params debe ser "
                "BarotropicPowerDiffusionParams."
            )

        try:
            analysis = (
                analyze_barotropic_case(
                    params=params,
                    case_id=case_id,
                    **analysis_kwargs,
                )
            )

            analyses[
                case_id
            ] = analysis

            row = (
                analysis.summary_dict()
            )

            # Información adicional útil para el análisis
            # científico.
            row[
                "estimated_limit_time"
            ] = analysis.evidence.get(
                "estimated_limit_time"
            )

            row[
                "boundary_type"
            ] = analysis.evidence.get(
                "boundary_type"
            )

            row[
                "turnaround_detected"
            ] = bool(
                analysis.evidence.get(
                    "turnaround_detected",
                    analysis.diagnostics.get(
                        "signed_summary",
                        {},
                    ).get(
                        "turnaround_detected",
                        False,
                    ),
                )
            )

            row[
                "contracting_branch"
            ] = bool(
                analysis.evidence.get(
                    "contracting_branch",
                    False,
                )
            )

            observed = (
                analysis
                .classification
                .as_dict()
                .get(
                    "observed_behavior",
                    {},
                )
            )

            row[
                "p_limit"
            ] = observed.get(
                "p",
                "unknown",
            )

            row[
                "solver_success"
            ] = observed.get(
                "solver_success",
                False,
            )

            row[
                "reached_t_end"
            ] = observed.get(
                "reached_t_end",
                False,
            )

            rows.append(
                row
            )

        except Exception as error:

            if not continue_on_error:
                raise

            errors[
                case_id
            ] = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            rows.append(
                {
                    "case_id": case_id,
                    "omega": float(
                        params.omega
                    ),
                    "alpha": float(
                        params.alpha
                    ),
                    "beta": float(
                        params.beta
                    ),
                    "rho0": float(
                        params.rho0
                    ),
                    "standard_type": None,
                    "behavior_label": (
                        "analysis_error"
                    ),
                    "status": (
                        "numerically_inconclusive"
                    ),
                    "is_singularity": None,
                    "finite_time": None,
                    "confidence": 0.0,
                    "error": errors[
                        case_id
                    ],
                }
            )

    summary = pd.DataFrame(
        rows
    )

    if not summary.empty:
        summary = (
            summary
            .sort_values(
                "case_id"
            )
            .reset_index(
                drop=True
            )
        )

    return BarotropicCatalogAnalysis(
        summary=summary,
        analyses=analyses,
        errors=errors,
    )
