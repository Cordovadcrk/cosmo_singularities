"""
classifiers.py

Funciones de clasificación de singularidades cosmológicas.

Este módulo contiene reglas analíticas y, más adelante, diagnósticos
numéricos. Por ahora implementamos la clasificación analítica del modelo

    p = -rho - A*rho**nu,
    Q = 0,

con A > 0 y rho > 0.

La clasificación se basa en el comportamiento asintótico cuando t -> ts,
donde ts es un tiempo futuro finito.
"""

from __future__ import annotations

import math

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.singularities import (
    ClassificationResult,
    ClassificationStatus,
    KnownCosmologicalBehavior,
    SingularityType,
)


def is_close_to(value: float, target: float, tol: float = 1e-12) -> bool:
    """
    Compara si un número real está suficientemente cerca de otro.

    Esta función es útil para tratar casos frontera como:

        nu = 0,
        nu = 1/2,
        nu = 1.

    En cálculos numéricos, un valor que debería ser 0.5 podría aparecer como
    0.5000000000000001. Por eso no conviene comparar floats solo con ==.

    Parameters
    ----------
    value:
        Valor numérico a comparar.

    target:
        Valor de referencia.

    tol:
        Tolerancia absoluta.

    Returns
    -------
    bool
        True si value está cerca de target dentro de la tolerancia.
    """

    return math.isclose(value, target, abs_tol=tol, rel_tol=0.0)


def classify_phantom_q0_by_nu(
    nu: float,
    A: float = 1.0,
    tol: float = 1e-12,
) -> SingularityType:
    """
    Clasifica analíticamente el modelo

        p = -rho - A*rho**nu,
        Q = 0,

    bajo la hipótesis A > 0.

    Parameters
    ----------
    nu:
        Exponente de la ecuación de estado inhomogénea.

    A:
        Amplitud del término phantom. En esta primera versión se asume A > 0.

    tol:
        Tolerancia para detectar casos frontera.

    Returns
    -------
    SingularityType
        Tipo de singularidad futuro esperado.

    Notes
    -----
    Para A > 0:

    - nu < 0:
        Tipo II. La densidad tiende a cero o a un valor finito, pero
        la presión diverge.

    - 0 < nu < 1/2:
        Tipo IV. rho, P, H y dot(H) permanecen finitos, pero pueden
        divergir derivadas superiores de H.

    - nu = 0:
        Caso frontera. Se marca como no singular en esta primera versión.

    - nu = 1/2:
        No hay singularidad futura finita en la clasificación estándar.

    - 1/2 < nu < 1:
        Tipo I / Big Rip. Divergen a(t), rho(t), |P(t)| y H(t).

    - nu = 1:
        Caso lineal phantom equivalente a w = -1 - A. Para A > 0 produce
        Big Rip.

    - nu > 1:
        Tipo III. Divergen rho(t), |P(t)| y H(t), pero a(t) permanece finito.
    """

    if not math.isfinite(nu):
        return SingularityType.AMBIGUOUS

    if not math.isfinite(A):
        return SingularityType.AMBIGUOUS

    if A <= 0:
        return SingularityType.AMBIGUOUS

    # Casos frontera
    if is_close_to(nu, 0.0, tol=tol):
        return SingularityType.NON_SINGULAR

    if is_close_to(nu, 0.5, tol=tol):
        return SingularityType.NON_SINGULAR

    if is_close_to(nu, 1.0, tol=tol):
        return SingularityType.TYPE_I

    # Intervalos abiertos
    if nu < 0.0:
        return SingularityType.TYPE_II

    if 0.0 < nu < 0.5:
        return SingularityType.TYPE_IV

    if 0.5 < nu < 1.0:
        return SingularityType.TYPE_I

    if nu > 1.0:
        return SingularityType.TYPE_III

    return SingularityType.AMBIGUOUS

def classify_phantom_qalpha_rho_by_nu(
    nu: float,
    A: float = 1.0,
    alpha: float = 0.0,
    tol: float = 1e-12,
) -> SingularityType:
    """
    Clasifica analíticamente el modelo

        p = -rho - A*rho**nu,
        Q = alpha*rho,

    bajo las hipótesis:

        A > 0,
        rho > 0,
        1 + alpha > 0.

    En la convención usada en este proyecto:

        3H**2 = rho + Q = (1 + alpha)*rho.

    Para alpha > -1, la difusión Q = alpha*rho modifica la escala temporal,
    pero la clasificación cualitativa por intervalos de nu coincide con el
    caso Q = 0.

    Parameters
    ----------
    nu:
        Exponente de la ecuación de estado.

    A:
        Amplitud phantom. Se asume A > 0.

    alpha:
        Parámetro de difusión. Se requiere 1 + alpha > 0.

    tol:
        Tolerancia para casos frontera.

    Returns
    -------
    SingularityType
        Tipo de singularidad esperado.
    """

    if not math.isfinite(alpha):
        return SingularityType.AMBIGUOUS

    if 1.0 + alpha <= 0.0:
        return SingularityType.AMBIGUOUS

    return classify_phantom_q0_by_nu(nu=nu, A=A, tol=tol)

_LIMIT_STATE_ALIASES = {
    "diverge": "diverges",
    "divergent": "diverges",
    "diverges": "diverges",
    "infinite": "infinite",
    "infinity": "infinite",
    "inf": "infinite",

    "finite": "finite",
    "bounded": "finite",

    "zero": "zero",
    "tends_to_zero": "zero",
    "goes_to_zero": "zero",

    "constant": "constant",
    "tends_to_constant": "constant",

    "oscillatory": "oscillatory",
    "bounded_oscillations": "oscillatory",

    "unknown": "unknown",
    "not_determined": "unknown",
    "inconclusive": "unknown",
}


def normalize_limit_state(
    value: Any,
) -> str:
    """
    Normaliza la descripción de un límite cosmológico.

    La función acepta etiquetas conocidas, pero conserva etiquetas
    nuevas que aparezcan durante la exploración del modelo.
    """

    if value is None:
        return "unknown"

    normalized = (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if not normalized:
        return "unknown"

    return _LIMIT_STATE_ALIASES.get(
        normalized,
        normalized,
    )

_REQUIRED_BACKGROUND_COLUMNS = {
    "t",
    "rho",
    "a",
    "H",
    "Hdot",
}


def validate_background(
    background: pd.DataFrame,
) -> None:
    """
    Valida la estructura mínima de una evolución cosmológica.
    """

    if not isinstance(
        background,
        pd.DataFrame,
    ):
        raise TypeError(
            "background debe ser un pandas.DataFrame."
        )

    missing_columns = (
        _REQUIRED_BACKGROUND_COLUMNS
        - set(background.columns)
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Faltan columnas requeridas en background: "
            f"{missing_text}."
        )

    if len(background) < 2:
        raise ValueError(
            "background debe contener al menos dos filas."
        )

    numerical_values = background[
        sorted(_REQUIRED_BACKGROUND_COLUMNS)
    ].to_numpy(dtype=float)

    if not np.isfinite(
        numerical_values
    ).all():
        raise ValueError(
            "background contiene valores no finitos."
        )

    t_values = background[
        "t"
    ].to_numpy(dtype=float)

    if np.any(
        np.diff(t_values) <= 0.0
    ):
        raise ValueError(
            "La columna t debe ser estrictamente creciente."
        )

    if np.any(
        background["rho"].to_numpy(
            dtype=float
        )
        <= 0.0
    ):
        raise ValueError(
            "La densidad debe permanecer positiva."
        )

    if np.any(
        background["a"].to_numpy(
            dtype=float
        )
        <= 0.0
    ):
        raise ValueError(
            "El factor de escala debe permanecer positivo."
        )

def tail_relative_spread(
    values,
    tail_fraction: float = 0.1,
) -> float:
    """
    Calcula la dispersión relativa del tramo final de una serie.

    Un valor pequeño indica que la serie se aproxima a una
    constante dentro del intervalo numérico estudiado.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size < 2:
        return np.nan

    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError(
            "tail_fraction debe pertenecer a (0, 1]."
        )

    tail_size = max(
        2,
        int(
            math.ceil(
                tail_fraction
                * array.size
            )
        ),
    )

    tail = array[-tail_size:]

    scale = max(
        float(
            np.max(
                np.abs(tail)
            )
        ),
        1e-300,
    )

    return float(
        (
            np.max(tail)
            - np.min(tail)
        )
        / scale
    )


def count_H_zero_crossings(
    H_values,
    zero_tolerance: float = 1e-10,
) -> int:
    """
    Cuenta cambios de signo reales de H.

    Los valores muy cercanos a cero se excluyen antes de contar
    los cruces para evitar falsos positivos numéricos.
    """

    H_array = np.asarray(
        H_values,
        dtype=float,
    )

    finite_values = H_array[
        np.isfinite(H_array)
    ]

    nonzero_values = finite_values[
        np.abs(finite_values)
        > zero_tolerance
    ]

    if nonzero_values.size < 2:
        return 0

    return int(
        np.sum(
            nonzero_values[:-1]
            * nonzero_values[1:]
            < 0.0
        )
    )

def extract_barotropic_observations(
    background: pd.DataFrame,
    *,
    signed_background: pd.DataFrame | None = None,
    limit_evidence: Mapping[str, Any] | None = None,
    tail_fraction: float = 0.1,
    zero_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """
    Extrae observaciones numéricas de una solución barotrópica.

    Parameters
    ----------
    background:
        Solución de la rama expansiva.

    signed_background:
        Solución opcional donde H se integra con signo. Cuando está
        disponible se utiliza para detectar turnaround y contracción.

    limit_evidence:
        Evidencia obtenida mediante convergencia de umbrales o
        tolerancias. Puede contener, por ejemplo:

            time_limit
            rho_limit
            a_limit
            H_limit
            Hdot_limit
            rho_dot_limit
            Hddot_limit
            stable_under_threshold_changes
            stable_under_tolerance_changes
            confidence

        La función no exige que todas las claves estén presentes.

    Returns
    -------
    dict
        Descripción abierta del comportamiento observado.
    """

    validate_background(
        background
    )

    if signed_background is not None:
        validate_background(
            signed_background
        )

        analyzed_background = (
            signed_background
        )
    else:
        analyzed_background = (
            background
        )

    evidence = dict(
        limit_evidence or {}
    )

    initial = (
        analyzed_background.iloc[0]
    )

    final = (
        analyzed_background.iloc[-1]
    )

    H_values = (
        analyzed_background["H"]
        .to_numpy(dtype=float)
    )

    H_zero_crossings = (
        count_H_zero_crossings(
            H_values,
            zero_tolerance=zero_tolerance,
        )
    )

    expansion_to_contraction = bool(
        H_zero_crossings >= 1
        and final["H"] < -zero_tolerance
    )

    observations = {
        "time_limit": normalize_limit_state(
            evidence.get(
                "time_limit",
                "unknown",
            )
        ),
        "rho": normalize_limit_state(
            evidence.get(
                "rho_limit",
                "unknown",
            )
        ),
        "a": normalize_limit_state(
            evidence.get(
                "a_limit",
                "unknown",
            )
        ),
        "H": normalize_limit_state(
            evidence.get(
                "H_limit",
                "unknown",
            )
        ),
        "Hdot": normalize_limit_state(
            evidence.get(
                "Hdot_limit",
                "unknown",
            )
        ),
        "rho_dot": normalize_limit_state(
            evidence.get(
                "rho_dot_limit",
                "unknown",
            )
        ),
        "Hddot": normalize_limit_state(
            evidence.get(
                "Hddot_limit",
                "unknown",
            )
        ),
        "termination_event": (
            analyzed_background.attrs.get(
                "termination_event"
            )
        ),
        "solver_success": bool(
            analyzed_background.attrs.get(
                "solver_success",
                False,
            )
        ),
        "reached_t_end": bool(
            analyzed_background.attrs.get(
                "reached_t_end",
                False,
            )
        ),
        "H_zero_crossings": (
            H_zero_crossings
        ),
        "expansion_to_contraction": (
            expansion_to_contraction
        ),
        "initial_H_sign": int(
            np.sign(
                float(initial["H"])
            )
        ),
        "final_H_sign": int(
            np.sign(
                float(final["H"])
            )
        ),
        "initial_values": {
            "t": float(initial["t"]),
            "rho": float(initial["rho"]),
            "a": float(initial["a"]),
            "H": float(initial["H"]),
            "Hdot": float(
                initial["Hdot"]
            ),
        },
        "final_values": {
            "t": float(final["t"]),
            "rho": float(final["rho"]),
            "a": float(final["a"]),
            "H": float(final["H"]),
            "Hdot": float(
                final["Hdot"]
            ),
        },
        "tail_relative_spread": {
            variable: tail_relative_spread(
                analyzed_background[
                    variable
                ].to_numpy(dtype=float),
                tail_fraction=tail_fraction,
            )
            for variable in [
                "rho",
                "a",
                "H",
                "Hdot",
            ]
        },
    }

    if (
        "maximum_constraint_residual"
        in analyzed_background.attrs
    ):
        observations[
            "maximum_constraint_residual"
        ] = float(
            analyzed_background.attrs[
                "maximum_constraint_residual"
            ]
        )

    return observations

def estimate_classification_confidence(
    observations: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> float:
    """
    Estima la solidez diagnóstica de una clasificación.

    No representa una probabilidad estadística.
    """

    if "confidence" in evidence:
        confidence = float(
            evidence["confidence"]
        )

        return float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

    confidence = 0.35

    if observations.get(
        "solver_success",
        False,
    ):
        confidence += 0.10

    if evidence.get(
        "stable_under_threshold_changes",
        False,
    ):
        confidence += 0.20

    if evidence.get(
        "stable_under_tolerance_changes",
        False,
    ):
        confidence += 0.20

    constraint_residual = (
        observations.get(
            "maximum_constraint_residual"
        )
    )

    if (
        constraint_residual is not None
        and np.isfinite(
            constraint_residual
        )
        and constraint_residual < 1e-6
    ):
        confidence += 0.10

    if observations.get(
        "time_limit"
    ) in {
        "finite",
        "infinite",
    }:
        confidence += 0.05

    return float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )

def classify_barotropic_power_diffusion(
    background: pd.DataFrame,
    params: Any,
    *,
    signed_background: pd.DataFrame | None = None,
    limit_evidence: Mapping[str, Any] | None = None,
) -> ClassificationResult:
    """
    Clasifica numéricamente el modelo

        p = omega*rho,
        Q = alpha*rho**beta.

    La función no deduce el resultado únicamente desde omega, alpha
    y beta. Primero analiza la solución numérica y la evidencia de
    convergencia suministrada.

    La clasificación estándar TYPE_I--TYPE_IV solo se asigna cuando
    los límites observados satisfacen los criterios correspondientes.
    """

    evidence = dict(
        limit_evidence or {}
    )

    observations = (
        extract_barotropic_observations(
            background=background,
            signed_background=(
                signed_background
            ),
            limit_evidence=evidence,
        )
    )

    confidence = (
        estimate_classification_confidence(
            observations=observations,
            evidence=evidence,
        )
    )

    solver_success = observations[
        "solver_success"
    ]

    time_limit = observations[
        "time_limit"
    ]

    rho_limit = observations["rho"]
    a_limit = observations["a"]
    H_limit = observations["H"]
    Hdot_limit = observations["Hdot"]

    rho_dot_limit = observations[
        "rho_dot"
    ]

    Hddot_limit = observations[
        "Hddot"
    ]

    # ---------------------------------------------------------
    # Límite de la presión barotrópica
    #
    # p = omega * rho
    # ---------------------------------------------------------

    omega = float(
        getattr(
            params,
            "omega",
            np.nan,
        )
    )

    if not np.isfinite(omega):
        p_limit = "unknown"

    elif math.isclose(
        omega,
        0.0,
        abs_tol=1e-14,
        rel_tol=0.0,
    ):
        p_limit = "zero"

    elif rho_limit == "diverges":
        p_limit = "diverges"

    elif rho_limit == "zero":
        p_limit = "zero"

    elif rho_limit in {
        "finite",
        "constant",
    }:
        p_limit = "finite"

    else:
        p_limit = "unknown"

    observations["p"] = p_limit

    termination_event = observations[
        "termination_event"
    ]


    finite_time = (
        True
        if time_limit == "finite"
        else False
        if time_limit == "infinite"
        else None
    )

    finite_states = {
        "finite",
        "zero",
        "constant",
    }

    principal_quantities_finite = all(
        state in finite_states
        for state in [
            rho_limit,
            a_limit,
            H_limit,
            Hdot_limit,
        ]
    )

    higher_derivative_divergence = (
        rho_dot_limit == "diverges"
        or Hddot_limit == "diverges"
    )

    def make_result(
        *,
        standard_type,
        behavior_label,
        status,
        is_singularity,
        finite_time_value,
        notes,
    ):
        return ClassificationResult(
            standard_type=standard_type,
            behavior_label=behavior_label,
            status=status,
            is_singularity=(
                is_singularity
            ),
            finite_time=(
                finite_time_value
            ),
            confidence=confidence,
            observed_behavior=(
                observations
            ),
            evidence=evidence,
            notes=notes,
        )

    # ---------------------------------------------------------
    # 1. Fallo o insuficiencia numérica
    # ---------------------------------------------------------

    if not solver_success:
        return make_result(
            standard_type=(
                SingularityType.AMBIGUOUS
            ),
            behavior_label=(
                KnownCosmologicalBehavior
                .NUMERICALLY_INCONCLUSIVE
            ),
            status=(
                ClassificationStatus
                .NUMERICALLY_INCONCLUSIVE
            ),
            is_singularity=None,
            finite_time_value=None,
            notes=(
                "El integrador no completó satisfactoriamente "
                "la evolución."
            ),
        )

    # ---------------------------------------------------------
    # 2. Turnaround y continuación contractiva
    # ---------------------------------------------------------

    if observations[
        "expansion_to_contraction"
    ]:
        if (
            finite_time is True
            and a_limit == "zero"
            and (
                rho_limit == "diverges"
                or H_limit == "diverges"
            )
        ):
            return make_result(
                standard_type=None,
                behavior_label=(
                    KnownCosmologicalBehavior
                    .TURNAROUND_TO_BIG_CRUNCH
                ),
                status=(
                    ClassificationStatus.ROBUST
                    if confidence >= 0.8
                    else ClassificationStatus
                    .PROVISIONAL
                ),
                is_singularity=True,
                finite_time_value=True,
                notes=(
                    "La solución cruza H=0, entra en "
                    "contracción y termina en un colapso "
                    "de tipo Big Crunch."
                ),
            )

        if (
            termination_event
            == "rho_dot_denominator_boundary"
            and higher_derivative_divergence
        ):
            standard_type = (
                SingularityType.TYPE_IV
                if (
                    finite_time is True
                    and principal_quantities_finite
                )
                else None
            )

            return make_result(
                standard_type=standard_type,
                behavior_label=(
                    KnownCosmologicalBehavior
                    .TURNAROUND_TO_HIGHER_DERIVATIVE_BOUNDARY
                ),
                status=(
                    ClassificationStatus.ROBUST
                    if confidence >= 0.8
                    else ClassificationStatus
                    .PROVISIONAL
                ),
                is_singularity=(
                    True
                    if finite_time is True
                    else None
                ),
                finite_time_value=(
                    finite_time
                ),
                notes=(
                    "La solución pasa de expansión a "
                    "contracción y alcanza una frontera "
                    "donde divergen derivadas superiores."
                ),
            )

        return make_result(
            standard_type=None,
            behavior_label=(
                KnownCosmologicalBehavior
                .TURNAROUND_TO_CONTRACTION
            ),
            status=(
                ClassificationStatus.PROVISIONAL
            ),
            is_singularity=False,
            finite_time_value=finite_time,
            notes=(
                "Se detectó un cruce de H positivo a H "
                "negativo. La evolución posterior no coincide "
                "todavía con una singularidad estándar."
            ),
        )

    # ---------------------------------------------------------
    # 3. Frontera D(rho)=0
    # ---------------------------------------------------------

    if (
        termination_event
        == "rho_dot_denominator_boundary"
    ):
        if (
            finite_time is True
            and principal_quantities_finite
            and higher_derivative_divergence
        ):
            return make_result(
                standard_type=(
                    SingularityType.TYPE_IV
                ),
                behavior_label=(
                    KnownCosmologicalBehavior
                    .HIGHER_DERIVATIVE_BOUNDARY
                ),
                status=(
                    ClassificationStatus.ROBUST
                    if confidence >= 0.8
                    else ClassificationStatus
                    .PROVISIONAL
                ),
                is_singularity=True,
                finite_time_value=True,
                notes=(
                    "rho, a, H y dot(H) permanecen finitos, "
                    "pero dot(rho) o una derivada superior de "
                    "H diverge en tiempo finito."
                ),
            )

        return make_result(
            standard_type=None,
            behavior_label=(
                KnownCosmologicalBehavior
                .CONTINUITY_DENOMINATOR_BOUNDARY
            ),
            status=(
                ClassificationStatus.PROVISIONAL
            ),
            is_singularity=None,
            finite_time_value=finite_time,
            notes=(
                "La solución alcanza D(rho)=0, pero la "
                "evidencia todavía no permite confirmar "
                "una singularidad tipo IV."
            ),
        )

    # ---------------------------------------------------------
    # 4. Frontera F(rho)=0
    # ---------------------------------------------------------

    if (
        termination_event
        == "effective_density_boundary"
    ):
        return make_result(
            standard_type=None,
            behavior_label=(
                KnownCosmologicalBehavior
                .FRIEDMANN_DOMAIN_BOUNDARY
            ),
            status=(
                ClassificationStatus.PROVISIONAL
            ),
            is_singularity=None,
            finite_time_value=finite_time,
            notes=(
                "La rama expansiva alcanza H=0. Se requiere "
                "la solución con H signado para determinar si "
                "existe turnaround, rebote u otra continuación."
            ),
        )

    # ---------------------------------------------------------
    # 5. Singularidades estándar en tiempo finito
    # ---------------------------------------------------------

    if finite_time is True:
        if (
            a_limit == "diverges"
            and rho_limit == "diverges"
            and H_limit == "diverges"
        ):
            return make_result(
                standard_type=(
                    SingularityType.TYPE_I
                ),
                behavior_label=(
                    KnownCosmologicalBehavior
                    .TYPE_I_BIG_RIP
                ),
                status=(
                    ClassificationStatus.ROBUST
                    if confidence >= 0.8
                    else ClassificationStatus
                    .PROVISIONAL
                ),
                is_singularity=True,
                finite_time_value=True,
                notes=(
                    "El factor de escala, la densidad y H "
                    "divergen en un tiempo futuro finito."
                ),
            )

    # ---------------------------------------------------------
    # Tipo II: sudden singularity
    # ---------------------------------------------------------

        if (
            finite_time is True
            and a_limit == "finite"
            and rho_limit in {
                "finite",
                "constant",
            }
            and p_limit == "diverges"
        ):
            return make_result(
                standard_type=(
                    SingularityType.TYPE_II
                ),
                behavior_label=(
                    KnownCosmologicalBehavior
                    .TYPE_II_SUDDEN
                ),
                status=(
                    ClassificationStatus.ROBUST
                    if confidence >= 0.8
                    else ClassificationStatus.PROVISIONAL
                ),
                is_singularity=True,
                finite_time_value=True,
                notes=(
                    "El factor de escala y la densidad "
                    "permanecen finitos, mientras la presión "
                    "diverge en un tiempo finito."
                ),
            )

        if (
            rho_limit == "finite"
            and a_limit == "finite"
            and H_limit == "finite"
            and Hdot_limit == "diverges"
        ):
            return make_result(
                standard_type=(
                    SingularityType.TYPE_II
                ),
                behavior_label=(
                    KnownCosmologicalBehavior
                    .TYPE_II_SUDDEN
                ),
                status=(
                    ClassificationStatus.PROVISIONAL
                ),
                is_singularity=True,
                finite_time_value=True,
                notes=(
                    "Las variables principales permanecen "
                    "finitas, pero dot(H) diverge."
                ),
            )

        if (
            principal_quantities_finite
            and higher_derivative_divergence
        ):
            return make_result(
                standard_type=(
                    SingularityType.TYPE_IV
                ),
                behavior_label=(
                    KnownCosmologicalBehavior
                    .TYPE_IV
                ),
                status=(
                    ClassificationStatus.PROVISIONAL
                ),
                is_singularity=True,
                finite_time_value=True,
                notes=(
                    "Las cantidades principales permanecen "
                    "finitas y diverge una derivada superior."
                ),
            )

        return make_result(
            standard_type=None,
            behavior_label=(
                KnownCosmologicalBehavior
                .FINITE_TIME_DIVERGENCE
            ),
            status=(
                ClassificationStatus
                .NEW_BEHAVIOR_CANDIDATE
            ),
            is_singularity=True,
            finite_time_value=True,
            notes=(
                "Se detectó un comportamiento singular en "
                "tiempo finito que no coincide de forma "
                "concluyente con los tipos I--IV."
            ),
        )

    # ---------------------------------------------------------
    # 6. Solución de Sitter exacta
    # ---------------------------------------------------------

    tail_spread = observations[
        "tail_relative_spread"
    ]

    if (
        np.isfinite(omega)
        and is_close_to(
            omega,
            -1.0,
            tol=1e-10,
        )
        and tail_spread["rho"] < 1e-8
        and tail_spread["H"] < 1e-8
        and abs(
            observations[
                "final_values"
            ]["Hdot"]
        )
        < 1e-10
    ):
        return make_result(
            standard_type=(
                SingularityType.NON_SINGULAR
            ),
            behavior_label=(
                KnownCosmologicalBehavior
                .DE_SITTER
            ),
            status=(
                ClassificationStatus.ROBUST
            ),
            is_singularity=False,
            finite_time_value=False,
            notes=(
                "rho y H permanecen constantes y "
                "dot(H)=0."
            ),
        )

    # ---------------------------------------------------------
    # 7. de Sitter asintótico
    # ---------------------------------------------------------

    if (
        finite_time is False
        and H_limit in {
            "finite",
            "constant",
        }
        and Hdot_limit == "zero"
    ):
        return make_result(
            standard_type=(
                SingularityType.NON_SINGULAR
            ),
            behavior_label=(
                KnownCosmologicalBehavior
                .ASYMPTOTIC_DE_SITTER
            ),
            status=(
                ClassificationStatus.ROBUST
                if confidence >= 0.8
                else ClassificationStatus.PROVISIONAL
            ),
            is_singularity=False,
            finite_time_value=False,
            notes=(
                "La evolución se aproxima a un estado "
                "con H constante y dot(H) tendiendo a cero."
            ),
        )

    # ---------------------------------------------------------
    # 8. Comportamientos de tiempo infinito
    # ---------------------------------------------------------

    if finite_time is False:

        # -----------------------------------------------------
        # 8.1 Rip dominado por difusión
        #
        # rho -> 0,
        # a   -> infinity,
        # H   -> infinity,
        # Hdot -> 0.
        # -----------------------------------------------------

        if (
            rho_limit == "zero"
            and a_limit == "diverges"
            and H_limit == "diverges"
            and Hdot_limit == "zero"
        ):
            return make_result(
                standard_type=(
                    SingularityType.NON_SINGULAR
                ),
                behavior_label=(
                    KnownCosmologicalBehavior
                    .DIFFUSION_DRIVEN_INFINITE_TIME_RIP
                ),
                status=(
                    ClassificationStatus
                    .NEW_BEHAVIOR_CANDIDATE
                ),
                is_singularity=False,
                finite_time_value=False,
                notes=(
                    "La densidad y dot(H) tienden a cero, "
                    "mientras a y H divergen únicamente "
                    "cuando t tiende a infinito. "
                    "La evolución está dominada por el "
                    "término de difusión."
                ),
            )

        # -----------------------------------------------------
        # 8.2 Little Sibling of the Big Rip
        #
        # t -> infinity,
        # rho -> infinity,
        # a -> infinity,
        # H -> infinity,
        # Hdot -> constante finita no nula.
        # -----------------------------------------------------

        final_Hdot = float(
            observations[
                "final_values"
            ]["Hdot"]
        )

        if (
            rho_limit == "diverges"
            and a_limit == "diverges"
            and H_limit == "diverges"
            and Hdot_limit == "constant"
            and abs(final_Hdot) > 1e-10
        ):
            return make_result(
                standard_type=(
                    SingularityType.NON_SINGULAR
                ),
                behavior_label=(
                    KnownCosmologicalBehavior
                    .LITTLE_SIBLING_BIG_RIP
                ),
                status=(
                    ClassificationStatus.ROBUST
                    if confidence >= 0.8
                    else ClassificationStatus.PROVISIONAL
                ),
                is_singularity=False,
                finite_time_value=False,
                notes=(
                    "a, rho y H divergen únicamente en "
                    "tiempo infinito, mientras dot(H) "
                    "tiende a una constante finita no nula. "
                    "El comportamiento es compatible con "
                    "un Little Sibling of the Big Rip."
                ),
            )

        # -----------------------------------------------------
        # 8.3 Little Rip
        #
        # El Little Sibling debe evaluarse antes, porque es
        # un caso más específico.
        # -----------------------------------------------------

        if (
            rho_limit == "diverges"
            and a_limit == "diverges"
            and H_limit == "diverges"
        ):
            return make_result(
                standard_type=(
                    SingularityType.NON_SINGULAR
                ),
                behavior_label=(
                    KnownCosmologicalBehavior
                    .LITTLE_RIP
                ),
                status=(
                    ClassificationStatus.ROBUST
                    if confidence >= 0.8
                    else ClassificationStatus.PROVISIONAL
                ),
                is_singularity=False,
                finite_time_value=False,
                notes=(
                    "a, rho y H divergen, pero la "
                    "divergencia solo se alcanza cuando "
                    "t tiende a infinito. El comportamiento "
                    "es compatible con un Little Rip."
                ),
            )

        # -----------------------------------------------------
        # 8.4 Divergencia infinita genérica
        # -----------------------------------------------------

        if any(
            state == "diverges"
            for state in [
                rho_limit,
                H_limit,
                Hdot_limit,
                rho_dot_limit,
                Hddot_limit,
            ]
        ):
            return make_result(
                standard_type=(
                    SingularityType.NON_SINGULAR
                ),
                behavior_label=(
                    KnownCosmologicalBehavior
                    .INFINITE_TIME_DIVERGENCE
                ),
                status=(
                    ClassificationStatus.ROBUST
                    if confidence >= 0.8
                    else ClassificationStatus.PROVISIONAL
                ),
                is_singularity=False,
                finite_time_value=False,
                notes=(
                    "Una o más cantidades dinámicas "
                    "divergen únicamente cuando t tiende "
                    "a infinito, pero el patrón no "
                    "corresponde de forma concluyente a "
                    "Little Rip ni Little Sibling."
                ),
            )
    # ---------------------------------------------------------
    # 9. Evolución regular
    # ---------------------------------------------------------

    if (
        finite_time is False
        and all(
            state != "diverges"
            for state in [
                rho_limit,
                H_limit,
                Hdot_limit,
            ]
        )
    ):
        return make_result(
            standard_type=(
                SingularityType.NON_SINGULAR
            ),
            behavior_label=(
                KnownCosmologicalBehavior
                .REGULAR_EXPANSION
            ),
            status=(
                ClassificationStatus.PROVISIONAL
            ),
            is_singularity=False,
            finite_time_value=False,
            notes=(
                "No se detectaron divergencias futuras en "
                "tiempo finito."
            ),
        )

    # ---------------------------------------------------------
    # 10. Comportamiento no clasificado
    # ---------------------------------------------------------

    stable_evidence = bool(
        evidence.get(
            "stable_under_threshold_changes",
            False,
        )
        or evidence.get(
            "stable_under_tolerance_changes",
            False,
        )
    )

    if stable_evidence:
        status = (
            ClassificationStatus
            .NEW_BEHAVIOR_CANDIDATE
        )

        behavior_label = (
            KnownCosmologicalBehavior
            .NEW_BEHAVIOR_CANDIDATE
        )

        notes = (
            "La evolución es numéricamente estable, pero "
            "no coincide con las clasificaciones conocidas."
        )

    else:
        status = (
            ClassificationStatus
            .NUMERICALLY_INCONCLUSIVE
        )

        behavior_label = (
            KnownCosmologicalBehavior
            .UNCLASSIFIED
        )

        notes = (
            "La evidencia disponible es insuficiente para "
            "obtener una clasificación robusta."
        )

    return make_result(
        standard_type=None,
        behavior_label=behavior_label,
        status=status,
        is_singularity=None,
        finite_time_value=finite_time,
        notes=notes,
    )