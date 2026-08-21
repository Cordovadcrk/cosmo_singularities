"""
datasets.py

Construcción de datasets para los modelos cosmológicos del proyecto.

La idea es barrer parámetros físicos, evaluar la evolución temporal de cada
modelo y guardar un resumen por combinación de parámetros.

Por ahora se implementan datasets para:

1. PhantomQ0Model:
       P = -rho - A*rho**nu
       Q = 0

2. PhantomQAlphaRhoModel:
       P = -rho - A*rho**nu
       Q = alpha*rho
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pandas as pd
from src.singularities import SingularityType
from src.models import (
    PhantomQ0Params,
    PhantomQ0Model,
    PhantomQAlphaRhoParams,
    PhantomQAlphaRhoModel,
)


def _safe_float(value):
    """
    Convierte valores numéricos a float de forma segura.

    Si el valor no es finito, devuelve NaN.
    """

    try:
        value = float(value)
    except Exception:
        return np.nan

    if not math.isfinite(value):
        return np.nan

    return value


def _summarize_background(
    df: pd.DataFrame,
    model_name: str,
    params: dict,
    singularity_type,
    singular_time,
    valid_solution: bool = True,
    error_message: str = "",
) -> dict:
    """
    Construye un resumen numérico de una evolución cosmológica.

    Parameters
    ----------
    df:
        DataFrame producido por evaluate_background_t.

    model_name:
        Nombre del modelo.

    params:
        Diccionario con parámetros físicos.

    singularity_type:
        Tipo de singularidad analítica.

    singular_time:
        Tiempo singular si existe. Puede ser None.

    valid_solution:
        Indica si la evolución fue calculada correctamente.

    error_message:
        Mensaje de error si la solución falló.

    Returns
    -------
    dict
        Registro listo para agregarse al dataset.
    """

    record = {
        "model_name": model_name,
        **params,
        "singularity_type": singularity_type.value,
        "singular_time": np.nan if singular_time is None else _safe_float(singular_time),
        "valid_solution": valid_solution,
        "error_message": error_message,
    }

    if not valid_solution or df is None or len(df) == 0:
        record.update(
            {
                "t_eval_min": np.nan,
                "t_eval_max": np.nan,
                "rho_initial": np.nan,
                "rho_final": np.nan,
                "rho_max": np.nan,
                "a_initial": np.nan,
                "a_final": np.nan,
                "a_max": np.nan,
                "H_initial": np.nan,
                "H_final": np.nan,
                "H_max": np.nan,
                "P_initial": np.nan,
                "P_final": np.nan,
                "P_abs_max": np.nan,
                "Hdot_initial": np.nan,
                "Hdot_final": np.nan,
                "Hdot_max": np.nan,
            }
        )
        return record

    record.update(
        {
            "t_eval_min": _safe_float(df["t"].iloc[0]),
            "t_eval_max": _safe_float(df["t"].iloc[-1]),
            "rho_initial": _safe_float(df["rho"].iloc[0]),
            "rho_final": _safe_float(df["rho"].iloc[-1]),
            "rho_max": _safe_float(df["rho"].max()),
            "a_initial": _safe_float(df["a"].iloc[0]),
            "a_final": _safe_float(df["a"].iloc[-1]),
            "a_max": _safe_float(df["a"].max()),
            "H_initial": _safe_float(df["H"].iloc[0]),
            "H_final": _safe_float(df["H"].iloc[-1]),
            "H_max": _safe_float(df["H"].max()),
            "P_initial": _safe_float(df["P"].iloc[0]),
            "P_final": _safe_float(df["P"].iloc[-1]),
            "P_abs_max": _safe_float(np.abs(df["P"]).max()),
            "Hdot_initial": _safe_float(df["Hdot"].iloc[0]),
            "Hdot_final": _safe_float(df["Hdot"].iloc[-1]),
            "Hdot_max": _safe_float(df["Hdot"].max()),
        }
    )

    return record


def _make_time_grid_for_model(
    model,
    t0: float,
    t_end_default: float,
    n_points: int,
    singular_fraction: float = 0.95,
):
    """
    Crea una grilla temporal segura.

    Si el modelo tiene tiempo singular finito, evalúa hasta una fracción
    del tiempo singular. Si no, evalúa hasta t_end_default.

    Parameters
    ----------
    model:
        Modelo cosmológico. Puede tener o no método singular_time.

    t0:
        Tiempo inicial.

    t_end_default:
        Tiempo máximo usado cuando no hay singularidad finita.

    n_points:
        Número de puntos temporales.

    singular_fraction:
        Fracción del tiempo singular usada para evitar evaluar exactamente
        en la divergencia.

    Returns
    -------
    numpy.ndarray
        Grilla temporal.
    """

    singular_time = None

    if hasattr(model, "singular_time"):
        singular_time = model.singular_time()

    if singular_time is not None and math.isfinite(singular_time):
        t_end = t0 + singular_fraction * (singular_time - t0)
    else:
        t_end = t_end_default

    if t_end <= t0:
        raise ValueError("La grilla temporal no es válida: t_end <= t0.")

    return np.linspace(t0, t_end, n_points)


def build_phantom_q0_dataset(
    A_values,
    nu_values,
    rho0_values=(1.0,),
    a0: float = 1.0,
    t0: float = 0.0,
    t_end_default: float = 10.0,
    n_points: int = 500,
    singular_fraction: float = 0.95,
) -> pd.DataFrame:
    """
    Construye un dataset para el modelo

        P = -rho - A*rho**nu,
        Q = 0.

    Parameters
    ----------
    A_values:
        Lista o arreglo de valores de A. Se espera A > 0.

    nu_values:
        Lista o arreglo de valores de nu.

    rho0_values:
        Lista o arreglo de densidades iniciales.

    a0:
        Factor de escala inicial.

    t0:
        Tiempo inicial.

    t_end_default:
        Tiempo máximo si no hay singularidad finita.

    n_points:
        Número de puntos temporales usados para resumir la evolución.

    singular_fraction:
        Fracción del tiempo singular usada cuando hay singularidad finita.

    Returns
    -------
    pandas.DataFrame
        Dataset resumido por combinación de parámetros.
    """

    records = []

    for A, nu, rho0 in product(A_values, nu_values, rho0_values):
        params_dict = {
            "A": float(A),
            "nu": float(nu),
            "alpha": np.nan,
            "rho0": float(rho0),
            "a0": float(a0),
            "t0": float(t0),
        }

        try:
            params = PhantomQ0Params(
                A=float(A),
                nu=float(nu),
                rho0=float(rho0),
                a0=float(a0),
                t0=float(t0),
            )

            model = PhantomQ0Model(params)

            singularity_type = model.classify_analytic()

            singular_time = None
            if hasattr(model, "singular_time"):
                singular_time = model.singular_time()

            t_values = _make_time_grid_for_model(
                model=model,
                t0=t0,
                t_end_default=t_end_default,
                n_points=n_points,
                singular_fraction=singular_fraction,
            )

            df_bg = model.evaluate_background_t(t_values)

            record = _summarize_background(
                df=df_bg,
                model_name="phantom_q0",
                params=params_dict,
                singularity_type=singularity_type,
                singular_time=singular_time,
                valid_solution=True,
                error_message="",
            )

        except Exception as e:
            record = {
                "model_name": "phantom_q0",
                **params_dict,
                "singularity_type": SingularityType.AMBIGUOUS.value,
                "singular_time": np.nan,
                "valid_solution": False,
                "error_message": str(e),
                "t_eval_min": np.nan,
                "t_eval_max": np.nan,
                "rho_initial": np.nan,
                "rho_final": np.nan,
                "rho_max": np.nan,
                "a_initial": np.nan,
                "a_final": np.nan,
                "a_max": np.nan,
                "H_initial": np.nan,
                "H_final": np.nan,
                "H_max": np.nan,
                "P_initial": np.nan,
                "P_final": np.nan,
                "P_abs_max": np.nan,
                "Hdot_initial": np.nan,
                "Hdot_final": np.nan,
                "Hdot_max": np.nan,
            }
            

        records.append(record)

    return pd.DataFrame(records)


def build_phantom_qalpha_dataset(
    A_values,
    nu_values,
    alpha_values,
    rho0_values=(1.0,),
    a0: float = 1.0,
    t0: float = 0.0,
    t_end_default: float = 10.0,
    n_points: int = 500,
    singular_fraction: float = 0.95,
) -> pd.DataFrame:
    """
    Construye un dataset para el modelo

        P = -rho - A*rho**nu,
        Q = alpha*rho.

    Parameters
    ----------
    A_values:
        Lista o arreglo de valores de A. Se espera A > 0.

    nu_values:
        Lista o arreglo de valores de nu.

    alpha_values:
        Lista o arreglo de valores de alpha. Se requiere 1 + alpha > 0.

    rho0_values:
        Lista o arreglo de densidades iniciales.

    a0:
        Factor de escala inicial.

    t0:
        Tiempo inicial.

    t_end_default:
        Tiempo máximo si no hay singularidad finita.

    n_points:
        Número de puntos temporales usados para resumir la evolución.

    singular_fraction:
        Fracción del tiempo singular usada cuando hay singularidad finita.

    Returns
    -------
    pandas.DataFrame
        Dataset resumido por combinación de parámetros.
    """

    records = []

    for A, nu, alpha, rho0 in product(
        A_values,
        nu_values,
        alpha_values,
        rho0_values,
    ):
        params_dict = {
            "A": float(A),
            "nu": float(nu),
            "alpha": float(alpha),
            "rho0": float(rho0),
            "a0": float(a0),
            "t0": float(t0),
        }

        try:
            params = PhantomQAlphaRhoParams(
                A=float(A),
                nu=float(nu),
                alpha=float(alpha),
                rho0=float(rho0),
                a0=float(a0),
                t0=float(t0),
            )

            model = PhantomQAlphaRhoModel(params)

            singularity_type = model.classify_analytic()
            singular_time = model.singular_time()

            t_values = _make_time_grid_for_model(
                model=model,
                t0=t0,
                t_end_default=t_end_default,
                n_points=n_points,
                singular_fraction=singular_fraction,
            )

            df_bg = model.evaluate_background_t(t_values)

            record = _summarize_background(
                df=df_bg,
                model_name="phantom_qalpha_rho",
                params=params_dict,
                singularity_type=singularity_type,
                singular_time=singular_time,
                valid_solution=True,
                error_message="",
            )

        except Exception as e:
            record = {
                "model_name": "phantom_qalpha_rho",
                **params_dict,
                "singularity_type": SingularityType.AMBIGUOUS.value,
                "singular_time": np.nan,
                "valid_solution": False,
                "error_message": str(e),
                "t_eval_min": np.nan,
                "t_eval_max": np.nan,
                "rho_initial": np.nan,
                "rho_final": np.nan,
                "rho_max": np.nan,
                "a_initial": np.nan,
                "a_final": np.nan,
                "a_max": np.nan,
                "H_initial": np.nan,
                "H_final": np.nan,
                "H_max": np.nan,
                "P_initial": np.nan,
                "P_final": np.nan,
                "P_abs_max": np.nan,
                "Hdot_initial": np.nan,
                "Hdot_final": np.nan,
                "Hdot_max": np.nan,
            }

        records.append(record)

    return pd.DataFrame(records)