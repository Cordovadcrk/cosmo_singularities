"""
models.py

Modelos cosmológicos utilizados en el proyecto de tesis.

En esta primera versión implementamos el modelo base:

    p = -rho - A*rho**nu,
    Q = 0.

El objetivo de este módulo es separar la física del modelo de la
clasificación de singularidades y de la generación de datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.special import lambertw

from src.classifiers import (
    classify_phantom_q0_by_nu,
    classify_phantom_qalpha_rho_by_nu,
)
from src.singularities import SingularityType
from src._analytic import evolving_density, power_increment, phantom_singular_time


@dataclass(frozen=True)
class PhantomQ0Params:
    """
    Parámetros del modelo phantom sin difusión.

    Modelo físico
    -------------
    p = -rho - A*rho**nu
    Q = 0

    Parámetros
    ----------
    A:
        Amplitud del término phantom. Para el caso que estamos estudiando,
        asumimos A > 0.

    nu:
        Exponente de la ecuación de estado inhomogénea.

    rho0:
        Densidad inicial. Debe ser rho0 > 0.

    a0:
        Factor de escala inicial. Usualmente se toma a0 = 1.

    t0:
        Tiempo inicial. Usualmente se toma t0 = 0.
    """

    A: float
    nu: float
    rho0: float
    a0: float = 1.0
    t0: float = 0.0

    def validate(self) -> None:
        """
        Valida que los parámetros tengan sentido físico básico.

        Raises
        ------
        ValueError
            Si algún parámetro no cumple las condiciones mínimas.
        """

        if not math.isfinite(self.A):
            raise ValueError("A debe ser un número finito.")

        if not math.isfinite(self.nu):
            raise ValueError("nu debe ser un número finito.")

        if not math.isfinite(self.rho0):
            raise ValueError("rho0 debe ser un número finito.")

        if not math.isfinite(self.a0):
            raise ValueError("a0 debe ser un número finito.")

        if not math.isfinite(self.t0):
            raise ValueError("t0 debe ser un número finito.")

        if self.A <= 0:
            raise ValueError("Para este modelo base se requiere A > 0.")

        if self.rho0 <= 0:
            raise ValueError("La densidad inicial debe cumplir rho0 > 0.")

        if self.a0 <= 0:
            raise ValueError("El factor de escala inicial debe cumplir a0 > 0.")


class PhantomQ0Model:
    """
    Modelo cosmológico phantom sin difusión.

    Modelo
    ------
    p = -rho - A*rho**nu
    Q = 0

    En unidades naturales con kappa^2 = 1 y universo plano, usamos:

        3H^2 = rho.

    Por tanto:

        H = sqrt(rho/3).

    Además, para relatividad general plana con un solo fluido:

        dot(H) = -1/2 (rho + p).

    Como

        rho + p = -A*rho**nu,

    entonces

        dot(H) = (A/2)*rho**nu.

    Esto es positivo para A > 0 y rho > 0, consistente con régimen phantom.
    """

    def __init__(self, params: PhantomQ0Params):
        params.validate()
        self.params = params

    def pressure(self, rho: float) -> float:
        """
        Calcula la presión P(rho).

        Parameters
        ----------
        rho:
            Densidad de energía positiva.

        Returns
        -------
        float
            Presión del fluido.
        """

        self._validate_rho(rho)
        A = self.params.A
        nu = self.params.nu

        return -rho - A * rho**nu

    def eos_w(self, rho: float) -> float:
        """
        Calcula el parámetro de estado efectivo w(rho) = P/rho.

        Para este modelo:

            w(rho) = -1 - A*rho**(nu - 1).

        Parameters
        ----------
        rho:
            Densidad de energía positiva.

        Returns
        -------
        float
            Parámetro de estado efectivo.
        """

        self._validate_rho(rho)
        return self.pressure(rho) / rho

    def hubble(self, rho: float) -> float:
        """
        Calcula H(rho) usando la ecuación de Friedmann plana:

            3H^2 = rho.

        Tomamos la rama expansiva H > 0.

        Parameters
        ----------
        rho:
            Densidad de energía positiva.

        Returns
        -------
        float
            Parámetro de Hubble.
        """

        self._validate_rho(rho)
        return math.sqrt(rho / 3.0)

    def rho_of_a(self, a: float) -> float:
        """Solución de d(rho)/dln(a)=3*A*rho**nu/c, con c=1.0."""
        self._validate_scale_factor(a)
        p = self.params
        return evolving_density(p.rho0, 1.0 - p.nu,
                                3.0 * p.A * (math.log(a) - math.log(p.a0)))

    def rho_of_t(self, t: float) -> float:
        """Solución exacta expansiva; estable alrededor de nu=1/2."""
        if not math.isfinite(t):
            raise ValueError("t debe ser un número finito.")
        p = self.params
        return evolving_density(p.rho0, 0.5 - p.nu,
                                math.sqrt(3.0) * p.A * (t - p.t0))

    def log_a_of_rho(self, rho: float) -> float:
        """ln(a) exacto, estable en nu=1 y sin exponenciar a enorme."""
        self._validate_rho(rho)
        p = self.params
        return math.log(p.a0) + power_increment(rho, p.rho0, 1.0 - p.nu) / (3.0 * p.A)

    def scale_factor_from_rho(self, rho: float) -> float:
        """a(rho); para escalas extremas usar log_a_of_rho."""
        return math.exp(self.log_a_of_rho(rho))

    def evaluate_background_t(self, t_values):
        """
        Evalúa las cantidades cosmológicas de fondo como funciones del tiempo.

        Modelo:
            P = -rho - A*rho**nu,
            Q = 0.

        Calcula:

            t,
            a(t),
            rho(t),
            Q(t),
            P(t),
            rho_eff(t),
            H(t),
            Hdot(t),
            w(t).

        Parameters
        ----------
        t_values:
            Valor o arreglo de valores de tiempo.

        Returns
        -------
        pandas.DataFrame
            Tabla con las cantidades cosmológicas evaluadas.
        """

        import numpy as np
        import pandas as pd

        t_array = np.asarray(t_values, dtype=float)

        if t_array.ndim == 0:
            t_array = t_array.reshape(1)

        if np.any(~np.isfinite(t_array)):
            raise ValueError("Todos los valores de t deben ser finitos.")

        rho_values = []
        a_values = []
        Q_values = []
        P_values = []
        rho_eff_values = []
        H_values = []
        Hdot_values = []
        w_values = []

        for t in t_array:
            rho = self.rho_of_t(float(t))
            a = self.scale_factor_from_rho(rho)

            Q = 0.0
            P = self.pressure(rho)
            rho_eff = rho + Q
            H = self.hubble(rho)
            Hdot = self.hdot(rho)
            w = self.eos_w(rho)

            rho_values.append(rho)
            a_values.append(a)
            Q_values.append(Q)
            P_values.append(P)
            rho_eff_values.append(rho_eff)
            H_values.append(H)
            Hdot_values.append(Hdot)
            w_values.append(w)

        return pd.DataFrame(
            {
                "t": t_array,
                "a": np.asarray(a_values),
                "rho": np.asarray(rho_values),
                "Q": np.asarray(Q_values),
                "P": np.asarray(P_values),
                "rho_eff": np.asarray(rho_eff_values),
                "H": np.asarray(H_values),
                "Hdot": np.asarray(Hdot_values),
                "w": np.asarray(w_values),
            }
        )

    @staticmethod
    def _validate_scale_factor(a: float) -> None:
        """
        Valida que a sea un factor de escala físico positivo y finito.
        """

        if not math.isfinite(a):
            raise ValueError("a debe ser un número finito.")

        if a <= 0:
            raise ValueError("El factor de escala debe cumplir a > 0.")

    def hdot(self, rho: float) -> float:
        """
        Calcula dot(H) como función de rho.

        En GR plana con un solo fluido:

            dot(H) = -1/2 (rho + P).

        Para P = -rho - A*rho**nu:

            rho + P = -A*rho**nu,

        por lo tanto:

            dot(H) = (A/2)*rho**nu.

        Parameters
        ----------
        rho:
            Densidad de energía positiva.

        Returns
        -------
        float
            Derivada temporal del parámetro de Hubble.
        """

        self._validate_rho(rho)
        A = self.params.A
        nu = self.params.nu

        return 0.5 * A * rho**nu

    def singular_time(self) -> float | None:
        """Tiempo singular futuro, o None para nu<=1/2."""
        p = self.params
        return phantom_singular_time(p.A, p.nu, p.rho0, p.t0, 1.0)

    def classify_analytic(self) -> SingularityType:
        """
        Clasifica analíticamente el modelo según el valor de nu.

        Returns
        -------
        SingularityType
            Tipo de singularidad esperado.
        """

        return classify_phantom_q0_by_nu(
            nu=self.params.nu,
            A=self.params.A,
        )

    @staticmethod
    def _validate_rho(rho: float) -> None:
        """
        Valida que rho sea una densidad física positiva y finita.
        """

        if not math.isfinite(rho):
            raise ValueError("rho debe ser un número finito.")

        if rho <= 0:
            raise ValueError("rho debe cumplir rho > 0.")
        
@dataclass(frozen=True)
class PhantomQAlphaRhoParams:
    """
    Parámetros del modelo phantom con difusión proporcional a la densidad.

    Modelo físico
    -------------
    p = -rho - A*rho**nu
    Q = alpha*rho

    Convención
    ----------
    3H**2 = rho + Q = (1 + alpha)*rho

    Parámetros
    ----------
    A:
        Amplitud del término phantom. Se asume A > 0.

    nu:
        Exponente de la ecuación de estado.

    alpha:
        Parámetro de difusión. Debe cumplir 1 + alpha > 0.

    rho0:
        Densidad inicial. Debe cumplir rho0 > 0.

    a0:
        Factor de escala inicial. Usualmente a0 = 1.

    t0:
        Tiempo inicial. Usualmente t0 = 0.
    """

    A: float
    nu: float
    alpha: float
    rho0: float
    a0: float = 1.0
    t0: float = 0.0

    def validate(self) -> None:
        if not math.isfinite(self.A):
            raise ValueError("A debe ser un número finito.")

        if not math.isfinite(self.nu):
            raise ValueError("nu debe ser un número finito.")

        if not math.isfinite(self.alpha):
            raise ValueError("alpha debe ser un número finito.")

        if not math.isfinite(self.rho0):
            raise ValueError("rho0 debe ser un número finito.")

        if not math.isfinite(self.a0):
            raise ValueError("a0 debe ser un número finito.")

        if not math.isfinite(self.t0):
            raise ValueError("t0 debe ser un número finito.")

        if self.A <= 0:
            raise ValueError("Para este modelo se requiere A > 0.")

        if 1.0 + self.alpha <= 0:
            raise ValueError("Se requiere 1 + alpha > 0 para que H sea real.")

        if self.rho0 <= 0:
            raise ValueError("La densidad inicial debe cumplir rho0 > 0.")

        if self.a0 <= 0:
            raise ValueError("El factor de escala inicial debe cumplir a0 > 0.")


class PhantomQAlphaRhoModel:
    """
    Modelo phantom con difusión unimodular proporcional a rho.

    Modelo
    ------
    p = -rho - A*rho**nu
    Q = alpha*rho

    Convención
    ----------
    3H**2 = rho + Q = (1 + alpha)*rho.

    La ecuación de continuidad unimodular es:

        dot(rho) + dot(Q) + 3H(rho + p) = 0.

    Como Q = alpha*rho:

        (1 + alpha)*dot(rho) + 3H(rho + p) = 0.

    Como rho + p = -A*rho**nu:

        (1 + alpha)*dot(rho) - 3A H rho**nu = 0.

    Por tanto:

        dot(rho) = 3A H rho**nu / (1 + alpha).
    """

    def __init__(self, params: PhantomQAlphaRhoParams):
        params.validate()
        self.params = params

    def diffusion(self, rho: float) -> float:
        """
        Calcula Q(rho) = alpha*rho.
        """

        self._validate_rho(rho)
        return self.params.alpha * rho

    def pressure(self, rho: float) -> float:
        """
        Calcula P(rho) = -rho - A*rho**nu.
        """

        self._validate_rho(rho)

        A = self.params.A
        nu = self.params.nu

        return -rho - A * rho**nu

    def eos_w(self, rho: float) -> float:
        """
        Calcula w(rho) = P/rho.
        """

        self._validate_rho(rho)
        return self.pressure(rho) / rho

    def hubble(self, rho: float) -> float:
        """
        Calcula H(rho) desde

            3H**2 = (1 + alpha)*rho.
        """

        self._validate_rho(rho)

        alpha = self.params.alpha

        return math.sqrt((1.0 + alpha) * rho / 3.0)

    def hdot(self, rho: float) -> float:
        """
        Calcula dot(H).

        A partir de:

            3H**2 = (1 + alpha)*rho,

        y de la ecuación de continuidad, se obtiene:

            dot(H) = (A/2)*rho**nu.

        Este resultado es independiente de alpha en esta parametrización.
        """

        self._validate_rho(rho)

        A = self.params.A
        nu = self.params.nu

        return 0.5 * A * rho**nu

    def rho_dot(self, rho: float) -> float:
        """
        Calcula dot(rho).

        Para este modelo:

            dot(rho) = sqrt(3)*A*rho**(nu + 1/2)/sqrt(1 + alpha).
        """

        self._validate_rho(rho)

        A = self.params.A
        nu = self.params.nu
        alpha = self.params.alpha

        return math.sqrt(3.0) * A * rho ** (nu + 0.5) / math.sqrt(1.0 + alpha)

    def rho_of_a(self, a: float) -> float:
        """Solución de d(rho)/dln(a)=3*A*rho**nu/c, con c=(1.0 + self.params.alpha)."""
        self._validate_scale_factor(a)
        p = self.params
        return evolving_density(p.rho0, 1.0 - p.nu,
                                3.0 * p.A / (1.0 + self.params.alpha) * (math.log(a) - math.log(p.a0)))

    def rho_of_t(self, t: float) -> float:
        """Solución exacta expansiva; estable alrededor de nu=1/2."""
        if not math.isfinite(t):
            raise ValueError("t debe ser un número finito.")
        p = self.params
        return evolving_density(p.rho0, 0.5 - p.nu,
                                math.sqrt(3.0) * p.A / math.sqrt((1.0 + self.params.alpha)) * (t - p.t0))

    def singular_time(self) -> float | None:
        """Tiempo singular futuro, o None para nu<=1/2."""
        p = self.params
        return phantom_singular_time(p.A, p.nu, p.rho0, p.t0, (1.0 + self.params.alpha))

    def log_a_of_rho(self, rho: float) -> float:
        """ln(a) exacto, estable en nu=1 y sin exponenciar a enorme."""
        self._validate_rho(rho)
        p = self.params
        return math.log(p.a0) + (1.0 + self.params.alpha) * power_increment(rho, p.rho0, 1.0 - p.nu) / (3.0 * p.A)

    def scale_factor_from_rho(self, rho: float) -> float:
        """a(rho); para escalas extremas usar log_a_of_rho."""
        return math.exp(self.log_a_of_rho(rho))

    def evaluate_background_t(self, t_values):
        """
        Evalúa las cantidades cosmológicas de fondo como funciones del tiempo.

        Modelo:
            P = -rho - A*rho**nu,
            Q = alpha*rho.

        Calcula:

            t,
            a(t),
            rho(t),
            Q(t),
            P(t),
            rho_eff(t),
            H(t),
            Hdot(t),
            w(t).

        Parameters
        ----------
        t_values:
            Valor o arreglo de valores de tiempo.

        Returns
        -------
        pandas.DataFrame
            Tabla con las cantidades cosmológicas evaluadas.
        """

        import numpy as np
        import pandas as pd

        t_array = np.asarray(t_values, dtype=float)

        if t_array.ndim == 0:
            t_array = t_array.reshape(1)

        if np.any(~np.isfinite(t_array)):
            raise ValueError("Todos los valores de t deben ser finitos.")

        rho_values = []
        a_values = []
        Q_values = []
        P_values = []
        rho_eff_values = []
        H_values = []
        Hdot_values = []
        w_values = []

        for t in t_array:
            rho = self.rho_of_t(float(t))
            a = self.scale_factor_from_rho(rho)

            Q = self.diffusion(rho)
            P = self.pressure(rho)
            rho_eff = rho + Q
            H = self.hubble(rho)
            Hdot = self.hdot(rho)
            w = self.eos_w(rho)

            rho_values.append(rho)
            a_values.append(a)
            Q_values.append(Q)
            P_values.append(P)
            rho_eff_values.append(rho_eff)
            H_values.append(H)
            Hdot_values.append(Hdot)
            w_values.append(w)

        return pd.DataFrame(
            {
                "t": t_array,
                "a": np.asarray(a_values),
                "rho": np.asarray(rho_values),
                "Q": np.asarray(Q_values),
                "P": np.asarray(P_values),
                "rho_eff": np.asarray(rho_eff_values),
                "H": np.asarray(H_values),
                "Hdot": np.asarray(Hdot_values),
                "w": np.asarray(w_values),
            }
        )

    def classify_analytic(self) -> SingularityType:
        """
        Clasifica analíticamente el modelo según nu.
        """

        return classify_phantom_qalpha_rho_by_nu(
            nu=self.params.nu,
            A=self.params.A,
            alpha=self.params.alpha,
        )

    @staticmethod
    def _validate_rho(rho: float) -> None:
        if not math.isfinite(rho):
            raise ValueError("rho debe ser un número finito.")

        if rho <= 0:
            raise ValueError("rho debe cumplir rho > 0.")

    @staticmethod
    def _validate_scale_factor(a: float) -> None:
        if not math.isfinite(a):
            raise ValueError("a debe ser un número finito.")

        if a <= 0:
            raise ValueError("El factor de escala debe cumplir a > 0.")
        
@dataclass(frozen=True)
class BarotropicPowerDiffusionParams:
    """
    Parámetros del modelo barotrópico con difusión tipo potencia.

    Modelo físico
    -------------
    p = omega*rho
    Q = alpha*rho**beta

    Convención
    ----------
    3H**2 = rho + Q = rho + alpha*rho**beta.

    Parámetros
    ----------
    omega:
        Parámetro de estado barotrópico. Para régimen phantom se considera
        omega < -1.

    alpha:
        Amplitud de la difusión.

    beta:
        Exponente de la difusión.

    rho0:
        Densidad inicial positiva.

    a0:
        Factor de escala inicial.

    t0:
        Tiempo inicial.
    """

    omega: float
    alpha: float
    beta: float
    rho0: float
    a0: float = 1.0
    t0: float = 0.0

    def validate(self) -> None:
        if not math.isfinite(self.omega):
            raise ValueError("omega debe ser un número finito.")

        if not math.isfinite(self.alpha):
            raise ValueError("alpha debe ser un número finito.")

        if not math.isfinite(self.beta):
            raise ValueError("beta debe ser un número finito.")

        if not math.isfinite(self.rho0):
            raise ValueError("rho0 debe ser un número finito.")

        if not math.isfinite(self.a0):
            raise ValueError("a0 debe ser un número finito.")

        if not math.isfinite(self.t0):
            raise ValueError("t0 debe ser un número finito.")

        if self.rho0 <= 0:
            raise ValueError("rho0 debe cumplir rho0 > 0.")

        if self.a0 <= 0:
            raise ValueError("a0 debe cumplir a0 > 0.")

        effective_density = self.rho0 + self.alpha * self.rho0**self.beta

        if effective_density <= 0:
            raise ValueError(
                "La densidad efectiva inicial rho0 + alpha*rho0**beta "
                "debe ser positiva para que H sea real."
            )


class BarotropicPowerDiffusionModel:
    """
    Modelo barotrópico con difusión de potencia.

    Modelo
    ------
    p = omega*rho
    Q = alpha*rho**beta

    Convención
    ----------
    3H**2 = rho + Q = rho + alpha*rho**beta.

    La continuidad unimodular es:

        dot(rho) + dot(Q) + 3H(rho + p) = 0.

    Como:

        Q = alpha*rho**beta,

    entonces:

        dot(Q) = alpha*beta*rho**(beta-1)*dot(rho).

    Por tanto:

        [1 + alpha*beta*rho**(beta-1)]*dot(rho)
        + 3H(1 + omega)rho = 0.

    Luego:

        dot(rho) =
        -3H(1 + omega)rho /
        [1 + alpha*beta*rho**(beta-1)].
    """

    def __init__(self, params: BarotropicPowerDiffusionParams):
        params.validate()
        self.params = params

    def F(self, rho):
        """
        Factor asociado al dominio de Friedmann:

            F(rho) = 1 + alpha*rho**(beta - 1).

        La ecuación de Friedmann puede escribirse como:

            3H**2 = rho*F(rho).

        Para que H sea real debe cumplirse F(rho) >= 0.
        """

        rho_array = np.asarray(rho, dtype=float)
        scalar_input = rho_array.ndim == 0

        if np.any(~np.isfinite(rho_array)):
            raise ValueError(
                "Todos los valores de rho deben ser finitos."
            )

        if np.any(rho_array <= 0.0):
            raise ValueError(
                "Todos los valores de rho deben ser positivos."
            )

        values = (
            1.0
            + self.params.alpha
            * rho_array ** (self.params.beta - 1.0)
        )

        if scalar_input:
            return float(values)

        return values

    def D(self, rho):
        """
        Factor asociado a la ecuación de continuidad:

            D(rho) = 1 + alpha*beta*rho**(beta - 1).

        La evolución de la densidad satisface:

            dot(rho)
            =
            -3H(1 + omega)rho / D(rho).

        Si D(rho) = 0, dot(rho) puede divergir.
        """

        rho_array = np.asarray(rho, dtype=float)
        scalar_input = rho_array.ndim == 0

        if np.any(~np.isfinite(rho_array)):
            raise ValueError(
                "Todos los valores de rho deben ser finitos."
            )

        if np.any(rho_array <= 0.0):
            raise ValueError(
                "Todos los valores de rho deben ser positivos."
            )

        values = (
            1.0
            + self.params.alpha
            * self.params.beta
            * rho_array ** (self.params.beta - 1.0)
        )

        if scalar_input:
            return float(values)

        return values        

    def signed_hubble(
        self,
        rho,
        branch: float = 1.0,
    ):
        """
        Evalúa el parámetro de Hubble permitiendo escoger su signo.

        Parameters
        ----------
        rho:
            Densidad positiva.

        branch:
            Signo de la rama cosmológica:

            branch = +1:
                expansión.

            branch = -1:
                contracción.

        Returns
        -------
        float or numpy.ndarray
            H(rho) con el signo indicado.
        """

        if branch not in (-1.0, 1.0):
            raise ValueError(
                "branch debe ser +1.0 o -1.0."
            )

        rho_array = np.asarray(rho, dtype=float)
        scalar_input = rho_array.ndim == 0

        F_values = np.asarray(
            self.F(rho_array),
            dtype=float,
        )

        radicand = (
            rho_array
            * F_values
            / 3.0
        )

        if np.any(radicand < -1e-13):
            raise ValueError(
                "Se intentó evaluar H fuera del dominio "
                "rho*F(rho) >= 0."
            )

        H_values = (
            branch
            * np.sqrt(
                np.maximum(
                    radicand,
                    0.0,
                )
            )
        )

        if scalar_input:
            return float(H_values)

        return H_values

    def log_a_of_rho(self, rho):
        """
        Calcula ln(a) a partir de la relación integrada entre
        la densidad y el factor de escala.

        Para beta != 1:

            ln(rho/rho0)
            + alpha*beta/(beta - 1)
              * (rho**(beta - 1) - rho0**(beta - 1))
            + 3(1 + omega)*ln(a/a0)
            = 0.

        Para beta = 1:

            (1 + alpha)*ln(rho/rho0)
            + 3(1 + omega)*ln(a/a0)
            = 0.

        Se retorna ln(a) para evitar overflow cuando a es
        extremadamente grande.
        """

        rho_array = np.asarray(rho, dtype=float)
        scalar_input = rho_array.ndim == 0

        if np.any(~np.isfinite(rho_array)):
            raise ValueError(
                "Todos los valores de rho deben ser finitos."
            )

        if np.any(rho_array <= 0.0):
            raise ValueError(
                "Todos los valores de rho deben ser positivos."
            )

        omega = self.params.omega
        alpha = self.params.alpha
        beta = self.params.beta
        rho0 = self.params.rho0
        a0 = self.params.a0

        if math.isclose(
            1.0 + omega,
            0.0,
            abs_tol=1e-14,
            rel_tol=0.0,
        ):
            raise ValueError(
                "Para omega=-1, rho es constante y no parametriza a. "
                "Use solve_background_t para la evolución de de Sitter."
            )

        if math.isclose(
            beta,
            1.0,
            abs_tol=1e-14,
            rel_tol=0.0,
        ):
            conservation_term = (
                (1.0 + alpha)
                * np.log(
                    rho_array / rho0
                )
            )

        else:
            conservation_term = (
                np.log(
                    rho_array / rho0
                )
                + (
                    alpha
                    * beta
                    / (beta - 1.0)
                )
                * (
                    rho_array ** (beta - 1.0)
                    - rho0 ** (beta - 1.0)
                )
            )

        values = (
            np.log(a0)
            - conservation_term
            / (
                3.0
                * (1.0 + omega)
            )
        )

        if scalar_input:
            return float(values)

        return values

    def diffusion(self, rho: float) -> float:
        """
        Calcula Q(rho) = alpha*rho**beta.
        """

        self._validate_rho(rho)

        alpha = self.params.alpha
        beta = self.params.beta

        return alpha * rho**beta

    def pressure(self, rho: float) -> float:
        """
        Calcula P(rho) = omega*rho.
        """

        self._validate_rho(rho)
        return self.params.omega * rho

    def eos_w(self, rho: float) -> float:
        """
        Devuelve omega.

        Se deja como función de rho para mantener interfaz común con otros
        modelos.
        """

        self._validate_rho(rho)
        return self.params.omega

    def effective_density(self, rho: float) -> float:
        """
        Calcula rho_eff = rho + Q.
        """

        self._validate_rho(rho)
        return rho + self.diffusion(rho)

    def hubble(self, rho: float) -> float:
        """
        Calcula H(rho) desde:

            3H**2 = rho + alpha*rho**beta.

        Se toma la rama expansiva H > 0.
        """

        rho_eff = self.effective_density(rho)

        if rho_eff <= 0:
            raise ValueError(
                "rho + Q debe ser positivo para que H sea real."
            )

        return math.sqrt(rho_eff / 3.0)

    def hdot(self, rho: float) -> float:
        """
        Calcula dot(H).

        A partir de las ecuaciones de Friedmann unimodulares:

            3H**2 = rho + Q,

            2dot(H) + 3H**2 = -p + Q,

        restando ambas:

            2dot(H) = -(rho + p).

        Como p = omega*rho:

            dot(H) = -1/2*(1 + omega)*rho.

        Para omega < -1, dot(H) > 0.
        """

        self._validate_rho(rho)

        omega = self.params.omega

        return -0.5 * (1.0 + omega) * rho

    def rho_dot(self, rho: float) -> float:
        """
        Calcula dot(rho) para integración numérica.

        dot(rho) =
        -3H(1 + omega)rho /
        [1 + alpha*beta*rho**(beta-1)].

        Si el denominador se acerca a cero, el modelo puede presentar
        una singularidad dinámica o una región no bien definida.
        """

        self._validate_rho(rho)

        omega = self.params.omega
        alpha = self.params.alpha
        beta = self.params.beta

        denominator = 1.0 + alpha * beta * rho ** (beta - 1.0)

        if math.isclose(denominator, 0.0, abs_tol=1e-12, rel_tol=0.0):
            raise ValueError(
                "El denominador de dot(rho) se anuló: "
                "1 + alpha*beta*rho**(beta-1) = 0."
            )

        H = self.hubble(rho)

        return -3.0 * H * (1.0 + omega) * rho / denominator
    
    def solve_background_t(
        self,
        t_end: float,
        n_points: int = 2000,
        *,
        rho_min: float = 1e-12,
        rho_max: float = 1e12,
        a_max: float = 1e12,
        boundary_tol: float = 1e-8,
        method: str = "DOP853",
        rtol: float = 1e-9,
        atol: float = 1e-11,
        max_step: float | None = None,
    ) -> pd.DataFrame:
        """
        Resuelve numéricamente la evolución temporal del modelo

            p = omega*rho,
            Q = alpha*rho**beta.

        Se integran las variables logarítmicas

            x = ln(rho),
            N = ln(a),

        para garantizar numéricamente que rho > 0 y a > 0.

        El sistema es

            dx/dt =
            -3*H*(1 + omega)
            / [1 + alpha*beta*rho**(beta - 1)],

            dN/dt = H,

        con

            H = sqrt[(rho + alpha*rho**beta)/3].

        El DataFrame resultante contiene únicamente

            t, rho, a, H, Hdot.

        La información sobre la detención del integrador se guarda
        en result.attrs.
        """

        # ---------------------------------------------------------
        # 1. Parámetros y validaciones
        # ---------------------------------------------------------

        t0 = self.params.t0
        rho0 = self.params.rho0
        a0 = self.params.a0

        omega = self.params.omega
        alpha = self.params.alpha
        beta = self.params.beta

        if not math.isfinite(t_end):
            raise ValueError("t_end debe ser finito.")

        if t_end <= t0:
            raise ValueError(
                "Para estudiar la evolución futura se requiere t_end > t0."
            )

        if not isinstance(n_points, int) or n_points < 2:
            raise ValueError(
                "n_points debe ser un entero mayor o igual que 2."
            )

        if not math.isfinite(rho_min) or rho_min <= 0:
            raise ValueError("rho_min debe ser positivo y finito.")

        if rho_min >= rho0:
            raise ValueError("rho_min debe ser menor que rho0.")

        if not math.isfinite(rho_max) or rho_max <= rho0:
            raise ValueError(
                "rho_max debe ser finito y mayor que rho0."
            )

        if not math.isfinite(a_max) or a_max <= a0:
            raise ValueError(
                "a_max debe ser finito y mayor que a0."
            )

        if not math.isfinite(boundary_tol) or boundary_tol <= 0:
            raise ValueError(
                "boundary_tol debe ser positivo y finito."
            )

        if rtol <= 0 or atol <= 0:
            raise ValueError("rtol y atol deben ser positivos.")

        if max_step is not None:
            if not math.isfinite(max_step) or max_step <= 0:
                raise ValueError(
                    "max_step debe ser positivo y finito."
                )

        # ---------------------------------------------------------
        # 2. Condiciones iniciales logarítmicas
        # ---------------------------------------------------------

        log_rho0 = math.log(rho0)
        log_a0 = math.log(a0)

        log_rho_min = math.log(rho_min)
        log_rho_max = math.log(rho_max)
        log_a_max = math.log(a_max)

        initial_power = rho0 ** (beta - 1.0)

        initial_domain = (
            1.0
            + alpha * initial_power
        )

        initial_denominator = (
            1.0
            + alpha * beta * initial_power
        )

        if initial_domain <= boundary_tol:
            raise ValueError(
                "La condición inicial no pertenece al dominio físico. "
                "Se requiere "
                "1 + alpha*rho0**(beta-1) > boundary_tol."
            )

        if abs(initial_denominator) <= boundary_tol:
            raise ValueError(
                "La condición inicial está demasiado cerca de "
                "1 + alpha*beta*rho0**(beta-1) = 0."
            )

        # ---------------------------------------------------------
        # 3. Reconstrucción de cantidades físicas
        # ---------------------------------------------------------

        def state_quantities(state):
            """
            Reconstruye rho y los dos factores que controlan
            el dominio dinámico.
            """

            log_rho = state[0]

            rho = math.exp(log_rho)

            rho_power = rho ** (beta - 1.0)

            domain_factor = (
                1.0
                + alpha * rho_power
            )

            denominator = (
                1.0
                + alpha * beta * rho_power
            )

            return rho, domain_factor, denominator

        # ---------------------------------------------------------
        # 4. Sistema de ecuaciones diferenciales
        # ---------------------------------------------------------

        def differential_system(_t, state):
            """
            Sistema para state = [ln(rho), ln(a)].
            """

            rho, domain_factor, denominator = (
                state_quantities(state)
            )

            # solve_ivp puede evaluar muy ligeramente fuera del
            # dominio mientras localiza la raíz de un evento.
            safe_domain = max(
                domain_factor,
                np.finfo(float).tiny,
            )

            if abs(denominator) < boundary_tol:
                sign = denominator if denominator != 0 else 1.0

                safe_denominator = math.copysign(
                    boundary_tol,
                    sign,
                )
            else:
                safe_denominator = denominator

            H = math.sqrt(
                rho * safe_domain / 3.0
            )

            dlog_rho_dt = (
                -3.0
                * H
                * (1.0 + omega)
                / safe_denominator
            )

            dlog_a_dt = H

            return np.array(
                [dlog_rho_dt, dlog_a_dt],
                dtype=float,
            )

        # ---------------------------------------------------------
        # 5. Eventos de detención
        # ---------------------------------------------------------

        def event_rho_min(_t, state):
            return state[0] - log_rho_min

        event_rho_min.terminal = True
        event_rho_min.direction = -1

        def event_rho_max(_t, state):
            return log_rho_max - state[0]

        event_rho_max.terminal = True
        event_rho_max.direction = -1

        def event_a_max(_t, state):
            return log_a_max - state[1]

        event_a_max.terminal = True
        event_a_max.direction = -1

        def event_effective_density_boundary(_t, state):
            _, domain_factor, _ = state_quantities(state)

            return domain_factor - boundary_tol

        event_effective_density_boundary.terminal = True
        event_effective_density_boundary.direction = -1

        def event_denominator_boundary(_t, state):
            _, _, denominator = state_quantities(state)

            return abs(denominator) - boundary_tol

        event_denominator_boundary.terminal = True
        event_denominator_boundary.direction = -1

        events = [
            event_rho_min,
            event_rho_max,
            event_a_max,
            event_effective_density_boundary,
            event_denominator_boundary,
        ]

        event_names = [
            "rho_min",
            "rho_max",
            "a_max",
            "effective_density_boundary",
            "rho_dot_denominator_boundary",
        ]

        # ---------------------------------------------------------
        # 6. Integración
        # ---------------------------------------------------------

        solution = solve_ivp(
            fun=differential_system,
            t_span=(t0, t_end),
            y0=np.array(
                [log_rho0, log_a0],
                dtype=float,
            ),
            method=method,
            dense_output=True,
            events=events,
            rtol=rtol,
            atol=atol,
            max_step=(
                np.inf
                if max_step is None
                else max_step
            ),
        )

        if not solution.success:
            raise RuntimeError(
                "El integrador no logró completar la evolución: "
                f"{solution.message}"
            )

        # ---------------------------------------------------------
        # 7. Causa y tiempo de detención
        # ---------------------------------------------------------

        termination_event = None
        termination_time = float(solution.t[-1])

        detected_events = []

        for event_times, event_name in zip(
            solution.t_events,
            event_names,
        ):
            if len(event_times) > 0:
                detected_events.append(
                    (float(event_times[0]), event_name)
                )

        if detected_events:
            termination_time, termination_event = min(
                detected_events,
                key=lambda item: item[0],
            )

        # ---------------------------------------------------------
        # 8. Reconstrucción sobre una grilla uniforme
        # ---------------------------------------------------------

        t_values = np.linspace(
            t0,
            termination_time,
            n_points,
        )

        log_rho_values, log_a_values = solution.sol(
            t_values
        )

        rho_values = np.exp(log_rho_values)
        a_values = np.exp(log_a_values)

        rho_power_values = (
            rho_values ** (beta - 1.0)
        )

        domain_values = (
            1.0
            + alpha * rho_power_values
        )

        if np.any(domain_values <= 0):
            raise RuntimeError(
                "La reconstrucción alcanzó una región con "
                "rho + Q <= 0."
            )

        H_values = np.sqrt(
            rho_values * domain_values / 3.0
        )

        Hdot_values = (
            -0.5
            * (1.0 + omega)
            * rho_values
        )

        result = pd.DataFrame(
            {
                "t": t_values,
                "rho": rho_values,
                "a": a_values,
                "H": H_values,
                "Hdot": Hdot_values,
            }
        )

        # ---------------------------------------------------------
        # 9. Metadatos del solucionador
        # ---------------------------------------------------------

        result.attrs["solver_success"] = solution.success
        result.attrs["solver_message"] = solution.message

        result.attrs["termination_event"] = (
            termination_event
        )

        result.attrs["termination_time"] = (
            termination_time
        )

        result.attrs["requested_t_end"] = t_end

        result.attrs["reached_t_end"] = (
            termination_event is None
        )

        result.attrs["method"] = method
        result.attrs["rtol"] = rtol
        result.attrs["atol"] = atol

        result.attrs["rho_min"] = rho_min
        result.attrs["rho_max"] = rho_max
        result.attrs["a_max"] = a_max
        result.attrs["boundary_tol"] = boundary_tol

        return result
    
    def solve_signed_background_t(
        self,
        t_end: float,
        n_points: int = 2000,
        *,
        rho_min: float = 1e-14,
        rho_max: float = 1e12,
        a_min: float = 1e-16,
        a_max: float = 1e80,
        boundary_tol: float = 1e-8,
        method: str = "DOP853",
        rtol: float = 1e-10,
        atol: float = 1e-12,
        max_step: float | None = 0.01,
    ) -> pd.DataFrame:
        """
        Integra rho, a y H tratando H como una variable
        dinámica con signo.

        Variables:

            x = ln(rho),
            N = ln(a),
            H.

        Sistema:

            dot(x)
            =
            -3H(1 + omega)/D(rho),

            dot(N) = H,

            dot(H)
            =
            -(1 + omega)rho/2.

        Este método permite continuar una solución después
        de H=0 y estudiar transiciones de expansión a
        contracción.
        """

        params = self.params

        if not math.isfinite(t_end):
            raise ValueError(
                "t_end debe ser finito."
            )

        if t_end <= params.t0:
            raise ValueError(
                "t_end debe ser mayor que t0."
            )

        if n_points < 2:
            raise ValueError(
                "n_points debe ser al menos 2."
            )

        if not (
            0.0
            < rho_min
            < params.rho0
            < rho_max
        ):
            raise ValueError(
                "Debe cumplirse "
                "0 < rho_min < rho0 < rho_max."
            )

        if not (
            0.0
            < a_min
            < params.a0
            < a_max
        ):
            raise ValueError(
                "Debe cumplirse "
                "0 < a_min < a0 < a_max."
            )

        if boundary_tol <= 0.0:
            raise ValueError(
                "boundary_tol debe ser positiva."
            )

        if rtol <= 0.0 or atol <= 0.0:
            raise ValueError(
                "rtol y atol deben ser positivos."
            )

        if max_step is None:
            max_step_value = np.inf
        else:
            if max_step <= 0.0:
                raise ValueError(
                    "max_step debe ser positivo."
                )

            max_step_value = max_step

        F0 = self.F(params.rho0)
        D0 = self.D(params.rho0)

        if F0 <= 0.0:
            raise ValueError(
                "El estado inicial no satisface F0 > 0. "
                f"F0={F0:.6e}."
            )

        if abs(D0) <= boundary_tol:
            raise ValueError(
                "El estado inicial está demasiado cerca "
                "de D(rho)=0. "
                f"D0={D0:.6e}."
            )

        H0 = self.hubble(
            params.rho0
        )

        sign_D0 = float(
            np.sign(D0)
        )

        initial_state = np.array(
            [
                np.log(params.rho0),
                np.log(params.a0),
                H0,
            ],
            dtype=float,
        )

        def rhs(t, state):
            rho = np.exp(
                state[0]
            )

            H = state[2]
            D_value = self.D(rho)

            return np.array(
                [
                    -3.0
                    * H
                    * (1.0 + params.omega)
                    / D_value,
                    H,
                    -0.5
                    * (1.0 + params.omega)
                    * rho,
                ],
                dtype=float,
            )

        log_rho_min = np.log(
            rho_min
        )

        log_rho_max = np.log(
            rho_max
        )

        log_a_min = np.log(
            a_min
        )

        log_a_max = np.log(
            a_max
        )

        def event_rho_min(t, state):
            return (
                state[0]
                - log_rho_min
            )

        def event_rho_max(t, state):
            return (
                log_rho_max
                - state[0]
            )

        def event_a_min(t, state):
            return (
                state[1]
                - log_a_min
            )

        def event_a_max(t, state):
            return (
                log_a_max
                - state[1]
            )

        def event_rho_dot_denominator_boundary(
            t,
            state,
        ):
            rho = np.exp(
                state[0]
            )

            return (
                sign_D0
                * self.D(rho)
                - boundary_tol
            )

        def event_H_zero(t, state):
            return state[2]

        terminal_events = [
            event_rho_min,
            event_rho_max,
            event_a_min,
            event_a_max,
            event_rho_dot_denominator_boundary,
        ]

        for event in terminal_events:
            event.terminal = True
            event.direction = -1

        event_H_zero.terminal = False
        event_H_zero.direction = -1

        events = (
            terminal_events
            + [event_H_zero]
        )

        solution = solve_ivp(
            rhs,
            (
                params.t0,
                t_end,
            ),
            initial_state,
            method=method,
            dense_output=True,
            events=events,
            rtol=rtol,
            atol=atol,
            max_step=max_step_value,
        )

        event_names = [
            "rho_min",
            "rho_max",
            "a_min",
            "a_max",
            "rho_dot_denominator_boundary",
        ]

        termination_event = None
        termination_time = None

        for event_name, event_times in zip(
            event_names,
            solution.t_events[:5],
        ):
            if len(event_times) == 0:
                continue

            candidate_time = float(
                event_times[0]
            )

            if (
                termination_time is None
                or candidate_time < termination_time
            ):
                termination_event = event_name
                termination_time = candidate_time

        if termination_time is None:
            termination_time = float(
                solution.t[-1]
            )

        t_grid = np.linspace(
            params.t0,
            termination_time,
            n_points,
        )

        state_grid = solution.sol(
            t_grid
        )

        rho_values = np.exp(
            state_grid[0]
        )

        a_values = np.exp(
            state_grid[1]
        )

        H_values = state_grid[2]

        Hdot_values = (
            -0.5
            * (1.0 + params.omega)
            * rho_values
        )

        F_values = np.asarray(
            self.F(rho_values),
            dtype=float,
        )

        constraint_left = (
            3.0
            * H_values**2
        )

        constraint_right = (
            rho_values
            * F_values
        )

        constraint_residual = (
            constraint_left
            - constraint_right
        ) / np.maximum(
            np.abs(constraint_left)
            + np.abs(constraint_right),
            1e-300,
        )

        result = pd.DataFrame(
            {
                "t": t_grid,
                "rho": rho_values,
                "a": a_values,
                "H": H_values,
                "Hdot": Hdot_values,
                "constraint_residual": (
                    constraint_residual
                ),
            }
        )

        H_zero_times = [
            float(value)
            for value in solution.t_events[5]
        ]

        result.attrs.update(
            {
                "solver_success": bool(
                    solution.success
                ),
                "solver_message": (
                    solution.message
                ),
                "termination_event": (
                    termination_event
                ),
                "termination_time": (
                    termination_time
                ),
                "requested_t_end": t_end,
                "reached_t_end": bool(
                    termination_event is None
                    and np.isclose(
                        termination_time,
                        t_end,
                    )
                ),
                "H_zero_times": (
                    H_zero_times
                ),
                "maximum_constraint_residual": float(
                    np.max(
                        np.abs(
                            constraint_residual
                        )
                    )
                ),
                "method": method,
                "rtol": rtol,
                "atol": atol,
                "rho_min": rho_min,
                "rho_max": rho_max,
                "a_min": a_min,
                "a_max": a_max,
                "boundary_tol": boundary_tol,
            }
        )

        return result

    def rho_of_a_lambert(
        self,
        a_values,
        branch: int | str = "auto",
        atol_imag: float = 1e-10,
    ):
        """
        Calcula rho(a) usando la solución analítica con la función de Lambert W.

        Modelo
        ------
        P = omega*rho
        Q = alpha*rho**beta

        La solución para beta != 1 y alpha*beta != 0 es:

            rho(a) = [
                W(alpha*beta*C1*a^{-3(beta-1)(1+omega)})
                /
                (alpha*beta)
            ]^{1/(beta-1)}

        donde

            C1 = rho0^{beta-1}
                 exp(alpha*beta*rho0^{beta-1})
                 a0^{3(beta-1)(1+omega)}.

        Parameters
        ----------
        a_values:
            Valor o arreglo de valores del factor de escala.

        branch:
            Rama de la función de Lambert W.

            - "auto":
                Escoge la rama que reproduce la condición inicial.
                Si alpha*beta*rho0^{beta-1} >= -1 usa rama 0.
                Si alpha*beta*rho0^{beta-1} < -1 usa rama -1.

            - 0:
                Rama principal de Lambert W.

            - -1:
                Rama real secundaria, útil cuando el argumento está en
                [-1/e, 0).

        atol_imag:
            Tolerancia para aceptar que la parte imaginaria numérica sea cero.

        Returns
        -------
        float or numpy.ndarray
            rho(a). Si a_values es escalar devuelve float. Si es arreglo,
            devuelve numpy.ndarray.

        Notes
        -----
        Casos especiales:

        - Si alpha = 0, entonces Q = 0 y se recupera:

              rho(a) = rho0*(a/a0)^[-3(1+omega)].

        - Si beta = 1, entonces Q = alpha*rho y:

              rho(a) = rho0*(a/a0)^[-3(1+omega)/(1+alpha)].

        - Si alpha*beta = 0, la fórmula de Lambert W no debe usarse.
        """

        a_array = np.asarray(a_values, dtype=float)
        scalar_input = a_array.ndim == 0

        if scalar_input:
            a_array = a_array.reshape(1)

        if np.any(~np.isfinite(a_array)):
            raise ValueError("Todos los valores de a deben ser finitos.")

        if np.any(a_array <= 0):
            raise ValueError("Todos los valores de a deben cumplir a > 0.")

        omega = self.params.omega
        alpha = self.params.alpha
        beta = self.params.beta
        rho0 = self.params.rho0
        a0 = self.params.a0

        # Caso especial alpha = 0: no hay difusión.
        if alpha == 0.0 or beta == 0.0:
            rho = rho0 * (a_array / a0) ** (-3.0 * (1.0 + omega))
            return float(rho[0]) if scalar_input else rho

        # Caso especial beta = 1: Q = alpha*rho.
        if math.isclose(beta, 1.0, abs_tol=1e-14, rel_tol=0.0):
            if math.isclose(1.0 + alpha, 0.0, abs_tol=1e-14, rel_tol=0.0):
                raise ValueError(
                    "Para beta = 1 se requiere 1 + alpha != 0."
                )

            rho = rho0 * (a_array / a0) ** (
                -3.0 * (1.0 + omega) / (1.0 + alpha)
            )

            return float(rho[0]) if scalar_input else rho

        # Caso general con Lambert W.
        alpha_beta = alpha * beta

        if math.isclose(alpha_beta, 0.0, abs_tol=1e-14, rel_tol=0.0):
            raise ValueError(
                "La fórmula de Lambert W requiere alpha*beta != 0. "
                "Usa los casos especiales alpha = 0 o beta = 0."
            )

        y0 = rho0 ** (beta - 1.0)

        C1 = (
            y0
            * math.exp(alpha_beta * y0)
            * a0 ** (3.0 * (beta - 1.0) * (1.0 + omega))
        )

        exponent_a = -3.0 * (beta - 1.0) * (1.0 + omega)

        argument = alpha_beta * C1 * a_array**exponent_a

        # Selección de rama.
        if branch == "auto":
            initial_x = alpha_beta * y0

            if initial_x >= -1.0:
                selected_branch = 0
            else:
                selected_branch = -1
        elif branch in (0, -1):
            selected_branch = int(branch)
        else:
            raise ValueError("branch debe ser 'auto', 0 o -1.")

        W = lambertw(argument, k=selected_branch)

        if np.any(np.abs(W.imag) > atol_imag):
            raise ValueError(
                "Lambert W devolvió valores complejos no despreciables. "
                "Revisa la rama elegida o el rango de a."
            )

        W_real = W.real

        # scipy's W_-1 can lose the distance to -1 near the branch point.
        # Evaluate its local expansion with a stable logarithmic argument.
        initial_x = alpha_beta * y0
        if initial_x < 0:
            delta = initial_x + 1.0
            log_distance = (np.log1p(-delta) + delta
                            + exponent_a * np.log(a_array / a0))
            near = (log_distance <= 0) & (log_distance > -1e-8)
            if np.any(near):
                p = np.sqrt(-2.0 * np.expm1(log_distance[near]))
                if selected_branch == -1:
                    p = -p
                W_real = W_real.copy()
                W_real[near] = -1+p-p**2/3+11*p**3/72-43*p**4/540+769*p**5/17280

        y = W_real / alpha_beta

        if np.any(~np.isfinite(y)):
            raise ValueError("La solución produjo valores no finitos.")

        if np.any(y <= 0):
            raise ValueError(
                "La solución produjo rho^{beta-1} <= 0, lo cual no permite "
                "obtener una densidad real positiva para estos parámetros."
            )

        rho = y ** (1.0 / (beta - 1.0))
        rho = np.where(a_array == a0, rho0, rho)

        if np.any(~np.isfinite(rho)):
            raise ValueError("La solución rho(a) produjo valores no finitos.")

        if np.any(rho <= 0):
            raise ValueError("La solución rho(a) produjo densidades no positivas.")

        return float(rho[0]) if scalar_input else rho
    
    def rho_of_a(self, a_values):
        return self.rho_of_a_lambert(a_values)

    def time_of_a(self, a_values):
        """
        Calcula t(a) usando

            H = dot(a)/a,

        por lo tanto

            dt/da = 1/(a H(a)).

        Esta función asume que a_values está ordenado de menor a mayor
        y que el primer valor corresponde a a0 o está muy cerca de a0.

        Parameters
        ----------
        a_values:
            Arreglo de valores del factor de escala.

        Returns
        -------
        numpy.ndarray
            Valores de tiempo cósmico t(a).
        """

        a_array = np.asarray(a_values, dtype=float)

        if a_array.ndim == 0:
            a_array = a_array.reshape(1)

        if np.any(~np.isfinite(a_array)):
            raise ValueError("Todos los valores de a deben ser finitos.")

        if np.any(a_array <= 0):
            raise ValueError("Todos los valores de a deben cumplir a > 0.")

        if np.any(np.diff(a_array) < 0):
            raise ValueError(
                "Para calcular t(a), a_values debe estar ordenado de menor a mayor."
            )

        a0 = self.params.a0

        if not np.isclose(a_array[0], a0, atol=1e-10, rtol=1e-10):
            raise ValueError(
                "Para esta versión, el primer valor de a_values debe coincidir "
                "con a0. Por ejemplo, usa a_values = np.logspace(0, 3, 1000) "
                "si a0 = 1."
            )

        rho_values = self.rho_of_a(a_array)

        Q_values = self.params.alpha * rho_values**self.params.beta

        rho_eff_values = rho_values + Q_values

        if np.any(rho_eff_values <= 0):
            raise ValueError(
                "rho_eff = rho + Q debe ser positiva para calcular H(a)."
            )

        H_values = np.sqrt(rho_eff_values / 3.0)

        integrand = 1.0 / (a_array * H_values)

        t_values = self.params.t0 + cumulative_trapezoid(
            integrand,
            a_array,
            initial=0.0,
        )

        return t_values

    def evaluate_background(self, a_values):
        """
        Evalúa las cantidades cosmológicas de fondo para el modelo

            P = omega*rho,
            Q = alpha*rho**beta.

        Calcula:

            a,
            t(a),
            rho(a),
            Q(a),
            P(a),
            rho_eff(a),
            H(a),
            Hdot(a).

        Parameters
        ----------
        a_values:
            Arreglo de valores del factor de escala.

        Returns
        -------
        pandas.DataFrame
            Tabla con las cantidades cosmológicas evaluadas.
        """

        a_array = np.asarray(a_values, dtype=float)

        if a_array.ndim == 0:
            a_array = a_array.reshape(1)

        if np.any(~np.isfinite(a_array)):
            raise ValueError("Todos los valores de a deben ser finitos.")

        if np.any(a_array <= 0):
            raise ValueError("Todos los valores de a deben cumplir a > 0.")

        rho_values = self.rho_of_a(a_array)

        Q_values = self.params.alpha * rho_values**self.params.beta

        P_values = self.params.omega * rho_values

        rho_eff_values = rho_values + Q_values

        if np.any(rho_eff_values <= 0):
            raise ValueError(
                "rho_eff = rho + Q debe ser positiva para calcular H(a)."
            )

        H_values = np.sqrt(rho_eff_values / 3.0)

        Hdot_values = -0.5 * (1.0 + self.params.omega) * rho_values

        t_values = self.time_of_a(a_array)

        return pd.DataFrame(
            {
                "a": a_array,
                "t": t_values,
                "rho": rho_values,
                "Q": Q_values,
                "P": P_values,
                "rho_eff": rho_eff_values,
                "H": H_values,
                "Hdot": Hdot_values,
            }
        )

    def classify_analytic(self) -> SingularityType:
        """
        Clasificación analítica no implementada todavía.

        Este modelo depende de alpha, beta, omega y del dominio dinámico.
        Por ahora lo marcamos como ambiguo hasta implementar un diagnóstico
        numérico o reglas específicas.
        """

        return SingularityType.AMBIGUOUS

    @staticmethod
    def _validate_rho(rho: float) -> None:
        if not math.isfinite(rho):
            raise ValueError("rho debe ser un número finito.")

        if rho <= 0:
            raise ValueError("rho debe cumplir rho > 0.")

@dataclass(frozen=True)
class CosCoshParams:
    """
    Parámetros del modelo Cos-Cosh para la ecuación de estado dinámica.

    Modelo:
        w(a) = wa*(a/a0)*cos(a/a0)
             - wb*(a/a0)*cosh(a/a0)
             - wc

    La normalización usual es:
        a0 = 1
        rho0 = 1
        t0 = 0
        H0 = 1

    Si se quiere trabajar en unidades físicas, H0 debe entregarse en unidades
    inversas del tiempo elegido.
    """

    wa: float = 5.203
    wb: float = 0.188
    wc: float = 3.634
    rho0: float = 1.0
    a0: float = 1.0
    t0: float = 0.0
    H0: float = 1.0

    def validate(self) -> None:
        values = {
            "wa": self.wa,
            "wb": self.wb,
            "wc": self.wc,
            "rho0": self.rho0,
            "a0": self.a0,
            "t0": self.t0,
            "H0": self.H0,
        }

        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} debe ser finito.")

        if self.rho0 <= 0:
            raise ValueError("rho0 debe ser positivo.")

        if self.a0 <= 0:
            raise ValueError("a0 debe ser positivo.")

        if self.H0 <= 0:
            raise ValueError("H0 debe ser positivo.")


class CosCoshModel:
    """
    Modelo cosmológico con parametrización Cos-Cosh de la ecuación de estado.

    Se trabaja en régimen de dominación de energía oscura.

    La variable numérica principal es

        x = ln(a/a0)

    de modo que

        a(x) = a0 * exp(x).

    La evolución se reconstruye mediante:

        dG/dx = 1 + w(a)
        d tau/dx = exp(3G/2)

    donde tau = H0*(t - t0).
    """

    def __init__(self, params: CosCoshParams):
        params.validate()
        self.params = params

    def scale_factor_from_x(self, x):
        """
        Devuelve a(x) = a0*exp(x).
        """

        x_array = np.asarray(x, dtype=float)
        return self.params.a0 * np.exp(x_array)

    def eos_w(self, a):
        """
        Ecuación de estado w(a) del modelo Cos-Cosh.
        """

        a_array = np.asarray(a, dtype=float)

        if np.any(a_array <= 0):
            raise ValueError("El factor de escala a debe ser positivo.")

        u = a_array / self.params.a0

        return (
            self.params.wa * u * np.cos(u)
            - self.params.wb * u * np.cosh(u)
            - self.params.wc
        )

    def pressure(self, rho, a):
        """
        Presión p(a) = w(a)*rho(a).
        """

        return self.eos_w(a) * rho

    def density_from_G(self, G):
        """
        rho = rho0*exp(-3G).
        """

        return self.params.rho0 * np.exp(-3.0 * G)

    def hubble_from_G(self, G):
        """
        H = H0*exp(-3G/2).
        """

        return self.params.H0 * np.exp(-1.5 * G)

    def evaluate_background_x(
        self,
        x_values,
    ):
        """
        Evalúa la evolución cosmológica usando una grilla en x = ln(a/a0).

        Parameters
        ----------
        x_values:
            Arreglo de valores de x. Debe comenzar en x=0 si se quiere imponer
            directamente la condición inicial a=a0, rho=rho0, t=t0.

        Returns
        -------
        pandas.DataFrame
            Tabla con x, t, tau, a, w, G, rho, P y H.
        """

        x_array = np.asarray(x_values, dtype=float)

        if x_array.ndim == 0:
            x_array = x_array.reshape(1)

        if np.any(~np.isfinite(x_array)):
            raise ValueError("Todos los valores de x deben ser finitos.")

        if np.any(np.diff(x_array) <= 0):
            raise ValueError("x_values debe estar estrictamente ordenado de menor a mayor.")

        if not np.isclose(x_array[0], 0.0, atol=1e-12, rtol=0.0):
            raise ValueError("Por ahora x_values debe comenzar en x=0.")

        a_values = self.scale_factor_from_x(x_array)
        w_values = self.eos_w(a_values)

        integrand_G = 1.0 + w_values

        G_values = cumulative_trapezoid(
            integrand_G,
            x_array,
            initial=0.0,
        )

        rho_values = self.density_from_G(G_values)
        H_values = self.hubble_from_G(G_values)
        P_values = w_values * rho_values

        integrand_tau = np.exp(1.5 * G_values)

        tau_values = cumulative_trapezoid(
            integrand_tau,
            x_array,
            initial=0.0,
        )

        t_values = self.params.t0 + tau_values / self.params.H0

        return pd.DataFrame(
            {
                "x": x_array,
                "t": t_values,
                "tau": tau_values,
                "a": a_values,
                "w": w_values,
                "G": G_values,
                "rho": rho_values,
                "P": P_values,
                "H": H_values,
            }
        )

    def evaluate_background_a(
        self,
        a_values,
    ):
        """
        Evalúa la evolución usando una grilla de factor de escala a.
        """

        a_array = np.asarray(a_values, dtype=float)

        if a_array.ndim == 0:
            a_array = a_array.reshape(1)

        if np.any(a_array <= 0):
            raise ValueError("Todos los valores de a deben ser positivos.")

        if np.any(np.diff(a_array) <= 0):
            raise ValueError("a_values debe estar estrictamente ordenado.")

        if not np.isclose(a_array[0], self.params.a0, atol=1e-12, rtol=0.0):
            raise ValueError("Por ahora a_values debe comenzar en a0.")

        x_values = np.log(a_array / self.params.a0)

        return self.evaluate_background_x(x_values)

    def estimate_singular_time(
        self,
        x_max: float = 8.0,
        n_points: int = 5000,
    ):
        """
        Estima el tiempo singular integrando hasta un x_max grande.

        En el modelo Cos-Cosh con wb > 0, w(a) tiende a -infinito para a grande,
        por lo que tau suele converger a un valor finito.

        Esta función entrega una estimación numérica de:

            t_s ≈ t(x_max)

        Parameters
        ----------
        x_max:
            Valor máximo de x = ln(a/a0).

        n_points:
            Número de puntos usados en la integración.

        Returns
        -------
        float
            Estimación numérica del tiempo crítico.
        """

        if x_max <= 0:
            raise ValueError("x_max debe ser positivo.")

        x_values = np.linspace(0.0, x_max, n_points)
        df = self.evaluate_background_x(x_values)

        return float(df["t"].iloc[-1])

    def classify_analytic(self):
        """
        Clasificación cualitativa esperada.

        Para los parámetros observacionales usuales con wb > 0, el término cosh
        domina en el futuro y w(a) -> -infinito. Numéricamente esto conduce a
        divergencia de rho, |P| y H en tiempo finito, compatible con Tipo I
        si además a -> infinito.

        Por ahora lo dejamos como Type I / Big Rip para wb > 0.
        """

        if self.params.wb > 0:
            return SingularityType.TYPE_I

        return SingularityType.AMBIGUOUS
