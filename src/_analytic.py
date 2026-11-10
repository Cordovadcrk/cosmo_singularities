"""Primitivas internas compartidas por los modelos phantom exactos.

Convención: 8*pi*G=1, H>0, A>0 y 3H²=c*rho con c>0.
No contienen clasificación ni dependencias de notebooks.
"""

import math


def power_increment(rho: float, rho0: float, exponent: float) -> float:
    """(rho**q-rho0**q)/q, con límite logarítmico y sin cancelación."""
    log_ratio = math.log(rho) - math.log(rho0)
    if exponent == 0.0:
        return log_ratio
    return rho0**exponent * math.expm1(exponent * log_ratio) / exponent


def evolving_density(rho0: float, exponent: float, increment: float) -> float:
    """Resuelve rho**q=rho0**q+q*increment, incluyendo q=0."""
    if exponent == 0.0:
        return math.exp(math.log(rho0) + increment)
    relative_change = exponent * increment / rho0**exponent
    if relative_change <= -1.0:
        raise ValueError("La solución rho dejó de ser real positiva; fuera del dominio temporal.")
    return math.exp(math.log(rho0) + math.log1p(relative_change) / exponent)


def phantom_singular_time(A, nu, rho0, t0, diffusion_factor=1.0):
    """Tiempo futuro de rho->infinito; None cuando nu<=1/2."""
    if nu <= 0.5:
        return None
    return t0 + rho0**(0.5 - nu) * math.sqrt(diffusion_factor) / (
        math.sqrt(3.0) * A * (nu - 0.5)
    )
