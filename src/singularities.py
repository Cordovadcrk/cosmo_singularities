"""
singularities.py

Definiciones comunes para describir singularidades y comportamientos
cosmológicos futuros.

El módulo distingue dos niveles:

1. Clasificación estándar:
   Tipo I, Tipo II, Tipo III, Tipo IV, caso no singular y caso ambiguo.

2. Comportamiento cosmológico observado:
   descripción abierta de la evolución de rho, a, H, dot(H) y derivadas
   superiores. Esta segunda capa permite registrar turnarounds, Big Crunch,
   divergencias en tiempo infinito, ramas dominadas por difusión y posibles
   soluciones nuevas sin forzarlas dentro de la clasificación estándar.

Este módulo no resuelve ecuaciones cosmológicas ni ejecuta integraciones.
Solo define estructuras y etiquetas comunes para el proyecto.
"""

import math

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class SingularityType(Enum):
    """
    Tipos de singularidades cosmológicas futuras.

    Usamos Enum para evitar inconsistencias de texto en el proyecto.
    Por ejemplo, evitamos escribir en distintas partes:
    "Tipo I", "type I", "BigRip", "big rip", etc.
    """

    TYPE_I = "Type I / Big Rip"
    TYPE_II = "Type II / Sudden singularity"
    TYPE_III = "Type III"
    TYPE_IV = "Type IV"
    NON_SINGULAR = "Non singular"
    AMBIGUOUS = "Ambiguous"

class ClassificationStatus(Enum):
    """
    Estado epistemológico de una clasificación.

    ROBUST:
        El comportamiento está respaldado por convergencia numérica,
        consistencia matemática y estabilidad frente a cambios en
        tolerancias o umbrales.

    PROVISIONAL:
        Existe evidencia suficiente para proponer una interpretación,
        pero aún faltan pruebas adicionales.

    NEW_BEHAVIOR_CANDIDATE:
        El comportamiento observado no coincide claramente con una
        clasificación existente y debe estudiarse como posible solución
        nueva.

    NUMERICALLY_INCONCLUSIVE:
        La integración o los diagnósticos disponibles no permiten obtener
        una conclusión confiable.
    """

    ROBUST = "robust"
    PROVISIONAL = "provisional"
    NEW_BEHAVIOR_CANDIDATE = "new_behavior_candidate"
    NUMERICALLY_INCONCLUSIVE = "numerically_inconclusive"

class KnownCosmologicalBehavior(Enum):
    """
    Etiquetas comunes para comportamientos cosmológicos observados.

    Esta enumeración proporciona nombres consistentes para los casos
    conocidos, pero no limita las posibles clasificaciones.

    ClassificationResult.behavior_label acepta también cadenas de texto
    nuevas que no aparezcan en esta enumeración.
    """

    REGULAR_EXPANSION = "regular_expansion"

    DE_SITTER = "de_sitter"

    ASYMPTOTIC_DE_SITTER = "asymptotic_de_sitter"

    TYPE_I_BIG_RIP = "type_i_big_rip"

    TYPE_II_SUDDEN = "type_ii_sudden"

    TYPE_III = "type_iii"

    TYPE_IV = "type_iv"

    BIG_CRUNCH = "big_crunch"

    TURNAROUND = "turnaround"

    TURNAROUND_TO_CONTRACTION = (
        "turnaround_to_contraction"
    )

    TURNAROUND_TO_BIG_CRUNCH = (
        "turnaround_to_big_crunch"
    )

    TURNAROUND_TO_HIGHER_DERIVATIVE_BOUNDARY = (
        "turnaround_to_higher_derivative_boundary"
    )

    FINITE_TIME_DIVERGENCE = (
        "finite_time_divergence"
    )

    INFINITE_TIME_DIVERGENCE = (
        "infinite_time_divergence"
    )

    LITTLE_RIP = (
        "little rip"
    )    

    LITTLE_SIBLING_BIG_RIP = (
        "little_sibling_of_the_big_rip"
    )

    DIFFUSION_DRIVEN_INFINITE_TIME_RIP = (
        "diffusion_driven_infinite_time_rip"
    )

    HIGHER_DERIVATIVE_BOUNDARY = (
        "higher_derivative_boundary"
    )

    FRIEDMANN_DOMAIN_BOUNDARY = (
        "friedmann_domain_boundary"
    )

    CONTINUITY_DENOMINATOR_BOUNDARY = (
        "continuity_denominator_boundary"
    )

    NEW_BEHAVIOR_CANDIDATE = (
        "new_behavior_candidate"
    )

    UNCLASSIFIED = "unclassified"

    NUMERICALLY_INCONCLUSIVE = (
        "numerically_inconclusive"
    )  

def _current_enum(value, enum_type):
    """Acepta miembros previos a reload sólo de la misma enum del módulo.

    Comparar únicamente el valor admitiría enumeraciones ajenas. La identidad
    nominal y la conversión por valor preservan la validación tras reload.
    """
    if isinstance(value, enum_type):
        return value
    if (isinstance(value, Enum)
            and type(value).__module__ == enum_type.__module__
            and type(value).__qualname__ == enum_type.__qualname__):
        return enum_type(value.value)
    raise TypeError(f"Se esperaba un miembro de {enum_type.__name__}.")


def normalize_behavior_label(
    behavior_label: str | KnownCosmologicalBehavior,
) -> str:
    """
    Convierte una etiqueta de comportamiento a una cadena estándar.

    La función acepta:

    - un miembro de KnownCosmologicalBehavior;
    - una cadena nueva definida durante el análisis.

    Esto permite mantener etiquetas conocidas y, al mismo tiempo,
    registrar soluciones nuevas.
    """

    if isinstance(behavior_label, Enum):
        return _current_enum(behavior_label, KnownCosmologicalBehavior).value

    if not isinstance(
        behavior_label,
        str,
    ):
        raise TypeError(
            "behavior_label debe ser una cadena o un miembro de "
            "KnownCosmologicalBehavior."
        )

    normalized_label = (
        behavior_label
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if not normalized_label:
        raise ValueError(
            "behavior_label no puede estar vacío."
        )

    # Conservar valores públicos históricos (incluido 'little rip'), pero
    # hacer que enum, cadena y lectura de CSV produzcan la misma etiqueta.
    for member in KnownCosmologicalBehavior:
        if member.value.replace(" ", "_").replace("-", "_") == normalized_label:
            return member.value
    return normalized_label

@dataclass(frozen=True)
class SingularityCriteria:
    """
    Criterios físicos cualitativos asociados a cada tipo de singularidad.

    Esta clase no clasifica automáticamente. Solo guarda la descripción
    matemática/física esperada cuando t -> ts, donde ts es un tiempo futuro
    finito.

    Parámetros
    ----------
    singularity_type:
        Tipo de singularidad cosmológica.

    scale_factor:
        Comportamiento esperado de a(t).

    density:
        Comportamiento esperado de rho(t).

    pressure:
        Comportamiento esperado de |P(t)|.

    hubble:
        Comportamiento esperado de H(t).

    hdot:
        Comportamiento esperado de dot(H)(t).

    higher_derivatives:
        Comportamiento esperado de derivadas superiores de H(t).

    description:
        Explicación breve del tipo de singularidad.
    """

    singularity_type: SingularityType
    scale_factor: str
    density: str
    pressure: str
    hubble: str
    hdot: str
    higher_derivatives: str
    description: str

@dataclass(frozen=True)
class ClassificationResult:
    """
    Resultado completo de la clasificación de una evolución cosmológica.

    La clasificación estándar es opcional. El comportamiento observado
    se conserva incluso cuando no coincide con los tipos I--IV.

    Parameters
    ----------
    standard_type:
        Clasificación estándar, cuando existe. Puede ser None cuando el
        comportamiento no coincide con la taxonomía usual.

    behavior_label:
        Etiqueta descriptiva abierta. Puede ser un comportamiento conocido
        o una etiqueta nueva propuesta durante el análisis.

    status:
        Nivel de certeza de la clasificación.

    is_singularity:
        True si existe evidencia de una singularidad física.
        False si la evolución es regular o asintótica.
        None si todavía no se ha determinado.

    finite_time:
        True si el límite ocurre en un tiempo futuro finito.
        False si ocurre solamente cuando t -> infinity.
        None si no se ha determinado.

    confidence:
        Confianza diagnóstica entre 0 y 1. No representa una probabilidad
        estadística; resume la solidez de la evidencia disponible.

    observed_behavior:
        Comportamiento observado de las variables cosmológicas. Este
        diccionario no está restringido a un conjunto cerrado de claves.

    evidence:
        Evidencia numérica y matemática que respalda la clasificación.

    notes:
        Comentarios físicos, advertencias o aspectos pendientes.
    """

    standard_type: SingularityType | None

    behavior_label: str

    status: ClassificationStatus

    is_singularity: bool | None

    finite_time: bool | None

    confidence: float

    observed_behavior: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict,
    )

    evidence: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict,
    )

    notes: str = ""

    def __post_init__(self):
        """
        Valida y normaliza el resultado después de su creación.
        """

        normalized_label = normalize_behavior_label(
            self.behavior_label
        )

        object.__setattr__(
            self,
            "behavior_label",
            normalized_label,
        )

        object.__setattr__(self, "status", _current_enum(self.status, ClassificationStatus))
        if self.standard_type is not None:
            object.__setattr__(self, "standard_type",
                               _current_enum(self.standard_type, SingularityType))

        if (
            self.is_singularity is not None
            and not isinstance(
                self.is_singularity,
                bool,
            )
        ):
            raise TypeError(
                "is_singularity debe ser bool o None."
            )

        if (
            self.finite_time is not None
            and not isinstance(
                self.finite_time,
                bool,
            )
        ):
            raise TypeError(
                "finite_time debe ser bool o None."
            )

        confidence = float(
            self.confidence
        )

        if not math.isfinite(
            confidence
        ):
            raise ValueError(
                "confidence debe ser finita."
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence debe pertenecer al intervalo [0, 1]."
            )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

        object.__setattr__(
            self,
            "observed_behavior",
            MappingProxyType(
                dict(
                    self.observed_behavior
                )
            ),
        )

        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(
                dict(
                    self.evidence
                )
            ),
        )

        object.__setattr__(
            self,
            "notes",
            str(self.notes).strip(),
        )

    def as_dict(self) -> dict[str, Any]:
        """
        Convierte el resultado a un diccionario serializable.

        Este formato será útil para construir DataFrames, guardar CSV
        y crear datasets para clasificación.
        """

        return {
            "standard_type": (
                self.standard_type.value
                if self.standard_type is not None
                else None
            ),
            "behavior_label": self.behavior_label,
            "status": self.status.value,
            "is_singularity": self.is_singularity,
            "finite_time": self.finite_time,
            "confidence": self.confidence,
            "observed_behavior": dict(
                self.observed_behavior
            ),
            "evidence": dict(
                self.evidence
            ),
            "notes": self.notes,
        }

SINGULARITY_CRITERIA = {
    SingularityType.TYPE_I: SingularityCriteria(
        singularity_type=SingularityType.TYPE_I,
        scale_factor="a(t) -> infinity",
        density="rho(t) -> infinity",
        pressure="|P(t)| -> infinity",
        hubble="H(t) -> infinity",
        hdot="typically divergent",
        higher_derivatives="typically divergent",
        description=(
            "Big Rip singularity. The scale factor, energy density, "
            "pressure magnitude and Hubble parameter diverge at a finite "
            "future time."
        ),
    ),

    SingularityType.TYPE_II: SingularityCriteria(
        singularity_type=SingularityType.TYPE_II,
        scale_factor="a(t) -> finite nonzero value",
        density="rho(t) -> finite value",
        pressure="|P(t)| -> infinity",
        hubble="H(t) -> finite value",
        hdot="may diverge",
        higher_derivatives="may diverge",
        description=(
            "Sudden singularity. The scale factor and energy density remain "
            "finite, while the pressure magnitude diverges at a finite "
            "future time."
        ),
    ),

    SingularityType.TYPE_III: SingularityCriteria(
        singularity_type=SingularityType.TYPE_III,
        scale_factor="a(t) -> finite nonzero value",
        density="rho(t) -> infinity",
        pressure="|P(t)| -> infinity",
        hubble="H(t) -> infinity",
        hdot="typically divergent",
        higher_derivatives="typically divergent",
        description=(
            "Type III singularity. The scale factor remains finite, but "
            "the energy density, pressure magnitude and Hubble parameter "
            "diverge at a finite future time."
        ),
    ),

    SingularityType.TYPE_IV: SingularityCriteria(
        singularity_type=SingularityType.TYPE_IV,
        scale_factor="a(t) -> finite nonzero value",
        density="rho(t) -> zero or finite value",
        pressure="|P(t)| -> zero or finite value",
        hubble="H(t) -> finite value",
        hdot="dot(H)(t) -> finite value",
        higher_derivatives="some higher derivative of H(t) diverges",
        description=(
            "Type IV singularity. The main cosmological quantities remain "
            "finite, but higher-order derivatives of the Hubble parameter "
            "diverge at a finite future time."
        ),
    ),

    SingularityType.NON_SINGULAR: SingularityCriteria(
        singularity_type=SingularityType.NON_SINGULAR,
        scale_factor="no finite-time divergence",
        density="no finite-time divergence",
        pressure="no finite-time divergence",
        hubble="no finite-time divergence",
        hdot="no finite-time divergence",
        higher_derivatives="no finite-time divergence detected",
        description=(
            "No finite-time future singularity. Quantities may remain finite "
            "or diverge only asymptotically as t -> infinity."
        ),
    ),

    SingularityType.AMBIGUOUS: SingularityCriteria(
        singularity_type=SingularityType.AMBIGUOUS,
        scale_factor="not determined",
        density="not determined",
        pressure="not determined",
        hubble="not determined",
        hdot="not determined",
        higher_derivatives="not determined",
        description=(
            "Ambiguous or numerically inconclusive case. The available "
            "information is not sufficient to assign a robust singularity type."
        ),
    ),
}


def get_singularity_criteria(
    singularity_type: SingularityType,
) -> SingularityCriteria:
    """
    Devuelve los criterios físicos asociados a un tipo de singularidad.

    Parameters
    ----------
    singularity_type:
        Tipo de singularidad definido en SingularityType.

    Returns
    -------
    SingularityCriteria
        Objeto con la descripción física del tipo de singularidad.
    """

    return SINGULARITY_CRITERIA[singularity_type]


def describe_singularity(singularity_type: SingularityType) -> str:
    """
    Devuelve una descripción breve de un tipo de singularidad.

    Parameters
    ----------
    singularity_type:
        Tipo de singularidad definido en SingularityType.

    Returns
    -------
    str
        Descripción textual breve.
    """

    criteria = get_singularity_criteria(singularity_type)
    return criteria.description

def describe_classification(
    result: ClassificationResult,
) -> str:
    """
    Devuelve una descripción textual compacta de una clasificación
    cosmológica completa.
    """

    if not isinstance(
        result,
        ClassificationResult,
    ):
        raise TypeError(
            "result debe ser ClassificationResult."
        )

    if result.standard_type is None:
        standard_description = (
            "sin clasificación estándar"
        )
    else:
        standard_description = (
            result.standard_type.value
        )

    if result.finite_time is True:
        time_description = "tiempo finito"
    elif result.finite_time is False:
        time_description = "tiempo infinito"
    else:
        time_description = "tiempo no determinado"

    if result.is_singularity is True:
        singularity_description = (
            "singularidad"
        )
    elif result.is_singularity is False:
        singularity_description = (
            "comportamiento no singular"
        )
    else:
        singularity_description = (
            "naturaleza singular no determinada"
        )

    return (
        f"{result.behavior_label}: "
        f"{standard_description}; "
        f"{singularity_description}; "
        f"{time_description}; "
        f"estado={result.status.value}; "
        f"confianza={result.confidence:.3f}."
    )
