"""Parametrización adimensional e identificabilidad de fondo barotrópico."""
import numpy as np
from src.models import BarotropicPowerDiffusionModel, BarotropicPowerDiffusionParams
from .background import CoreBackground, DomainError


def barotropic_background(omega_m, omega, alpha_bar, beta, H0=70., branch=None):
    """rho0=1 fija unidades; alpha numérica coincide con alpha_bar.

    branch: +1 o -1 fija el signo de D0 para impedir mezclar componentes
    desconectadas del dominio durante una optimización.
    """
    if not np.isfinite([omega_m,omega,alpha_bar,beta,H0]).all():
        raise DomainError('Parámetros no finitos.')
    if alpha_bar<=-1 or not 0<=omega_m<1 or H0<=0:
        raise DomainError('F0>0, 0<=Om<1 y H0>0 requeridos.')
    d0=1+beta*alpha_bar
    if d0==0 or (branch is not None and branch*d0<=0):
        raise DomainError('Rama D0 incompatible o degenerada.')
    if branch not in (None,-1,1):raise ValueError('branch debe ser -1, +1 o None.')
    core=BarotropicPowerDiffusionModel(BarotropicPowerDiffusionParams(omega,alpha_bar,beta,1.))
    return CoreBackground(core,H0=H0,omega_m=omega_m)


def degeneracies(omega,alpha_bar,beta):
    """Identidades EXACTAS, no umbrales arbitrarios para casi-degeneraciones."""
    notes=[]
    if omega==-1:notes.append('Fondo LCDM: alpha_bar y beta no identificables en rama regular.')
    if alpha_bar==0:notes.append('Sin difusión: beta no identificable.')
    if beta==0:notes.append('Q constante: transferencia nula; no es difusión activa.')
    if beta==1:notes.append('Sólo w_eff=(omega-alpha_bar)/(1+alpha_bar) es identificable en el sector oscuro.')
    return notes
