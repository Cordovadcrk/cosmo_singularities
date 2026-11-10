"""Observaciones de fondo: núcleo reutilizado, catálogos, distancias y ajustes."""
from .background import CoreBackground,Distances,DomainError
from .data import load_desi_dr2,load_pantheon_plus
from .likelihood import Gaussian,BAOData,SNData,joint_chi2
from .fitting import fit_background
from .barotropic import barotropic_background,degeneracies
