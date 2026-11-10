"""Conditional profile adapters; use the same likelihood and physical models."""
from dataclasses import dataclass
import numpy as np
from .background import Distances,DomainError


@dataclass(frozen=True)
class FixedShape:
    base: object
    parameter: str
    value: float

    def __getattr__(self,name):return getattr(self.base,name)
    @property
    def names(self):return [n for n in self.base.names if n!=self.parameter]
    @property
    def bounds(self):return [b for n,b in zip(self.base.names,self.base.bounds) if n!=self.parameter]
    def expand(self,theta):
        v=dict(zip(self.names,theta));v[self.parameter]=self.value
        return [v[n] for n in self.base.names]
    def background(self,theta):return self.base.background(self.expand(theta))
    def values(self,theta):return self.base.values(self.expand(theta))


class CPLBackground:
    """Benchmark only, NOT an additional UG diffusion solution.

    w(a)=w0+wa(1-a); integrating d ln epsilon/d ln a=-3(1+w)
    gives epsilon/epsilon0=(1+z)^[3(1+w0+wa)] exp[-3 wa z/(1+z)].
    Distances and covariance likelihood are shared with the core adapter.
    """
    def __init__(self,omega_m,w0,wa,H0=70.):
        self.omega_m,self.w0,self.wa,self.H0=omega_m,w0,wa,H0
    def E(self,z):
        z=np.asarray(z,float)
        if np.any(~np.isfinite(z)) or np.any(z<0):raise DomainError('Invalid z.')
        de=np.exp(3*(1+self.w0+self.wa)*np.log1p(z)-3*self.wa*z/(1+z))
        return np.sqrt(self.omega_m*(1+z)**3+(1-self.omega_m)*de)
    def distances(self,zmax,points=4097):return Distances(self,zmax,points)


class CPLSpec:
    """Protocol used by search_profile; values/D0 are not applicable to CPL."""
    names=['omega_m','w0','wa']
    bounds=[(.01,.8),(-3.,.5),(-5.,5.)]
    kind='CPL';beta_fixed=None;branch='not_applicable';wide=True;d_floor=0.;phantom=False
    def background(self,theta):return CPLBackground(*theta)
    def values(self,theta):return theta[1],0.,1.
