"""Adaptación del core a redshift. H0 en km/s/Mpc; distancias en Mpc."""
import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import PchipInterpolator
from src.models import BarotropicPowerDiffusionModel, PhantomQ0Model, PhantomQAlphaRhoModel

C_KMS = 299792.458


class DomainError(ValueError):
    """La rama cosmológica no cubre el intervalo observacional."""


class CoreBackground:
    """E²=Om*(1+z)^3+Or*(1+z)^4+(1-Om-Or)*epsilon_de(z)/epsilon_de(0).

    Om=Or=0 recupera el fluido único. Om>0 es una EXTENSIÓN explícita:
    materia conservada separada y difusión sólo en el sector oscuro.
    La relación rho(a) del core sigue válida porque H cancela en continuidad.
    No se reutiliza dot(H) del fluido único como dot(H) del sistema extendido.
    """
    def __init__(self, model, H0=70., omega_m=0., omega_r=0.):
        if not isinstance(model,(BarotropicPowerDiffusionModel,PhantomQ0Model,PhantomQAlphaRhoModel)):
            raise TypeError('Modelo sin adaptador observacional; no inferir su física.')
        if not np.isfinite([H0,omega_m,omega_r]).all() or H0<=0 or min(omega_m,omega_r)<0 or omega_m+omega_r>=1:
            raise ValueError('Se requiere H0>0, Om,Or>=0 y Om+Or<1.')
        self.model,self.H0,self.omega_m,self.omega_r=model,H0,omega_m,omega_r
        self.hcore0=model.hubble(model.params.rho0)

    def density(self,z):
        """Densidad material del sector oscuro en unidades del core."""
        z=np.asarray(z,dtype=float)
        if np.any(~np.isfinite(z)) or np.any(z<0):raise DomainError('Redshift debe ser finito y >=0.')
        a=self.model.params.a0/(1+z)
        try:
            if isinstance(self.model,BarotropicPowerDiffusionModel):
                rho=np.asarray(self.model.rho_of_a(a.reshape(-1))).reshape(z.shape)
                D=self.model.D(rho)
                if np.any(D*self.model.D(self.model.params.rho0)<=0):raise DomainError('Cruce de D=0.')
            else:
                rho=np.array([self.model.rho_of_a(float(v)) for v in a.flat]).reshape(z.shape)
            if np.any(~np.isfinite(rho)) or np.any(rho<=0):raise DomainError('Densidad inválida.')
            # Verifica F>0 mediante el core, también para llamadas diagnósticas.
            if isinstance(self.model,BarotropicPowerDiffusionModel):
                F=self.model.F(rho)
                if np.any(~np.isfinite(F)) or np.any(F<=0):raise DomainError('F<=0 o no finito.')
            else:
                for r in rho.flat:self.model.hubble(float(r))
        except (ValueError,OverflowError,FloatingPointError) as exc:
            raise DomainError(str(exc)) from exc
        return rho

    def E(self,z):
        z=np.asarray(z,dtype=float)
        rho=self.density(z)
        if isinstance(self.model,BarotropicPowerDiffusionModel):
            de=(self.model.signed_hubble(rho)/self.hcore0)**2
        else:
            de=np.array([(self.model.hubble(float(r))/self.hcore0)**2 for r in rho.flat]).reshape(z.shape)
        e2=self.omega_m*(1+z)**3+self.omega_r*(1+z)**4+(1-self.omega_m-self.omega_r)*de
        if np.any(~np.isfinite(e2)) or np.any(e2<=0):raise DomainError('H² inválido.')
        return np.sqrt(e2)

    def dark_sector(self,z):
        """Diagnósticos barotrópicos; Q/(3H0²), Qdot/(3H0³), Hdot/H0².

        Las escalas H0 se entienden convertidas a unidades coherentes de tiempo.
        No se usa rho_dot del universo de un fluido para la extensión con materia.
        """
        if not isinstance(self.model,BarotropicPowerDiffusionModel):
            raise TypeError('Diagnóstico disponible para el modelo barotrópico.')
        z=np.asarray(z,float);rho=self.density(z);p=self.model.params
        q=p.alpha*rho**p.beta;D=self.model.D(rho)
        scale=(1-self.omega_m-self.omega_r)/(3*self.hcore0**2)
        enthalpy=(1+p.omega)*rho*scale
        E=self.E(z)
        transfer=-3*E*enthalpy*(D-1)/D
        return dict(rho=rho,F=self.model.F(rho),D=D,
                    w_de=-1+(1+p.omega)*rho/(rho+q),
                    rho_over_3H02=rho*scale,Q_over_3H02=q*scale,
                    Qdot_over_3H03=transfer,
                    Hdot_over_H02=-1.5*(self.omega_m*(1+z)**3+
                        4/3*self.omega_r*(1+z)**4+enthalpy))

    def distances(self,zmax,points=4097):
        return Distances(self,zmax,points)


class Distances:
    def __init__(self,background,zmax,points=4097):
        if not np.isfinite(zmax) or zmax<=0 or points<33:raise ValueError('zmax>0 y points>=33.')
        self.background,self.zmax=background,zmax
        grid=np.expm1(np.linspace(0,np.log1p(zmax),points))
        grid[-1]=zmax
        e=background.E(grid)
        self.chi=PchipInterpolator(grid,cumulative_simpson(1/e,x=grid,initial=0),extrapolate=False)

    def _z(self,z):
        z=np.asarray(z,float)
        if np.any(~np.isfinite(z)) or np.any(z<0) or np.any(z>self.zmax):raise DomainError('Fuera del intervalo de distancias.')
        return z

    def DM(self,z):return C_KMS/self.background.H0*self.chi(self._z(z))
    def DH(self,z):return C_KMS/(self.background.H0*self.background.E(self._z(z)))
    def DA(self,z):return self.DM(z)/(1+self._z(z))
    def DV(self,z):return np.cbrt(self._z(z)*self.DM(z)**2*self.DH(z))
    def DL(self,z,zhel=None):
        z=self._z(z)
        zhel=z if zhel is None else np.asarray(zhel,float)
        if np.any(~np.isfinite(zhel)) or np.any(zhel<=-1):raise DomainError('zHEL inválido.')
        return (1+zhel)*self.DM(z)
    def mu(self,z,zhel=None):
        dl=self.DL(z,zhel)
        if np.any(dl<=0):raise DomainError('mu requiere distancia positiva.')
        return 5*np.log10(dl)+25
