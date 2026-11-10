"""Verosimilitudes gaussianas con covarianza completa y perfil de intercepto SN."""
import numpy as np
from scipy.linalg import cholesky, solve_triangular


class Gaussian:
    def __init__(self,y,cov):
        self.y=np.asarray(y,float); self.cov=np.asarray(cov,float)
        if self.y.ndim!=1 or self.cov.shape!=(len(self.y),len(self.y)) or not len(self.y):
            raise ValueError('Dimensión de covarianza incompatible.')
        if not np.isfinite(self.y).all() or not np.isfinite(self.cov).all():raise ValueError('Datos no finitos.')
        if not np.allclose(self.cov,self.cov.T,rtol=1e-10,atol=1e-12):raise ValueError('Covarianza no simétrica.')
        self.L=cholesky(self.cov,lower=True)
        self.logdet=2*np.log(np.diag(self.L)).sum()
        self.ones=self.whiten(np.ones(len(self.y)))

    def whiten(self,r):return solve_triangular(self.L,r,lower=True,check_finite=True)
    def chi2(self,prediction):
        p=np.asarray(prediction,float)
        if p.shape!=self.y.shape:raise ValueError('Predicción de dimensión incompatible.')
        r=self.whiten(self.y-p)
        return float(r@r)
    def loglike(self,prediction):return -.5*(self.chi2(prediction)+self.logdet+len(self.y)*np.log(2*np.pi))
    def profile_offset(self,prediction):
        """Perfil exacto de y=prediction+M; no es marginalización con prior propio."""
        p=np.asarray(prediction,float)
        if p.shape!=self.y.shape:raise ValueError('Predicción de dimensión incompatible.')
        r=self.whiten(self.y-p); offset=float(self.ones@r/(self.ones@self.ones))
        residual=r-offset*self.ones
        return float(residual@residual),offset


class BAOData(Gaussian):
    def __init__(self,z,y,kinds,cov):
        super().__init__(y,cov)
        self.z=np.asarray(z,float); self.kinds=np.asarray(kinds,str)
        if self.z.shape!=self.y.shape or self.kinds.shape!=self.y.shape or np.any(self.z<=0) or not np.isfinite(self.z).all():raise ValueError('Redshifts BAO inválidos.')
        if not set(self.kinds)<= {'DM_over_rs','DH_over_rs','DV_over_rs'}:raise ValueError('Observable BAO desconocido.')
    def prediction(self,distances,rd):
        if not np.isfinite(rd) or rd<=0:raise ValueError('rd debe ser positivo en Mpc.')
        return np.array([getattr(distances,k.split('_')[0])(z)/rd for z,k in zip(self.z,self.kinds)])


class SNData(Gaussian):
    def __init__(self,z,zhel,mb,cov):
        super().__init__(mb,cov)
        self.z=np.asarray(z,float);self.zhel=np.asarray(zhel,float)
        if self.z.shape!=self.y.shape or self.zhel.shape!=self.y.shape or np.any(self.z<=0) or not np.isfinite(self.z).all() or not np.isfinite(self.zhel).all():raise ValueError('Redshifts SN inválidos.')
    def profile(self,distances):return self.profile_offset(distances.mu(self.z,self.zhel))


def joint_chi2(background,rd,bao=None,sn=None,points=4097):
    datasets=[d for d in (bao,sn) if d is not None]
    if not datasets:raise ValueError('Faltan datos.')
    distances=background.distances(max(float(d.z.max()) for d in datasets),points)
    b=bao.chi2(bao.prediction(distances,rd)) if bao is not None else 0.
    s,M=sn.profile(distances) if sn is not None else (0.,None)
    return dict(chi2=b+s,chi2_bao=b,chi2_sn=s,M_profile=M)
