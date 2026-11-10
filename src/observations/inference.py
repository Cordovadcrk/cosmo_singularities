"""Profile inference using existing backgrounds, distances and full covariances.

No Bayesian evidence or calibrated likelihood-ratio significance is inferred.
The power-law null contains non-identifiable parameters and singular boundaries.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import differential_evolution, minimize
from .background import DomainError
from .barotropic import barotropic_background


class ProfileLikelihood:
    """Profile M_SN and positive BAO scale S=H0*rd exactly.

    At S=1 the BAO prediction is u; prediction(S)=u/S. Whitening by
    the full covariance gives 1/S=(u_w.y_w)/(u_w.u_w).
    Profiling is not Bayesian marginalization in S.
    """
    def __init__(self,bao=None,sn=None,points=1025):
        if bao is None and sn is None:raise ValueError('No data.')
        self.bao,self.sn,self.points=bao,sn,points
        self.zmax=max(float(d.z.max()) for d in (bao,sn) if d is not None)
        self.bao_yw=None if bao is None else bao.whiten(bao.y)

    def evaluate(self,bg,points=None):
        d=bg.distances(self.zmax,self.points if points is None else points)
        b,S=0.,None
        if self.bao is not None:
            u=self.bao.whiten(self.bao.prediction(d,1/bg.H0))
            amplitude=float(u@self.bao_yw/(u@u))
            if amplitude<=0 or not np.isfinite(amplitude):raise DomainError('BAO scale has no finite positive optimum.')
            S=1/amplitude
            r=self.bao_yw-amplitude*u
            b=float(r@r)
        s,M=self.sn.profile(d) if self.sn is not None else (0.,None)
        return dict(chi2=b+s,chi2_bao=b,chi2_sn=s,H0_rd_km_s=S,M_profile=M)


@dataclass(frozen=True)
class PowerSpec:
    """Shape parameters; no unidentifiable BAO scale in SN-only fits."""
    kind: str='power'
    beta_fixed: float|None=None
    branch: str='positive'  # positive, negative_alpha_positive, negative_alpha_negative
    wide: bool=False
    d_floor: float=1e-5
    phantom: bool=False

    @property
    def names(self):
        return ['omega_m']+([] if self.kind=='lcdm' else ['omega'])+(
            [] if self.kind in ('lcdm','noQ') else ['alpha_bar'])+(
            ['beta'] if self.kind=='power' and self.beta_fixed is None else [])

    @property
    def bounds(self):
        bounds=[(.01,.8) if self.wide else (.05,.6)]
        if self.kind!='lcdm':bounds.append((-3.,-1.000001) if self.phantom else ((-3.,.5) if self.wide else (-1.8,-.3)))
        if self.kind not in ('lcdm','noQ'):
            bounds.append((-.95,20.) if self.wide else (-.8,5.))
        if self.kind=='power' and self.beta_fixed is None:
            bounds.append((-6.,6.) if self.wide else (-3.,3.))
        return bounds

    def values(self,theta):
        v=dict(zip(self.names,theta))
        return v.get('omega',-1.),v.get('alpha_bar',0.),(
            0. if self.kind=='constant' else (self.beta_fixed if self.beta_fixed is not None else v.get('beta',1.)))

    def background(self,theta):
        w,a,b=self.values(theta)
        D=1+a*b
        if abs(D)<self.d_floor:raise DomainError('Numerical/regularity floor on |D0|.')
        if self.branch=='positive':sign=1
        elif self.branch=='negative_alpha_positive':
            if a<=0 or b>=0:raise DomainError('Wrong connected D<0 component.')
            sign=-1
        elif self.branch=='negative_alpha_negative':
            if a>=0 or b<=1:raise DomainError('Wrong connected D<0 component.')
            sign=-1
        else:raise ValueError('Unknown branch.')
        return barotropic_background(theta[0],w,a,b,branch=sign)


def search_profile(spec,likelihood,seed=1,maxiter=60,starts=()):
    """Seeded global search plus local refinements; retain valid incumbents.

    Failed local convergence is recorded, never silently promoted to a
    certified minimum. A known nested-model incumbent cannot be discarded.
    """
    bounds=np.array(spec.bounds,float);lo=bounds[:,0];span=bounds[:,1]-lo
    def objective(u):
        if np.any(u<0) or np.any(u>1):return 1e50
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                return likelihood.evaluate(spec.background(lo+span*u))['chi2']
        except (DomainError,OverflowError,FloatingPointError):return 1e50
    candidates=[]
    for t in starts:
        u=(np.asarray(t)-lo)/span
        value=objective(u)
        if value<1e40:candidates.append((value,u.copy(),'supplied',False))
    result=differential_evolution(objective,[(0,1)]*len(lo),rng=np.random.default_rng(seed),
        maxiter=maxiter,popsize=8,tol=2e-7,atol=1e-6,polish=False)
    attempts=[dict(method='differential_evolution',success=bool(result.success),message=str(result.message),chi2=float(result.fun))]
    if result.fun<1e40:candidates.append((result.fun,result.x,'differential_evolution',bool(result.success)))
    if not candidates:raise RuntimeError('No valid point in search.')
    seeds=sorted(candidates,key=lambda v:v[0])[:3]
    for _,u,_,_ in seeds:
        for method in ['SLSQP','Powell']:
            options={'maxiter':250,'ftol':1e-9}
            if method=='Powell':options['xtol']=1e-6
            r=minimize(objective,u,method=method,bounds=[(0,1)]*len(lo),options=options)
            attempts.append(dict(method=method,success=bool(r.success),message=str(r.message),chi2=float(r.fun)))
            if r.fun<1e40:candidates.append((r.fun,r.x,method,bool(r.success)))
    value,u,method,converged=min(candidates,key=lambda v:v[0]);theta=lo+span*u
    bg=spec.background(theta);detail=likelihood.evaluate(bg,points=4097)
    fine=likelihood.evaluate(bg,points=8193)
    w,a,b=spec.values(theta)
    return dict(parameters=theta.tolist(),names=spec.names,bounds=spec.bounds,
        **detail,search_chi2=float(value),selected_method=method,local_converged=converged,
        grid_difference=detail['chi2']-float(value),refined_grid_difference=fine['chi2']-detail['chi2'],
        numerical_pass=bool(max(abs(detail['chi2']-value),abs(fine['chi2']-detail['chi2']))<.01),
        D0=1+a*b,at_bound=[bool(x<.002 or x>.998) for x in u],
        near_D_floor=bool(abs(1+a*b)<10*spec.d_floor),attempts=attempts,
        spec=dict(kind=spec.kind,beta_fixed=spec.beta_fixed,branch=spec.branch,wide=spec.wide,d_floor=spec.d_floor,phantom=spec.phantom))
