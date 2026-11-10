"""Ajuste acotado de chi²; no devuelve intervalos posteriores ni evidencia."""
import numpy as np
from scipy.optimize import minimize
from .background import DomainError
from .likelihood import joint_chi2

def fit_background(factory,starts,bounds,bao=None,sn=None,methods=('Powell',)):
    """factory(theta)->(CoreBackground, rd[Mpc]); varios inicios independientes."""
    def objective(theta):
        try:
            bg,rd=factory(theta)
            return joint_chi2(bg,rd,bao,sn)['chi2']
        except (DomainError,OverflowError,FloatingPointError):return 1e100
    lower,upper=np.asarray(bounds,float).T
    width=upper-lower
    if np.any(width<=0):raise ValueError('Límites inválidos.')
    results=[]
    for start in starts:
        for method in methods:
            if method not in ('Powell','SLSQP'):raise ValueError('Método no soportado.')
            options=({'xtol':1e-5,'ftol':1e-7,'maxiter':200} if method=='Powell'
                     else {'ftol':1e-8,'maxiter':300})
            r=minimize(lambda u:objective(lower+width*u),(np.asarray(start)-lower)/width,
                       method=method,bounds=[(0.,1.)]*len(bounds),options=options)
            r.x=lower+width*r.x
            r.method=method
            results.append(r)
    valid=[r for r in results if r.success and np.isfinite(r.fun) and r.fun<1e99]
    if not valid:raise RuntimeError('Ningún ajuste convergió en el dominio físico.')
    best=min(valid,key=lambda r:r.fun)
    initial_best=min(objective(np.asarray(s,float)) for s in starts)
    if best.fun>initial_best+1e-4:
        raise RuntimeError('Optimizadores empeoraron el mejor punto inicial; resultado rechazado.')
    bg,rd=factory(best.x)
    return dict(parameters=best.x.tolist(),**joint_chi2(bg,rd,bao,sn),
                success=True,message=str(best.message),
                at_bound=[bool(min(abs(x-lo),abs(x-hi))<1e-3*(hi-lo)) for x,(lo,hi) in zip(best.x,bounds)],
                starts=[dict(success=bool(r.success),method=r.method,message=str(r.message),chi2=float(r.fun),parameters=r.x.tolist()) for r in results])
