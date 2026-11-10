"""Exploración multistart con omega libre; NO posterior ni evidencia bayesiana.

python -m scripts.fit_barotropic_observations --dataset joint
Opciones: bao, sn, joint. SN sola omite H0*rd (no identificable).
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.observations import (barotropic_background,degeneracies,load_desi_dr2,
    load_pantheon_plus,fit_background,joint_chi2)

ROOT=Path(__file__).resolve().parents[1]
SPECS={
 'LCDM':dict(names=[],bounds=[],starts=[[],[]]),
 'wCDM':dict(names=['omega'],bounds=[(-1.8,-.3)],starts=[[-1.15],[-.85]]),
 'diffusion_Dpositive':dict(names=['omega','alpha_bar','beta'],
     bounds=[(-1.8,-.3),(-.8,5.),(-3.,3.)],starts=[[-1.15,.2,2.],[-.85,.3,-.5]],branch=1),
 'diffusion_Dnegative':dict(names=['omega','alpha_bar','beta'],
     bounds=[(-1.8,-.3),(-.8,5.),(-3.,3.)],starts=[[-1.05,4.,-1.],[-.9,3.,-2.]],branch=-1),
}

def setup(name,dataset):
    spec=SPECS[name]; names=['omega_m']+([] if dataset=='sn' else ['H0_rd_km_s'])+spec['names']
    bounds=[(.05,.6)]+([] if dataset=='sn' else [(7000.,13000.)])+spec['bounds']
    starts=[[om]+([] if dataset=='sn' else [hr])+tail for om,hr,tail in zip([.3,.4],[10000.,10500.],spec['starts'])]
    def factory(theta):
        v=dict(zip(names,theta))
        b=barotropic_background(v['omega_m'],v.get('omega',-1.),v.get('alpha_bar',0.),v.get('beta',1.),branch=spec.get('branch'))
        return b,v.get('H0_rd_km_s',10000.)/70.
    return names,bounds,starts,factory

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['bao','sn','joint'],default='joint')
    parser.add_argument('--models',nargs='+',choices=list(SPECS),default=list(SPECS))
    args=parser.parse_args()
    bao=load_desi_dr2() if args.dataset!='sn' else None
    sn=load_pantheon_plus() if args.dataset!='bao' else None
    zmax=max(d.z.max() for d in (bao,sn) if d is not None)
    out=ROOT/'data/observations/results'/('free_omega_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    report=dict(scope='Exploratory bounded multistart optimization; no posterior, no global optimum guarantee.',
        dataset=args.dataset,H0_reference=70.,n_bao=0 if bao is None else len(bao.y),n_sn=0 if sn is None else len(sn.y),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        data_manifests={n:json.loads((ROOT/'data/observations'/n/'manifest.json').read_text()) for n in ['desi_dr2','pantheon_plus']},
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*(ROOT/'src').rglob('*.py'),Path(__file__)]},results={})
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    z=np.linspace(0,zmax,250)
    for name in args.models:
        names,bounds,starts,factory=setup(name,args.dataset)
        if name=='diffusion_Dpositive' and 'wCDM' in report['results'] and report['results']['wCDM'].get('success'):
            # The D>0 family contains wCDM exactly at alpha_bar=0.
            starts.append(report['results']['wCDM']['parameters']+[0.,2.])
        print('Fitting',args.dataset,name,flush=True)
        try:
            result=fit_background(factory,starts,bounds,bao,sn,methods=('Powell','SLSQP'))
        except RuntimeError as exc:
            report['results'][name]=dict(success=False,error=str(exc))
            (out/'fits.json').write_text(json.dumps(report,indent=2)+'\n')
            continue
        bg,rd=factory(result['parameters']);p=bg.model.params
        result.update(names=names,bounds=bounds,initial_points=starts,
            degeneracies=degeneracies(p.omega,p.alpha,p.beta),D0=float(bg.model.D(1.)))
        delta=joint_chi2(bg,rd,bao,sn,points=8193)['chi2']-result['chi2']
        result['grid_delta_chi2']=delta;result['grid_check_passed']=abs(delta)<.01
        report['results'][name]=result
        (out/'fits.json').write_text(json.dumps(report,indent=2)+'\n')
        d=bg.dark_sector(z)
        np.savetxt(out/(name+'_trajectory.csv'),np.column_stack([z,bg.E(z)]+[d[k] for k in d]),
            delimiter=',',header=','.join(['z','E']+list(d)),comments='')
        for ax,key in zip(axes,['w_de','Q_over_3H02','Qdot_over_3H03']):ax.plot(z,d[key],label=name)
        print(name,dict(zip(names,result['parameters'])),'chi2=',result['chi2'],'boundary=',result['at_bound'],flush=True)
    for ax,label in zip(axes,[r'$w_{DE}$',r'$Q/(3H_0^2)$',r'$\dot Q/(3H_0^3)$']):
        ax.set(xlabel='z',ylabel=label);ax.grid(alpha=.25)
    axes[2].set_yscale('symlog',linthresh=.1)
    axes[0].axhline(-1,color='k',ls=':',lw=1);axes[0].legend(fontsize=7)
    fig.suptitle('Ajustes exploratorios — sin bandas de incertidumbre')
    fig.savefig(out/'reconstructions.png',dpi=180);plt.close(fig)
    print(out,flush=True)

if __name__=='__main__':main()
