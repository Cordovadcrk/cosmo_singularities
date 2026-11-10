"""Comparación condicional DESI DR2 + Pantheon+; no inferencia final del paper."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models import BarotropicPowerDiffusionModel,BarotropicPowerDiffusionParams
from src.observations import *

ROOT=Path(__file__).resolve().parents[1]
CASES={'LCDM_control':(-1.,0.,1.),'phantom_no_diffusion':(-1.2,0.,1.),
       'phantom_diffusion':(-1.2,4.,-1.)}

def factory(case,theta):
    w,alpha,beta=CASES[case]
    core=BarotropicPowerDiffusionModel(BarotropicPowerDiffusionParams(w,alpha,beta,1))
    # H0=70 es gauge de escala: sólo se ajusta H0*rd; M_SN se perfila.
    return CoreBackground(core,H0=70,omega_m=theta[0]),theta[1]/70

def main():
    bao=load_desi_dr2();sn=load_pantheon_plus()
    output=ROOT/'data/observations/results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    results={}
    for name in CASES:
        result=fit_background(lambda t:factory(name,t),[[.3,10000],[.4,10500]],[(.05,.6),(7000,13000)],bao,sn)
        bg,rd=factory(name,result['parameters'])
        refined=joint_chi2(bg,rd,bao,sn,points=8193)
        result['chi2_grid_refinement_difference']=refined['chi2']-result['chi2']
        assert abs(result['chi2_grid_refinement_difference'])<.01
        results[name]=result
        print(name,result['parameters'],result['chi2'],flush=True)
    metadata=dict(scope='Ajustes condicionales, omega/alpha_bar/beta FIJOS; Om y H0*rd libres, M perfilada.',
        names=['omega_m','H0_rd_km_s'],H0_reference=70,n_bao=len(bao.y),n_sn=len(sn.y),
        cases=CASES,results=results,
        data_manifests={n:json.loads((ROOT/'data/observations'/n/'manifest.json').read_text()) for n in ['desi_dr2','pantheon_plus']},
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').rglob('*.py')})
    (output/'fit_results.json').write_text(json.dumps(metadata,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    ref,rr=factory('LCDM_control',results['LCDM_control']['parameters']);refd=ref.distances(2.5)
    residual=sn.y-results['LCDM_control']['M_profile']-refd.mu(sn.z,sn.zhel)
    axes[0].scatter(sn.z,residual,s=3,alpha=.2,color='gray',label='Pantheon+ (sin binning)')
    z=np.geomspace(.01,2.5,300)
    for name,r in results.items():
        bg,rd=factory(name,r['parameters']);d=bg.distances(2.5)
        axes[0].plot(z,d.mu(z)-refd.mu(z)+r['M_profile']-results['LCDM_control']['M_profile'],label=name)
        axes[1].plot(np.arange(len(bao.y)),bao.prediction(d,rd)/bao.prediction(refd,rr),label=name)
    axes[0].set(xscale='log',xlabel='zHD',ylabel='Residuo de magnitud respecto al control',ylim=(-1,1))
    reference=bao.prediction(refd,rr)
    axes[1].errorbar(np.arange(len(bao.y)),bao.y/reference,yerr=np.sqrt(np.diag(bao.cov))/reference,fmt='ko',ms=4,label='DESI DR2')
    axes[1].set(xlabel='Índice del vector BAO oficial',ylabel='Observable / predicción LCDM control')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    print(output)

if __name__=='__main__':main()
