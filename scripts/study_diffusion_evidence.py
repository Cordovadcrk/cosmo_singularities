"""Reproducible profile study of the power law, without Bayesian-evidence claims.

Run from cosmo_singularities: python -m scripts.study_diffusion_evidence
Saves each completed stage. Re-running creates a new immutable result folder.
"""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
import hashlib,json,platform,warnings
import numpy as np
import scipy
import pandas as pd
from src.observations import load_desi_dr2,load_pantheon_plus
from src.observations.inference import PowerSpec,ProfileLikelihood,search_profile

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'data/observations/results'/('Q_evidence_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    report=dict(status='running',scope='Bounded profile likelihood study; no posterior/evidence/significance claims',
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*(ROOT/'src').rglob('*.py'),Path(__file__)]},
        fits={},profiles={},sensitivity={})
    def save():
        temp=out/'report.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(out/'report.json')
    manifests={}
    for name in ['desi_dr2','pantheon_plus']:
        folder=ROOT/'data/observations'/name
        m=json.loads((folder/'manifest.json').read_text())
        for f in m['files']:
            assert hashlib.sha256((folder/f['name']).read_bytes()).hexdigest()==f['sha256']
        manifests[name]=m
    report['data_manifests']=manifests
    bao=load_desi_dr2();sn=load_pantheon_plus(selection='cosmology')
    old=load_pantheon_plus(selection='legacy_no_calibrators')
    report['data_selection']=dict(primary='zHD>0.01, no Cepheid likelihood',n_sn=len(sn.y),n_bao=len(bao.y),
        legacy_n_sn=len(old.y),primary_rows=sn.row_indices.tolist(),legacy_rows=old.row_indices.tolist(),
        reference='https://raw.githubusercontent.com/CobayaSampler/cobaya/master/cobaya/likelihoods/sn/pantheonplus.py')
    from scipy.linalg import cho_solve
    from src.observations import barotropic_background
    reference=barotropic_background(.3,-1,0,1).distances(max(sn.z),1025)
    residual=sn.y-reference.mu(sn.z,sn.zhel)
    one=np.ones(len(residual));ci_r=cho_solve((sn.L,True),residual);ci_1=cho_solve((sn.L,True),one)
    M=(one@ci_r)/(one@ci_1)
    corrected=residual-M
    chi_reference=corrected@cho_solve((sn.L,True),corrected)
    report['projected_likelihood_check_abs_difference']=abs(float(chi_reference)-sn.profile(reference)[0])
    save()
    specs={'LCDM':PowerSpec(kind='lcdm'),'noQ':PowerSpec(kind='noQ'),
           'constantQ':PowerSpec(kind='constant'), 'power_Dpositive':PowerSpec(),
           'power_Dnegative_aplus':PowerSpec(branch='negative_alpha_positive'),
           'power_Dnegative_aminus':PowerSpec(branch='negative_alpha_negative')}
    datasets={'bao':ProfileLikelihood(bao=bao),'sn':ProfileLikelihood(sn=sn),'joint':ProfileLikelihood(bao,sn)}
    def run(spec,like,key,starts=(),maxiter=60,seed=101):
        print('Fitting',key,flush=True)
        try:
            r=search_profile(spec,like,seed=seed,maxiter=maxiter,starts=starts)
            print(key,'chi2',round(r['chi2'],6),'D0',r['D0'],'bounds',r['at_bound'],flush=True)
            return r
        except RuntimeError as exc:
            return dict(error=str(exc),spec=spec.__dict__)
    for dataset,like in datasets.items():
        report['fits'][dataset]={}
        for name,spec in specs.items():
            starts=[]
            if name=='LCDM':starts=[[.3]]
            elif name=='noQ':starts=[[.3,-1],[.3,-.9]]
            elif name in ('constantQ','power_Dpositive'):
                t=report['fits'][dataset]['noQ']['parameters']
                starts=[t+[0.]+([] if name=='constantQ' else [2.])]
            elif name=='power_Dnegative_aplus':starts=[[.3,-.9,3,-1], [.3,-1.05,4,-1]]
            else:starts=[[.3,-1.1,-.6,2],[.3,-1.01,-.7,2.5]]
            report['fits'][dataset][name]=run(spec,like,dataset+'/'+name,starts)
            save()
    # beta=1 is exactly noQ with w_eff, so do not fit redundant omega+alpha.
    report['linearQ']=dict(equivalence='w_eff=(omega-alpha_bar)/(1+alpha_bar)',
        result='Identical background likelihood to noQ over the same w_eff domain; omega and alpha cannot be measured separately.')
    # Sensitivity to search box, distance to fold, legacy SN selection, phantom restriction.
    joint=datasets['joint']
    tasks={
        'wide_constantQ':(PowerSpec(kind='constant',wide=True),joint,report['fits']['joint']['constantQ']),
        'wide_power_positive':(PowerSpec(wide=True),joint,report['fits']['joint']['power_Dpositive']),
        'wide_power_negative_aplus':(PowerSpec(wide=True,branch='negative_alpha_positive'),joint,report['fits']['joint']['power_Dnegative_aplus']),
        'wide_power_negative_aminus':(PowerSpec(wide=True,branch='negative_alpha_negative'),joint,report['fits']['joint']['power_Dnegative_aminus']),
        'legacy_noQ':(PowerSpec(kind='noQ'),ProfileLikelihood(bao,old),report['fits']['joint']['noQ']),
        'phantom_power_positive':(PowerSpec(phantom=True),joint,None),
        'phantom_power_negative_aplus':(PowerSpec(phantom=True,branch='negative_alpha_positive'),joint,None),
        'phantom_power_negative_aminus':(PowerSpec(phantom=True,branch='negative_alpha_negative'),joint,None),
    }
    for floor in [.1,.01,.001]:
        tasks['Dnegative_floor_'+str(floor)]=(PowerSpec(branch='negative_alpha_positive',d_floor=floor),joint,None)
    for key,(spec,like,previous) in tasks.items():
        starts=[previous['parameters']] if previous is not None and 'parameters' in previous else []
        if key=='wide_power_positive':starts.append(report['fits']['joint']['noQ']['parameters']+[0.,2.])
        report['sensitivity'][key]=run(spec,like,key,starts,maxiter=80,seed=202)
        save()
    # Profiles over beta. Each point includes exact alpha=0 nested-model seed.
    # Include both disconnected D<0 components where the chosen beta permits them.
    for beta in [-6.,-3.,-2.,-1.,-.5,0.,.5,1.,1.5,2.,3.,6.]:
        entries={}
        for branch in ['positive']+(['negative_alpha_positive'] if beta<0 else ['negative_alpha_negative'] if beta>1 else []):
            spec=PowerSpec(beta_fixed=beta,branch=branch,wide=True)
            starts=[report['fits']['joint']['noQ']['parameters']+[0.]] if branch=='positive' else []
            entries[branch]=run(spec,joint,f'beta={beta}/{branch}',starts,maxiter=55,seed=303)
        report['profiles'][str(beta)]=entries
        save()
    report['status']='completed'
    save()
    print('OUTPUT',out,flush=True)

if __name__=='__main__':
    # Invalid physical candidates are rejected explicitly in search_profile.
    main()
