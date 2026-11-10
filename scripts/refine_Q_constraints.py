"""Reconcile nested models, finish beta profiles, add omega/alpha profiles and CPL.
Usage: python -m scripts.refine_Q_constraints /absolute/path/to/Q_evidence_run
"""
from pathlib import Path
from dataclasses import replace
import sys,json,hashlib
import numpy as np
from src.observations import load_desi_dr2,load_pantheon_plus
from src.observations.inference import PowerSpec,ProfileLikelihood,search_profile
from src.observations.constraints import FixedShape,CPLSpec


def main():
    out=Path(sys.argv[1]).resolve();source=out/'report.json'
    base=json.loads(source.read_text())
    if base['status']!='completed':raise RuntimeError('Wait for main study completion.')
    target=out/'constraints.json'
    if target.exists():raise FileExistsError('Refusing to overwrite constraints.')
    result=dict(status='running',parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path('src').rglob('*.py'),Path(__file__)]},
        reconciled={},beta_profiles={},omega_profiles={},alpha_profiles={},CPL={},phantom={})
    def save():
        tmp=out/'constraints.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(target)
    bao=load_desi_dr2();sn=load_pantheon_plus(selection='cosmology')
    likes={'bao':ProfileLikelihood(bao=bao),'sn':ProfileLikelihood(sn=sn),'joint':ProfileLikelihood(bao,sn)}
    def fit(spec,like,key,starts=(),n=55):
        print(key,flush=True)
        try:
            r=search_profile(spec,like,seed=909,maxiter=n,starts=starts)
            print('chi2',r['chi2'],'D0',r['D0'],flush=True)
            return r
        except RuntimeError as exc:return dict(error=str(exc))
    # Every positive-D search is seeded from all available nested families.
    for dataset,like in likes.items():
        old=base['fits'][dataset]
        seeds=[old['noQ']['parameters']+[0.,2.],old['constantQ']['parameters']+[0.],old['power_Dpositive']['parameters']]
        result['reconciled'][dataset]=fit(PowerSpec(),like,dataset+'/positive nested check',seeds)
        t=old['noQ']['parameters']
        cpl=fit(CPLSpec(),like,dataset+'/CPL',[t+[0.]],n=75)
        if 'parameters' in cpl:
            _,w0,wa=cpl['parameters'];den=w0+1+wa
            zcross=-(1+w0)/den if den!=0 else -1
            cpl['D0']=None;cpl['near_D_floor']=False
            cpl['best_fit_crossing_z']=float(zcross) if 0<zcross<like.zmax else None
            cpl['crossing_significance']='Not assessed; a best-fit crossing is not evidence.'
        result['CPL'][dataset]=cpl;save()
    noQ=base['fits']['joint']['noQ'];joint=likes['joint']
    # Repair all beta profiles with physically feasible seeds in each component.
    for beta,entries in base['profiles'].items():
        beta=float(beta);result['beta_profiles'][str(beta)]={}
        for branch,old in entries.items():
            spec=PowerSpec(beta_fixed=beta,branch=branch,wide=True)
            if branch=='positive':seeds=[noQ['parameters']+[0.]]
            else:
                alpha=-1.1/beta
                seeds=[[.3,-1.0001,alpha],[.3,-.9999,alpha]]
            if 'parameters' in old:seeds.append(old['parameters'])
            r=fit(spec,joint,f'beta={beta}/{branch}',seeds,n=45)
            result['beta_profiles'][str(beta)][branch]=r;save()
    for w in [-1.2,-1.1,-1.,-.98,-.96,-.94,-.92,-.90,-.88,-.86,-.84,-.82,-.8,-.75,-.7]:
        spec=FixedShape(PowerSpec(kind='noQ',wide=True),'omega',w)
        r=fit(spec,joint,f'noQ omega={w}',[[noQ['parameters'][0]]],n=30)
        r['fixed_parameter']={'omega':w};result['omega_profiles'][str(w)]=r;save()
    for a in [-.95,-.8,-.4,0.,.25,1.,3.,5.,10.,20.]:
        entries={}
        for branch in ['positive']+(['negative_alpha_positive'] if a>0 else ['negative_alpha_negative'] if a<0 else []):
            spec=FixedShape(PowerSpec(wide=True,branch=branch),'alpha_bar',a)
            # beta=0 gives a valid constant-Q starting point for the positive branch.
            if branch=='positive':seeds=[[.3,-.99,0.]]
            else:
                beta=-1.1/a
                if not -6<=beta<=6:continue
                seeds=[[.3,-1.0001,beta],[.3,-.9999,beta]]
            r=fit(spec,joint,f'alpha={a}/{branch}',seeds,n=45)
            r['fixed_parameter']={'alpha_bar':a};entries[branch]=r
        result['alpha_profiles'][str(a)]=entries;save()
    for branch in ['positive','negative_alpha_positive','negative_alpha_negative']:
        seed={'positive':[.3,-1.00001,.1,2.],
            'negative_alpha_positive':[.3,-1.00001,2.,-1.],
            'negative_alpha_negative':[.3,-1.00001,-.7,2.]}[branch]
        result['phantom'][branch]=fit(PowerSpec(branch=branch,phantom=True),joint,'phantom/'+branch,[seed],n=60)
        save()
    # Reconcile full wide family with the fixed-beta and alpha incumbent points.
    wide={}
    for branch in ['positive','negative_alpha_positive','negative_alpha_negative']:
        seeds=[]
        for beta,entries in result['beta_profiles'].items():
            r=entries.get(branch,{})
            if 'parameters' in r:seeds.append((r['chi2'],r['parameters']+[float(beta)]))
        for a,entries in result['alpha_profiles'].items():
            r=entries.get(branch,{})
            if 'parameters' in r:
                om,w,b=r['parameters'];seeds.append((r['chi2'],[om,w,float(a),b]))
        seeds=[s for _,s in sorted(seeds,key=lambda t:t[0])[:3]]
        wide[branch]=fit(PowerSpec(wide=True,branch=branch),joint,'wide final/'+branch,seeds,n=80)
    result['wide_final']=wide;result['status']='completed';save()
    print('OUTPUT',target,flush=True)

if __name__=='__main__':main()
