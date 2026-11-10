"""Validate and render a completed Q profile study. No fits are changed."""
from pathlib import Path
import sys,json,hashlib,base64
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp,quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.observations.inference import PowerSpec
from src.observations.constraints import CPLSpec

ROOT=Path(__file__).resolve().parents[1]

def best(entries):
    good=[r for r in entries.values() if 'chi2' in r and r['numerical_pass']]
    return min(good,key=lambda r:r['chi2']) if good else None

def main():
    out=Path(sys.argv[1]).resolve()
    base=json.loads((out/'report.json').read_text());ref=json.loads((out/'constraints.json').read_text())
    if base['status']!='completed' or ref['status']!='completed':raise RuntimeError('Incomplete study.')
    summary=dict(parent_hashes={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['report.json','constraints.json']},
        scope='Profiles only; neither posterior nor evidence. Discrete profiles do not certify global extrema.',
        validations={},comparisons={})
    rows=[]
    for dataset,entries in base['fits'].items():
        models=dict(entries);models['power_Dpositive']=ref['reconciled'][dataset];models['CPL']=ref['CPL'][dataset]
        for name,r in models.items():
            if 'chi2' not in r:continue
            rows.append(dict(dataset=dataset,model=name,chi2=r['chi2'],delta_vs_noQ=r['chi2']-entries['noQ']['chi2'],
                numerical_pass=r['numerical_pass'],local_converged=r['local_converged'],boundary=any(r['at_bound']),
                near_D_floor=r['near_D_floor'],D0=r['D0']))
    for name,r in ref['wide_final'].items():
        rows.append(dict(dataset='joint_wide',model=name,chi2=r['chi2'],delta_vs_noQ=r['chi2']-base['fits']['joint']['noQ']['chi2'],
            numerical_pass=r['numerical_pass'],local_converged=r['local_converged'],boundary=any(r['at_bound']),
            near_D_floor=r['near_D_floor'],D0=r['D0']))
    table=pd.DataFrame(rows);table.to_csv(out/'comparison.csv',index=False)
    power_best=best(ref['wide_final'])
    if power_best is None:raise RuntimeError('No validated power-law fit.')
    profiles={}
    for key in ['beta','alpha']:
        profiles[key]=[(float(x),best(entries)['chi2']-power_best['chi2']) for x,entries in ref[key+'_profiles'].items() if best(entries)]
        profiles[key].sort()
    wprofile=sorted((float(w),r['chi2']-base['fits']['joint']['noQ']['chi2']) for w,r in ref['omega_profiles'].items() if 'chi2' in r and r['numerical_pass'])
    summary['profiles']=dict(beta=profiles['beta'],alpha_bar=profiles['alpha'],omega_noQ=wprofile)
    # Approximate regular-model likelihood intervals, conditional on noQ only.
    wx,wy=np.array(wprofile).T;interp=PchipInterpolator(wx,wy)
    intervals={}
    for level in [1.,3.841459]:
        roots=[]
        for l,h,vl,vh in zip(wx[:-1],wx[1:],wy[:-1],wy[1:]):
            if (vl-level)*(vh-level)<0:roots.append(float(brentq(lambda w:float(interp(w))-level,l,h)))
        intervals[str(level)]=roots
    summary['noQ_conditional_profile_crossings']=intervals
    summary['noQ_interval_note']='PCHIP interpolation of discrete profiles. Approximate regular-model likelihood thresholds; not diffusion intervals or Bayesian credible intervals.'
    z=np.unique(np.r_[0,np.geomspace(1e-8,2.33,350)])
    trajectories={}
    fits={'LCDM':base['fits']['joint']['LCDM'],'noQ':base['fits']['joint']['noQ'],
        'constantQ':base['sensitivity']['wide_constantQ'],**{'power_'+k:v for k,v in ref['wide_final'].items()}}
    for name,r in fits.items():
        spec=PowerSpec(**r['spec']);bg=spec.background(r['parameters']);d=bg.dark_sector(z)
        p=bg.model.params
        # Independent time-variable formulation in N=ln(a), using D from core.
        N=-np.log1p(z)
        sol=solve_ivp(lambda n,u:[-3*(1+p.omega)/bg.model.D(np.exp(u[0]))],
            (0,float(N[-1])),[0.],t_eval=N,rtol=1e-10,atol=1e-12,max_step=.02,method='DOP853')
        if not sol.success:raise RuntimeError(name+': independent ODE failed: '+sol.message)
        rho_ode=np.exp(sol.y[0]);rho_error=float(np.max(np.abs(rho_ode/d['rho']-1)))
        if rho_error>1e-5:raise RuntimeError(name+': density validation failed.')
        distances=bg.distances(2.33,8193)
        check_z=[.1,1.,2.33]
        independent=np.array([299792.458/bg.H0*quad(lambda zz:1/float(bg.E(zz)),0,zz,epsabs=1e-10,epsrel=1e-10)[0] for zz in check_z])
        distance_error=float(np.max(np.abs(distances.DM(check_z)/independent-1)))
        if distance_error>1e-6:raise RuntimeError(name+': distance validation failed.')
        summary['validations'][name]=dict(max_relative_density_ODE_error=rho_error,max_relative_distance_quad_error=distance_error,
            min_abs_D=float(np.min(abs(d['D']))),min_F=float(np.min(d['F'])),
            entropy_sign_Tpositive_on_grid=bool(np.all(d['Qdot_over_3H03']<=1e-10)),
            caveat='Entropy flag is only the sampled sign of -Qdot; not S>0 or stability.')
        frame=pd.DataFrame(dict(z=z,E=bg.E(z),**d));frame.to_csv(out/(name+'_validated_trajectory.csv'),index=False)
        trajectories[name]=frame
    cpl=ref['CPL']['joint'];cb=CPLSpec().background(cpl['parameters'])
    _,w0,wa=cpl['parameters'];summary['CPL_best_fit_crossing_z']=cpl['best_fit_crossing_z']
    summary['power_improvement_over_noQ']=base['fits']['joint']['noQ']['chi2']-power_best['chi2']
    summary['CPL_improvement_over_noQ']=base['fits']['joint']['noQ']['chi2']-cpl['chi2']
    summary['beta_profile_delta_range']=[min(y for _,y in profiles['beta']),max(y for _,y in profiles['beta'])]
    summary['alpha_profile_delta_range']=[min(y for _,y in profiles['alpha']),max(y for _,y in profiles['alpha'])]
    (out/'validation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    # Compact publication-independent review figures; all curves are fitted points.
    fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for a,key,label in zip(ax[:2],['beta','alpha'],[r'$\beta$',r'$\bar\alpha$']):
        x,y=np.array(profiles[key]).T;a.plot(x,y,'o-');a.set(xlabel=label,ylabel=r'$\Delta\chi^2$ respecto al mejor punto de potencia')
        a.set_ylim(-.01,max(.2,max(y)*1.15));a.grid(alpha=.25)
    ax[2].plot(wx,wy,'o-');ax[2].axhline(1,color='gray',ls=':');ax[2].axhline(3.841459,color='gray',ls='--')
    ax[2].set(xlabel=r'$\omega$ (sin Q)',ylabel=r'$\Delta\chi^2$');ax[2].grid(alpha=.25)
    fig.suptitle('Perfiles discretos: se reajustan los parámetros restantes')
    fig.savefig(out/'profiles.png',dpi=180);fig.savefig(out/'profiles.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    selected=['LCDM','noQ','constantQ','power_positive','power_negative_alpha_positive','power_negative_alpha_negative']
    for name in selected:
        f=trajectories[name]
        for a,key in zip(ax,['w_de','Q_over_3H02','Qdot_over_3H03']):a.plot(f.z,f[key],label=name)
    ax[0].plot(z,w0+wa*z/(1+z),'k--',label='CPL (referencia)');ax[0].axhline(-1,color='gray',ls=':')
    for a,label in zip(ax,[r'$w_{DE}$',r'$Q/(3H_0^2)$',r'$\dot Q/(3H_0^3)$']):a.set(xlabel='z',ylabel=label);a.grid(alpha=.25)
    ax[2].set_yscale('symlog',linthresh=.1);ax[0].legend(fontsize=6)
    fig.suptitle('Reconstrucciones de puntos de ajuste; no son bandas posteriores')
    fig.savefig(out/'reconstructions.png',dpi=180);fig.savefig(out/'reconstructions.pdf');plt.close(fig)
    # Report text uses only computed quantities.
    text='# Resultado de la comparación de Q\n\n'
    text+='Fecha local: 9 de noviembre de 2026. BAO DR2 + Pantheon+ cosmológico (1590 entradas).\n\n'
    text+='| Datos | Modelo | chi² | Delta vs noQ | Frontera | Control numérico |\n|---|---|---:|---:|---|---|\n'
    for row in rows:
        text+=f"| {row['dataset']} | {row['model']} | {row['chi2']:.6f} | {row['delta_vs_noQ']:.6f} | {row['boundary'] or row['near_D_floor']} | {row['numerical_pass']} |\n"
    text+='\n'
    text+=f"Mejora máxima encontrada de potencia frente a noQ: Delta chi²={summary['power_improvement_over_noQ']:.6f}.\n\n"
    text+=f"Rango del perfil discreto de beta: {summary['beta_profile_delta_range']}; se exploró beta de −6 a 6.\n\n"
    text+='Los valores explorados de beta no quedan separados por cambios apreciables de likelihood. No hay aquí una selección observacional de un exponente ni una detección de difusión. Esto no demuestra que todos los valores reales de beta sean equivalentes.\n\n'
    text+='Q lineal es exactamente equivalente a un fluido de w constante al nivel de fondo. Q constante no intercambia energía. La forma de potencia puede motivarse como ansatz fenomenológico con límites controlables, pero estos ajustes no justifican afirmar que sea una ley seleccionada por los datos.\n\n'
    text+=f"CPL: mejora frente a noQ={summary['CPL_improvement_over_noQ']:.6f}; cruce del mejor ajuste en el rango observado: {summary['CPL_best_fit_crossing_z']}. No se ha estimado significancia del cruce.\n\n"
    text+='Los cortes D0, rangos y signos de Q están registrados. Un indicador de entropía favorable sólo verifica -Qdot>=0 sobre la malla bajo T>0; no establece estabilidad termodinámica.\n\n'
    text+='Último paso realizado: validación independiente de densidades y distancias, revisión de perfiles y empaquetado reproducible. El siguiente paso científico es interpretar juntos el alcance para el manuscrito. Si se desea una afirmación de evidencia o significancia, hará falta una inferencia/calibración estadística específica, no convertir estos perfiles automáticamente en sigmas.\n'
    (out/'RESULTADOS.md').write_text(text)
    # Notebook is a reader of immutable results, not an expensive hidden rerun.
    rel=out.relative_to(ROOT)
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))
    md('# Viabilidad y elección de Q\n\nComparación terminada de perfiles. Leer RESULTADOS.md y docs/ESTUDIO_FUNCION_Q.md antes de interpretar parámetros. No son resultados posteriores ni evidencia bayesiana.\n')
    code("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nroot=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/observations').is_dir() or (p/'cosmo_singularities/src/observations').is_dir())\nif not (root/'src/observations').is_dir(): root=root/'cosmo_singularities'\nrun=root/"+repr(str(rel))+"\nsummary=json.loads((run/'validation_summary.json').read_text())\ndisplay(Markdown((run/'RESULTADOS.md').read_text()))\n")
    for fn in ['profiles.png','reconstructions.png']:
        code("display(Image(filename=str(run/"+repr(fn)+")))\n")
    code("display(pd.DataFrame(summary['validations']).T)\nprint('Umbrales condicionales noQ:',summary['noQ_conditional_profile_crossings'])\n")
    md('## Qué revisar juntos\n\n1. La compatibilidad del fondo no identifica por sí sola Q(rho).\n2. El perfil de beta y la equivalencia lineal limitan la justificación observacional del ansatz.\n3. La subfamilia phantom del paper se evalúa separadamente.\n4. Un cruce de CPL en un punto de ajuste no es una detección ni un cruce de nuestro modelo.\n\nLas ejecuciones originales y refinadas conservan sus intentos y fallos. Los resultados refinados incluyen semillas físicas para ramas estrechas.\n')
    notebook=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5)
    path=ROOT/'notebooks/11_viabilidad_y_eleccion_Q.ipynb'
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False))
    print(path);print(out/'RESULTADOS.md')

if __name__=='__main__':main()
