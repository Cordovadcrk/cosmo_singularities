"""Figuras comparativas del paper, calculadas con el core existente.

Desde cosmo_singularities: python -m scripts.plot_diffusion_beta
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator
from src.models import BarotropicPowerDiffusionParams, BarotropicPowerDiffusionModel
from src.barotropic_analysis import state_at_density

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'figures' / 'comparacion_beta'
BETAS = (-0.5, -1.0, -2.0, -3.0)
STYLES = [('blue', '-'), ('teal', '--'), ('#ff7f0e', '-.'), ('red', ':')]
PANELS = [('rho', r'$\rho/\rho_0$'), ('Q', r'$Q/\rho_0$'),
          ('H', r'$H/\sqrt{\rho_0}$'), ('Hdot', r'$\dot H/\rho_0$'),
          ('N', r'$\ln(a/a_0)$'), ('Qdot', r'$\dot Q/\rho_0^{3/2}$')]


def calculate(omega=-1.2, alpha_bar=4.0, rho0=1.0, a0=1.0, points=550):
    gamma = -(1+omega)
    if gamma <= 0:
        raise ValueError('Se requiere omega<-1.')
    # Densidad monótona: resolver por cuadratura evita integrar hasta t~1e21.
    # Muestreo extra cerca de t0 para que las curvas comiencen suavemente.
    x = np.unique(np.r_[np.geomspace(1e-8, .95, points),
                        1-np.geomspace(1e-5, .05, 100)])[::-1]
    frames, checks = [], []
    for beta in BETAS:
        params = BarotropicPowerDiffusionParams(omega, alpha_bar*rho0**(1-beta), beta, rho0, a0)
        model = BarotropicPowerDiffusionModel(params)
        if model.D(rho0) >= 0:
            raise ValueError('Las figuras requieren la rama D0<0 para cada beta.')
        rows = []
        for ratio in x:
            state = state_at_density(params, rho0*ratio, epsabs=1e-10, epsrel=1e-10)
            r = state['rho']
            N = state['log_a']-np.log(a0)
            rows.append(dict(beta=beta, tau=gamma*np.sqrt(rho0/3)*(state['t']-params.t0),
                             rho=ratio, Q=model.diffusion(r)/rho0,
                             H=state['H']/np.sqrt(rho0), Hdot=state['Hdot']/rho0,
                             N=N, Qdot=(model.D(r)-1)*model.rho_dot(r)/rho0**1.5,
                             F=state['F'], D=state['D'], quadrature_error=state['quadrature_error']))
        frame = pd.DataFrame(rows)
        assert (np.diff(frame.tau)>0).all() and (frame.D<0).all()
        residual = float(np.max(np.abs(3*frame.H**2-frame.rho-frame.Q)/(frame.rho+frame.Q)))
        assert residual < 1e-12
        spread = 0.0
        for target in [.95, 1e-3, 1e-8]:
            times = [state_at_density(params, rho0*target, epsabs=e, epsrel=e)['t'] for e in [1e-8,1e-12]]
            spread = max(spread, abs(times[0]/times[1]-1))
        assert spread < 1e-7
        checks.append(dict(beta=beta, friedmann_residual=residual, time_tolerance_relative_spread=spread))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), checks


def draw(ax, data, key, ylabel, full=True):
    for beta, (color, style) in zip(BETAS, STYLES):
        frame = data.loc[data.beta==beta]
        if key == 'a':
            frame = frame.loc[frame.N <= 80*np.log(10)]
            values = np.exp(frame.N)
        else:
            values = frame[key]
        ax.plot(frame.tau, values, color=color, ls=style, lw=1.5,
                label=rf'$\beta={beta:g}$')
    ax.set(xscale='log', yscale='log', xlabel=r'$\tau$', ylabel=ylabel)
    if full:
        ax.set_xlim(1e-5, 1e21)
        ax.set_xticks([1e-5, 1, 1e5, 1e10, 1e15, 1e20])
    ax.yaxis.set_major_locator(LogLocator(numticks=5))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which='major', color='#bfbfbf', linewidth=.65)
    ax.legend(loc='best', framealpha=1, edgecolor='black', fancybox=False, fontsize=10)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'cm', 'font.size':12,
                         'axes.labelsize':15, 'svg.fonttype':'none', 'savefig.dpi':200})
    data, checks = calculate()
    data.to_csv(OUTPUT/'trajectories.csv', index=False)
    for key, label in PANELS+[('a', r'$a/a_0$')]:
        fig, ax = plt.subplots(figsize=(6.2,4.5), layout='constrained')
        draw(ax, data, key, label, full=key!='a')
        if key=='a':
            ax.set_yticks([1,1e20,1e40,1e60,1e80])
            ax.set_title(r'Vista limitada: $a/a_0\leq 10^{80}$', fontsize=11)
        for extension in ('png','svg'):
            fig.savefig(OUTPUT/f'{key}_comparacion_beta.{extension}')
        plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(15.5,8.2),layout='constrained')
    for ax,(key,label) in zip(axes.flat,PANELS):
        draw(ax,data,key,label)
    fig.suptitle(r'$\omega=-1.2,\quad\bar\alpha=4,\quad\beta=-0.5,-1,-2,-3$',fontsize=17)
    for extension in ('png','svg'):
        fig.savefig(OUTPUT/f'vista_conjunta.{extension}')
    plt.close(fig)
    metadata = dict(omega=-1.2,alpha_bar=4,rho0=1,a0=1,betas=BETAS,
                    tau='gamma*sqrt(rho0/3)*(t-t0)', density_cutoff=1e-8,
                    note='Mismo corte de densidad; distinto tiempo final para cada beta. N=ln(a/a0) no es a.',
                    verification=checks,
                    core_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')})
    (OUTPUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    print(OUTPUT)


if __name__=='__main__':
    main()
