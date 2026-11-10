"""Validación pequeña reproducible; nunca sobrescribe datasets históricos.

Ejecutar desde cosmo_singularities: python -m scripts.validate_core
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

from src.models import BarotropicPowerDiffusionModel, BarotropicPowerDiffusionParams
from src.parameter_space import build_parameter_grid, analyze_parameter_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output_dir or Path('data/validation')/stamp
    output.mkdir(parents=True, exist_ok=False)
    cases = [(-.3, 1), (.3, 1.5), (.3, 2), (.3, 3), (4, -1), (-.3, 1.5)]
    grid = pd.concat([build_parameter_grid([a], [b], [-1.2]) for a,b in cases], ignore_index=True)
    grid['point_id'] = np.arange(len(grid))
    options = dict(t_end=2.0, n_points=160, density_decades=6, rtol=1e-9, atol=1e-11)
    frame = analyze_parameter_grid(grid, analysis_kwargs=options,
                                   checkpoint_path=output/'checkpoint.csv', checkpoint_every=1,
                                   verbose=False)
    # Campo separado: no confundir predicción analítica con etiqueta del clasificador.
    frame['analytic_reference'] = ['type_i_big_rip','type_i_big_rip','little rip',
                                   'little rip','diffusion_driven_infinite_time_rip',
                                   'higher_derivative_boundary']
    diagnostics = []
    for alpha,beta in cases:
        m = BarotropicPowerDiffusionModel(BarotropicPowerDiffusionParams(-1.2,alpha,beta,1))
        bg = m.solve_background_t(.1,n_points=50,rtol=1e-10,atol=1e-12)
        rho = bg.rho.to_numpy()
        friedmann = np.max(np.abs(3*bg.H.to_numpy()**2-rho-alpha*rho**beta)/(rho+np.abs(alpha*rho**beta)))
        # Sigma=T*dot(S)/(3VH)=-Q'(rho)*dot(rho)/(3H), usando el core.
        sigma = np.array([-(m.D(r)-1)*m.rho_dot(r)/(3*m.hubble(r)) for r in rho])
        diagnostics.append((friedmann,float(sigma.min()),float(sigma.max())))
    frame[['friedmann_relative_residual','sigma_min','sigma_max']] = diagnostics
    frame.to_csv(output/'summary.csv',index=False)
    metadata = {'utc':stamp, 'options':options, 'python':platform.python_version(),
                'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,
                'core_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path('src').glob('*.py'))},
                'scope':'Seis controles; no certifica convergencia asintótica ni novedad.'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(frame[['alpha','beta','behavior_label','analytic_reference','error']].to_string(index=False))
    print(f'Resultados: {output.resolve()}')
    if frame.error.notna().any() or frame.friedmann_relative_residual.max()>1e-10:
        raise SystemExit('Falló la validación; consultar summary.csv.')


if __name__ == '__main__':
    main()
