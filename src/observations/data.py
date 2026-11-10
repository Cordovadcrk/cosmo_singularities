"""Lectores de productos oficiales; selección idéntica en vector y covarianza."""
from pathlib import Path
import numpy as np
import pandas as pd
from .likelihood import BAOData,SNData

DEFAULT_DATA=Path(__file__).resolve().parents[2]/'data/observations'

def load_desi_dr2(folder=None):
    folder=Path(folder) if folder else DEFAULT_DATA/'desi_dr2'
    table=pd.read_csv(folder/'bao_mean.txt',sep=r'\s+',comment='#',header=None,names=['z','value','kind'])
    return BAOData(table.z,table.value,table.kind,np.loadtxt(folder/'bao_cov.txt'))

def load_pantheon_plus(folder=None,zmin=.01,selection='legacy_no_calibrators'):
    """SN sin calibración SH0ES: zHD>0.01, IS_CALIBRATOR=0, m_b_corr.

    Conservar observaciones repetidas: están correlacionadas en la matriz.
    No sumar errores diagonales a STAT+SYS (ya contiene estadística).
    """
    if zmin<0 or not np.isfinite(zmin):raise ValueError('zmin inválido.')
    folder=Path(folder) if folder else DEFAULT_DATA/'pantheon_plus'
    table=pd.read_csv(folder/'sn.dat',sep=r'\s+')
    raw=np.loadtxt(folder/'sn.cov')
    n=int(raw[0])
    if raw[0]!=n or n!=len(table) or raw.size!=1+n*n:raise ValueError('Covarianza Pantheon+ incompatible.')
    if selection not in ('cosmology','legacy_no_calibrators'):
        raise ValueError('Selección Pantheon+ desconocida.')
    mask=table.zHD>zmin
    if selection=='legacy_no_calibrators':mask &= table.IS_CALIBRATOR==0
    cov=raw[1:].reshape(n,n)[np.ix_(mask,mask)]
    result=SNData(table.zHD[mask],table.zHEL[mask],table.m_b_corr[mask],cov)
    result.selection=selection
    result.row_indices=np.flatnonzero(mask)
    return result
