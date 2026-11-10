"""Descarga productos públicos sin sobrescribir archivos; registra commits y SHA256."""
from pathlib import Path
import subprocess
from urllib.parse import quote
import json
import hashlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]/'data/observations'
SOURCES = {
 'desi_dr2': ('CobayaSampler/bao_data', 'master', {
   'bao_mean.txt':'desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt',
   'bao_cov.txt':'desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt'}),
 'pantheon_plus': ('PantheonPlusSH0ES/DataRelease', 'main', {
   'sn.dat':'Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat',
   'sn.cov':'Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov',
   'README.txt':'Pantheon+_Data/4_DISTANCES_AND_COVAR/README'})}

def fetch(url):
    return subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                           '--max-time', '120', url], check=True, capture_output=True).stdout

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    for name,(repo,branch,files) in SOURCES.items():
        folder=ROOT/name; folder.mkdir(exist_ok=True)
        manifest=folder/'manifest.json'
        if manifest.exists():
            records=json.loads(manifest.read_text())
            for item in records['files']:
                if hashlib.sha256((folder/item['name']).read_bytes()).hexdigest()!=item['sha256']:
                    raise ValueError('Archivo modificado: '+item['name'])
            print(name, 'ya descargado y verificado'); continue
        commit=json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/{branch}'))['sha']
        records=[]
        for local,remote in files.items():
            url=f'https://raw.githubusercontent.com/{repo}/{commit}/'+quote(remote,safe='/')
            data=fetch(url); path=folder/local
            if path.exists() and path.read_bytes()!=data:
                raise FileExistsError(f'No sobrescribir {path}')
            path.write_bytes(data)
            records.append(dict(name=local,url=url,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
            print(name,local,len(data),flush=True)
        manifest.write_text(json.dumps(dict(repository=repo,commit=commit,
            downloaded_utc=datetime.now(timezone.utc).isoformat(),files=records),indent=2)+'\n')

if __name__=='__main__':main()
