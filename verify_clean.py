"""Valida a execução a partir apenas dos arquivos portáteis de entrega."""
from pathlib import Path
import shutil, nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parent
target=root/'_verification';target.mkdir(exist_ok=True)
name='ViktorRibeiroDosSantosCunha_rm570917_pbl_fase6.ipynb'
for file in [name,'experiments.py','dataset.zip','results.zip','yolov5nu.pt']:
    shutil.copy2(root/file,target/file)
nb=nbformat.read(target/name,as_version=4)
nbformat.validate(nb)
NotebookClient(nb,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(target)}}).execute()
assert not [o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
print('Verificação em pasta limpa: 8 células executadas, 0 erros; dataset e pesos extraídos dos ZIPs.')
