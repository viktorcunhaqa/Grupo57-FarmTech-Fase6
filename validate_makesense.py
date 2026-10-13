from pathlib import Path
import zipfile,json,numpy as np
r=Path(__file__).resolve().parent
z=zipfile.ZipFile(r/'rotulos_makesense_export.zip')
names=[n for n in z.namelist() if n.endswith('.txt') and Path(n).name!='labels.txt']
expected={p.name for p in (r/'dataset/labels/train').glob('*.txt')}
assert {Path(n).name for n in names}==expected and len(names)==64
diffs=[];boxes=0
for n in names:
    a=np.array([[float(v) for v in line.split()] for line in z.read(n).decode().splitlines()])
    b=np.loadtxt(r/'dataset/labels/train'/Path(n).name,ndmin=2)
    assert a.shape==b.shape
    assert np.array_equal(a[:,0],b[:,0]),n
    difference=float(np.max(np.abs(a-b)))
    boxes+=len(a)
    if difference:diffs.append({'file':n,'maximum_normalized_difference':difference})
    assert np.allclose(a,b,atol=2e-6,rtol=0),(n,difference)
report={'images':len(names),'boxes':boxes,'classes':['aeroplane','sheep'],
        'source':'VOC2007 -> COCO completo -> Make Sense -> exportação YOLO',
        'maximum_difference':max([d['maximum_normalized_difference'] for d in diffs],default=0),
        'note':'Make Sense arredonda coordenadas a seis decimais e ajusta caixas nas bordas. Nenhuma classe, imagem ou objeto mudou; diferenças numéricas <= 0.000002.',
        'differences':diffs}
(r/'annotation_review/makesense_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
