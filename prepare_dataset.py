"""Recorte reproduzível do VOC2007: 40 aviões + 40 ovelhas, sem vazamento entre splits."""
from pathlib import Path
import urllib.request
import tarfile
import xml.etree.ElementTree as ET
import random, shutil, json, hashlib, csv

ROOT = Path(__file__).resolve().parent
CLASSES = ['aeroplane', 'sheep']
SEED = 57

def main():
    cache = ROOT / '_cache'
    cache.mkdir(exist_ok=True)
    archive = cache / 'VOCtrainval_06-Nov-2007.tar'
    voc = cache / 'VOCdevkit' / 'VOC2007'
    if not (voc / 'Annotations').exists():
        if not archive.exists():
            url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
            print('Baixando VOC2007:', url, flush=True)
            urllib.request.urlretrieve(url, archive)
        print('Extraindo VOC2007...', flush=True)
        with tarfile.open(archive) as tf:
            tf.extractall(cache, filter='data')
    rng = random.Random(SEED)
    records = []
    seen = set()
    for class_id, cls in enumerate(CLASSES):
        candidates = []
        for xml in sorted((voc / 'Annotations').glob('*.xml')):
            tree = ET.parse(xml).getroot()
            objects = tree.findall('object')
            targets = [o for o in objects if o.findtext('name') == cls and o.findtext('difficult') == '0']
            other = CLASSES[1-class_id]
            # O alvo precisa ser claro e a outra classe ausente, para a CNN de imagem inteira.
            if not targets or any(o.findtext('name') == other for o in objects):
                continue
            # Não ignoramos objetos difíceis: imagens que os contêm são excluídas.
            if any(o.findtext('name') == cls and o.findtext('difficult') != '0' for o in objects):
                continue
            w, h = int(tree.findtext('size/width')), int(tree.findtext('size/height'))
            boxes = []
            for obj in targets:
                box = obj.find('bndbox')
                x1, y1, x2, y2 = [float(box.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]
                boxes.append([class_id, (x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h])
            if max(b[3]*b[4] for b in boxes) < 0.08:
                continue
            candidates.append((xml.stem, boxes))
        rng.shuffle(candidates)
        if len(candidates) < 40:
            raise RuntimeError(f'Apenas {len(candidates)} candidatos para {cls}')
        for i, (stem, boxes) in enumerate(candidates[:40]):
            split = 'train' if i < 32 else 'val' if i < 36 else 'test'
            dest = ROOT / 'dataset' / 'images' / split
            labels = ROOT / 'dataset' / 'labels' / split
            dest.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            source = voc / 'JPEGImages' / f'{stem}.jpg'
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            assert digest not in seen, 'Imagem duplicada'
            seen.add(digest)
            shutil.copy2(source, dest / source.name)
            (labels / f'{stem}.txt').write_text('\n'.join(' '.join(f'{v:.6f}' if j else str(v) for j,v in enumerate(b)) for b in boxes)+'\n', encoding='utf-8')
            records.append({'image':source.name, 'class_id':class_id,'class':cls,'split':split,'sha256':digest,'objects':len(boxes)})
    with (ROOT/'dataset'/'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0])); wr.writeheader(); wr.writerows(records)
    (ROOT/'dataset'/'data.yaml').write_text('path: dataset\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: aeroplane\n  1: sheep\n', encoding='utf-8')
    (ROOT/'dataset'/'provenance.json').write_text(json.dumps({'source':'PASCAL VOC2007 trainval','source_url':'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/','seed':SEED,'annotation_origin':'VOC2007 XML convertido para YOLO; revisão/importação Make Sense registrada separadamente','notice':'Imagens pertencem aos respectivos autores. Uso educacional, conforme orientações e créditos da base VOC. Não se atribui licença autoral própria às imagens.','counts':{'train':64,'val':8,'test':8}},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Dataset concluído: 64 treino, 8 validação, 8 teste.', flush=True)

if __name__ == '__main__':
    main()
