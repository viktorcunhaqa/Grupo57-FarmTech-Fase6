"""Treinamento e avaliação reais do projeto FarmTech, Grupo 57.

Detector e classificador têm tarefas diferentes: mAP mede localização e classe;
acurácia de imagem mede apenas a classe dominante. Nunca equiparamos as métricas.
"""
from pathlib import Path
import time, json, random, platform, csv
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
SEED = 57
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NAMES = ['aeroplane', 'sheep']

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)

def dataset_summary():
    manifest = pd.read_csv(ROOT/'dataset'/'manifest.csv')
    assert manifest['sha256'].nunique() == 80
    assert manifest.groupby(['class','split']).size().to_dict() == {
        ('aeroplane','train'):32,('aeroplane','val'):4,('aeroplane','test'):4,
        ('sheep','train'):32,('sheep','val'):4,('sheep','test'):4}
    return manifest.groupby(['class','split']).size().unstack()

def train_yolo(epochs):
    seed_all()
    name = f'yolo_{epochs}'
    # Mesma inicialização COCO nas duas experiências; não é continuação de treino.
    model = YOLO(str(ROOT/'yolov5nu.pt'))
    t0 = time.perf_counter()
    model.train(data=str(ROOT/'dataset'/'runtime.yaml'),epochs=epochs,
        imgsz=320,batch=8,device=0 if torch.cuda.is_available() else 'cpu',
        workers=0,seed=SEED,deterministic=True,patience=0,
        project=str(OUT),name=name,exist_ok=True,plots=True,
        amp=False,cache=False,verbose=False,optimizer='AdamW',lr0=0.001,
        close_mosaic=0,save=True)
    seconds = time.perf_counter()-t0
    model = YOLO(str(OUT/name/'weights'/'best.pt'))
    scores = {}
    for split in ['val','test']:
        metrics=model.val(data=str(ROOT/'dataset'/'runtime.yaml'),split=split,
            imgsz=320,batch=8,device=0 if torch.cuda.is_available() else 'cpu',
            workers=0,plots=True,project=str(OUT),name=f'{name}_{split}',exist_ok=True,verbose=False)
        scores[split] = {k:float(v) for k,v in metrics.results_dict.items()}
    result={'model':name,'epochs':epochs,'train_seconds':seconds,'metrics':scores}
    (OUT/f'{name}_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return result

def bbox_iou(a,b):
    x1,y1=np.maximum(a[:2],b[:2]);x2,y2=np.minimum(a[2:],b[2:])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter+1e-9)

def evaluate_detector(model, tag, baseline=False):
    """Mesmo teste, limiar de confiança 0,25 e IoU 0,50 para todos os detectores."""
    manifest=pd.read_csv(ROOT/'dataset'/'manifest.csv')
    test=manifest[manifest.split=='test'].sort_values('image')
    photos=[Image.open(ROOT/'dataset'/'images'/'test'/n).convert('RGB') for n in test.image]
    def infer():
        return model.predict(photos,imgsz=320,conf=0.25,iou=0.7,
            device=0 if torch.cuda.is_available() else 'cpu',verbose=False,
            classes=[4,18] if baseline else None)
    for _ in range(3): infer()
    latencies=[]
    for _ in range(5):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0=time.perf_counter();preds=infer()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        latencies.append((time.perf_counter()-t0)*1000/len(photos))
    folder=OUT/tag;folder.mkdir(exist_ok=True)
    totals=np.zeros((2,3),int); correct=0; rows=[]
    for row,img,pred in zip(test.to_dict('records'),photos,preds):
        xyxy=pred.boxes.xyxy.cpu().numpy(); cl=pred.boxes.cls.cpu().numpy().astype(int); cf=pred.boxes.conf.cpu().numpy()
        if baseline: cl=np.array([{4:0,18:1}[int(c)] for c in cl])
        # Para comparar com CNN: maior confiança define uma única classe por imagem.
        guess=int(cl[np.argmax(cf)]) if len(cf) else -1
        correct+=int(guess==row['class_id'])
        lines=(ROOT/'dataset'/'labels'/'test'/Path(row['image']).with_suffix('.txt')).read_text().splitlines()
        gt=[]
        for line in lines:
            c,cx,cy,w,h=map(float,line.split());W,H=img.size
            gt.append((int(c),np.array([(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H])))
        matched=set()
        for j in np.argsort(-cf):
            possibilities=[(bbox_iou(xyxy[j],box),k) for k,(c,box) in enumerate(gt) if c==cl[j] and k not in matched]
            iou,k=max(possibilities,default=(0,-1))
            if iou>=0.5: totals[cl[j],0]+=1;matched.add(k)
            else: totals[cl[j],1]+=1
        for k,(c,_) in enumerate(gt):
            if k not in matched: totals[c,2]+=1
        pred.save(filename=str(folder/row['image']))
        rows.append({'image':row['image'],'true':row['class_id'],'predicted':guess,'correct':guess==row['class_id'],'detections':len(cf)})
    tp,fp,fn=totals.sum(axis=0)
    details=[]
    for c,(ct,cp,cn) in enumerate(totals):
        details.append({'class':NAMES[c],'tp':int(ct),'fp':int(cp),'fn':int(cn),'precision':float(ct/(ct+cp)) if ct+cp else 0.,'recall':float(ct/(ct+cn)) if ct+cn else 0.})
    result={'model':tag,'test_images':len(test),'image_accuracy':correct/len(test),
        'precision_at_conf025_iou05':float(tp/(tp+fp)) if tp+fp else 0.,
        'recall_at_conf025_iou05':float(tp/(tp+fn)) if tp+fn else 0.,
        'tp':int(tp),'fp':int(fp),'fn':int(fn),'inference_ms_per_image':float(np.median(latencies)),
        'per_class':details,'images':rows}
    (OUT/f'{tag}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

class ImageDataset(Dataset):
    def __init__(self,split,augment=False):
        self.rows=pd.read_csv(ROOT/'dataset'/'manifest.csv').query('split == @split').sort_values('image').to_dict('records')
        self.transform=transforms.Compose([transforms.Resize((96,96)),
            *([transforms.RandomHorizontalFlip(),transforms.ColorJitter(0.15,0.15,0.15)] if augment else []),
            transforms.ToTensor(),transforms.Normalize([0.5]*3,[0.5]*3)])
        self.split=split
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]; img=Image.open(ROOT/'dataset'/'images'/self.split/row['image']).convert('RGB')
        return self.transform(img),int(row['class_id'])

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Inicialização aleatória: nenhum peso pré-treinado nesta CNN.
        self.network=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),nn.Dropout(0.25),nn.Linear(64*4*4,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,x):return self.network(x)

def train_cnn(epochs=30):
    seed_all();net=TinyCNN().to(DEVICE)
    train=DataLoader(ImageDataset('train',True),batch_size=8,shuffle=True,num_workers=0)
    val=DataLoader(ImageDataset('val'),batch_size=8,shuffle=False,num_workers=0)
    criterion=nn.CrossEntropyLoss();optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
    history=[];best=(float('-inf'),float('-inf'));folder=OUT/'cnn';folder.mkdir(exist_ok=True)
    t0=time.perf_counter()
    for epoch in range(epochs):
        net.train();tl=0.;tc=0
        for x,y in train:
            x,y=x.to(DEVICE),y.to(DEVICE);optimizer.zero_grad();logits=net(x)
            loss=criterion(logits,y);loss.backward();optimizer.step()
            tl+=loss.item()*len(y);tc+=int((logits.argmax(1)==y).sum())
        net.eval();vl=0.;vc=0
        with torch.no_grad():
            for x,y in val:
                x,y=x.to(DEVICE),y.to(DEVICE);logits=net(x)
                vl+=criterion(logits,y).item()*len(y);vc+=int((logits.argmax(1)==y).sum())
        entry={'epoch':epoch+1,'train_loss':tl/64,'train_accuracy':tc/64,'val_loss':vl/8,'val_accuracy':vc/8}
        history.append(entry)
        key=(vc/8,-vl/8)
        if key>best:best=key;torch.save(net.state_dict(),folder/'best.pt')
        print('CNN',entry,flush=True)
    seconds=time.perf_counter()-t0
    net.load_state_dict(torch.load(folder/'best.pt',map_location=DEVICE,weights_only=True));net.eval()
    test=ImageDataset('test');batch=torch.stack([test[i][0] for i in range(len(test))]).to(DEVICE)
    labels=torch.tensor([test[i][1] for i in range(len(test))],device=DEVICE)
    times=[]
    with torch.no_grad():
        for _ in range(3):net(batch)
        for _ in range(5):
            if torch.cuda.is_available():torch.cuda.synchronize()
            begin=time.perf_counter();logits=net(batch)
            if torch.cuda.is_available():torch.cuda.synchronize()
            times.append((time.perf_counter()-begin)*1000/len(test))
        preds=logits.argmax(1);loss=criterion(logits,labels).item()
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(labels.cpu().numpy(),preds.cpu().numpy(),labels=[0,1])
    fig,axs=plt.subplots(1,3,figsize=(12,3.4));hist=pd.DataFrame(history)
    hist.plot(x='epoch',y=['train_loss','val_loss'],ax=axs[0]);hist.plot(x='epoch',y=['train_accuracy','val_accuracy'],ax=axs[1])
    axs[2].imshow(cm,cmap='Blues');axs[2].set_xticks([0,1],NAMES);axs[2].set_yticks([0,1],NAMES)
    axs[2].set_xlabel('Previsto');axs[2].set_ylabel('Real');axs[2].set_title('Teste CNN: matriz de confusão')
    for i in range(2):
        for j in range(2):axs[2].text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.tight_layout();fig.savefig(folder/'curves.png',dpi=150);plt.close(fig)
    hist.to_csv(folder/'history.csv',index=False)
    result={'model':'cnn_from_scratch','epochs':epochs,'parameters':sum(p.numel() for p in net.parameters()),
        'train_seconds':seconds,'best_validation_accuracy':best[0],'test_loss':loss,
        'image_accuracy':float((preds==labels).float().mean()),'inference_ms_per_image':float(np.median(times)),
        'confusion_matrix':cm.tolist(),'images':[{'image':r['image'],'true':r['class_id'],'predicted':int(p)} for r,p in zip(test.rows,preds.cpu())]}
    (OUT/'cnn_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def comparison():
    det30=json.loads((OUT/'yolo_30_test_eval_evaluation.json').read_text())
    det60=json.loads((OUT/'yolo_60_test_eval_evaluation.json').read_text())
    base=json.loads((OUT/'baseline_coco_evaluation.json').read_text())
    cnn=json.loads((OUT/'cnn_summary.json').read_text())
    rows=[]
    for tag,d in [('YOLOv5u 30 épocas',det30),('YOLOv5u 60 épocas',det60),('YOLOv5u COCO padrão',base),('CNN do zero',cnn)]:
        if '30 épocas' in tag:train=json.loads((OUT/'yolo_30_summary.json').read_text());ep=30
        elif '60 épocas' in tag:train=json.loads((OUT/'yolo_60_summary.json').read_text());ep=60
        else:train=None;ep=0 if 'COCO' in tag else 30
        rows.append({'modelo':tag,'epocas':ep,'acuracia_imagem_teste':d['image_accuracy'],
            'erro_imagem_teste':1-d['image_accuracy'],'precisao_deteccao':d.get('precision_at_conf025_iou05',np.nan),
            'recall_deteccao':d.get('recall_at_conf025_iou05',np.nan),
            'mAP50_teste':train['metrics']['test']['metrics/mAP50(B)'] if train else np.nan,
            'mAP50_95_teste':train['metrics']['test']['metrics/mAP50-95(B)'] if train else np.nan,
            'treinamento_s':train['train_seconds'] if train else d.get('train_seconds',0),
            'inferencia_ms_imagem':d['inference_ms_per_image']})
    df=pd.DataFrame(rows);df.to_csv(OUT/'comparison.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(12,4));axs[0].bar(df.modelo,df.acuracia_imagem_teste,color=['#2e86ab','#26734d','#b47712','#775a99'])
    axs[0].set_ylim(0,1.05);axs[0].set_title('Acurácia de imagem no teste (8 imagens)');axs[0].tick_params(axis='x',rotation=20)
    axs[1].bar(df.modelo,df.inferencia_ms_imagem,color=['#2e86ab','#26734d','#b47712','#775a99']);axs[1].set_title('Tempo por imagem (ms; mediana de 5 repetições)');axs[1].tick_params(axis='x',rotation=20)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    return df

def main():
    import os
    os.chdir(ROOT)
    import ultralytics, torchvision
    environment={'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
        'ultralytics':ultralytics.__version__,'device':DEVICE,
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'seed':SEED}
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    print(environment,flush=True);print(dataset_summary(),flush=True)
    (ROOT/'dataset'/'runtime.yaml').write_text(f'path: {str(ROOT/"dataset").replace(chr(92),"/")}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: aeroplane\n  1: sheep\n',encoding='utf-8')
    if not (ROOT/'yolov5nu.pt').exists():
        YOLO('yolov5nu.pt')
    for ep in [30,60]:
        if not (OUT/f'yolo_{ep}_summary.json').exists():train_yolo(ep)
        evaluate_detector(YOLO(str(OUT/f'yolo_{ep}'/'weights'/'best.pt')),f'yolo_{ep}_test_eval')
    evaluate_detector(YOLO(str(ROOT/'yolov5nu.pt')),'baseline_coco',baseline=True)
    train_cnn();print(comparison().to_string(index=False),flush=True)

if __name__=='__main__':main()
