# FarmTech Solutions | Grupo 57 | Fase 6

**Viktor Ribeiro dos Santos Cunha — RM570917**  
FIAP — O despertar da Rede Neural — entrega solo do Grupo 57.

Demonstração funcional de visão computacional com **aviões e ovelhas**, usando 80 imagens do PASCAL VOC2007. O projeto compara YOLOv5u customizado em dois treinamentos independentes (30 e 60 épocas), YOLOv5u padrão pré-treinado no COCO e uma CNN treinada do zero.

## Comece pelo notebook

- [Notebook executado no GitHub](ViktorRibeiroDosSantosCunha_rm570917_pbl_fase6.ipynb)
- [Abrir no Google Colab](https://colab.research.google.com/github/viktorcunhaqa/Grupo57-FarmTech-Fase6/blob/main/ViktorRibeiroDosSantosCunha_rm570917_pbl_fase6.ipynb)
- [Vídeo demonstrativo não listado — 4min54s](https://youtu.be/BpnzcV_1Fmk)
- [Organização no Google Drive pessoal](https://drive.google.com/drive/folders/11EauoXtrl-vOgP1YcRnT8a3CgXvgN67Q)

O notebook contém o passo a passo, as saídas executadas, os gráficos, a inferência nas imagens de teste, a comparação crítica e as limitações. No Colab, ele monta o Drive e prepara os pacotes. As imagens e os pesos também acompanham o repositório, de modo que a reprodução não depende de acesso ao Drive pessoal do autor.

## Arquivos e reprodução

| Arquivo | Função |
|---|---|
| `dataset.zip` | 80 imagens, 80 rótulos, manifesto SHA-256, YAML e origem |
| `results.zip` | Pesos `best.pt`, logs CSV, curvas, métricas JSON e imagens de teste |
| `yolov5nu.pt` | Pesos COCO usados como mesma inicialização e como baseline |
| `experiments.py` | Implementações comentadas de treino, avaliação e comparação |
| `prepare_dataset.py` | Seleção reproduzível do VOC2007, com semente 57 |
| `rotulos_treino_makesense.zip` | TXT YOLO das 64 imagens de treino e definição das classes |
| `annotation_review.zip` | Folhas de revisão de todas as 64 imagens e evidência da ferramenta |
| `requirements.txt` | Dependências da execução |

No Colab, use GPU e execute as células em ordem. Por padrão, o notebook usa os resultados entregues e **executa novamente as inferências**. Para repetir os treinamentos completos, defina `RETRAIN=True` na célula indicada. Execute em uma cópia local/Drive e preserve a versão entregue.

Para executar localmente:

```bash
python -m pip install -r requirements.txt
# Abra o notebook na raiz do projeto. Ele extrai dataset.zip e results.zip.
# Alternativamente, extraia os pacotes e execute:
python experiments.py
```

`experiments.py` foi executado com Python 3.12 e RTX 2060 SUPER. Os resultados originais estão registrados em `results/environment.json`. Sem GPU, a execução permanece possível, mas o treinamento será mais lento. A organização 32/4/4 por classe é auditada no notebook.

## O que a demonstração mostra

As duas redes customizadas e a CNN acertaram a classe das 8 imagens de teste; o baseline acertou 7/8. **Isso não equivale a detecção perfeita.** O relatório mostra falsos positivos, falsos negativos e métricas de localização. O teste pequeno limita a generalização. O modelo de 30 épocas foi preferido pela validação; não se selecionou o modelo usando o teste.

Os pesos customizados usam a variante **YOLOv5u**, explicitada no relatório, e não a implementação legada do YOLOv5 com anchors. A CNN não localiza objetos e não possui mAP comparável aos detectores. O notebook explica as diferenças das tarefas e das medições de tempo.

## Origem e créditos

Imagens e caixas de referência: [PASCAL VOC2007](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/). As anotações XML originais foram convertidas para YOLO; a etapa de importação/revisão no Make Sense é identificada nas evidências, sem atribuir autoria original às marcações. Os direitos das imagens permanecem com seus autores; este é um uso educacional da base, com origem registrada.

Modelos e biblioteca: [Ultralytics YOLOv5u](https://docs.ultralytics.com/models/yolov5/). Rede CNN e rotina de aprendizagem: [PyTorch](https://docs.pytorch.org/docs/stable/index.html). Rotulação: [Make Sense](https://www.makesense.ai/). O vídeo utiliza narração sintetizada em português e evidências reais da execução.

As atividades opcionais **Ir Além** não compõem esta entrega obrigatória.

**Versão de entrega: 13/10/2026. Não realizar commits após o envio à FIAP.**
