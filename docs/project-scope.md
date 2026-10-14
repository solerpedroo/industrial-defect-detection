# Industrial Defect Detection & Anomaly Analysis System
**Project Scope — Version 1.0**

## 1. Identificação

**Nome:** Industrial Defect Detection  
**Tipo:** Aplicação web integrada a um pipeline de Machine Learning  
**Área:** Computer Vision, Deep Learning e Industrial Anomaly Detection  
**Objetivo:** Portfólio técnico e preparação para Iniciação Científica  
**Repositório:** `industrial-defect-detection`  
**Dataset principal:** MVTec AD (CC BY-NC-SA 4.0, uso não comercial)  
**Linguagem principal:** Python  
**Interface:** Next.js + TypeScript

### 1.1 Objetivo geral

Desenvolver um sistema de inspeção visual capaz de identificar anomalias e localizar regiões potencialmente defeituosas em imagens industriais, usando abordagens diferentes de aprendizado de máquina.

O sistema prepara datasets, constrói modelos, executa inferência, gera mapas de anomalia, compara algoritmos e avalia desempenho quantitativo.

Diferente de um laboratório de processamento clássico, este projeto treina e compara modelos. O primeiro método é um autoencoder convolucional implementado no repositório. O segundo é PatchCore, integrado por um adaptador sobre Anomalib. PaDiM fica para uma fase posterior.

### 1.2 Objetivos específicos

1. Estudar métodos de detecção de anomalias não supervisionada.
2. Preparar dados com separação rigorosa entre treino, validação e teste.
3. Implementar um autoencoder convolucional em PyTorch, treinado só com imagens normais.
4. Integrar PatchCore por extração de características e memória de patches normais.
5. Localizar regiões anômalas com mapas de calor e máscaras.
6. Avaliar classificação por imagem e localização por pixel.
7. Comparar precisão, custo computacional e robustez.
8. Documentar limitações por categoria industrial.
9. Produzir experimentos reproduzíveis.

### 1.3 Perguntas de inspeção

1. Existe algum defeito? Classificação da imagem como normal ou anômala.
2. Onde está o defeito? Mapa de calor ou máscara.
3. Qual a extensão relativa? Área da máscara e score de anomalia. O score não é uma medida física de severidade nem um prejuízo financeiro.

## 2. Delimitação

A primeira versão detecta e localiza anomalias em imagens estáticas.

- **Modelo A:** Convolutional Autoencoder, treinado pelo projeto.
- **Modelo B:** PatchCore via Anomalib, sem backpropagation do backbone durante a construção da memória.
- **Modelo C (fora do MVP):** PaDiM.

O treino usa somente imagens normais. Imagens anômalas e máscaras servem à avaliação. Thresholds são definidos na validação, nunca no teste oficial.

### Fora do escopo do MVP

- Linha industrial contínua, CLPs, esteiras e maquinário.
- Garantia de uso em inspeção crítica.
- Classificação supervisionada de todos os tipos de defeito.
- Aplicativo mobile, treino multi-GPU, contas, cobrança e autenticação.
- Estimativa automática de prejuízo financeiro.
- Relatório PDF (Markdown, CSV, JSON e PNG entram no MVP).

## 3. Dataset

MVTec AD: 15 categorias, imagens normais de treino e máscaras de avaliação. Categorias iniciais:

| Categoria | Papel |
| --- | --- |
| Bottle | Objeto com geometria definida |
| Leather | Textura e superfície contínua |
| Metal Nut | Peça metálica |

O código pode ser público. As imagens do dataset não entram no Git.

## 4. Módulos funcionais

### Módulo 01 — Dataset Management

Importar um dataset local, ler a árvore do MVTec AD, listar categorias, contar imagens normais e anômalas, checar arquivos inválidos, mostrar amostras e máscaras, separar treino/validação/teste e impedir vazamento.

### Módulo 02 — Image Preprocessing

Redimensionar, normalizar RGB, converter para cinza quando necessário, aplicar augmentations compatíveis com a categoria e registrar a pipeline. Augmentations candidatas: rotação, espelhamento quando a orientação permitir, brilho, contraste, ruído gaussiano, desfoque e cor. Nem toda transformação é válida para todo produto.

### Módulo 03 — Convolutional Autoencoder

Encoder convolucional com ReLU e representação latente. Decoder com upsampling ou convoluções transpostas, reconstruindo o tamanho da entrada. Treino com Adam e MSE, early stopping na validação normal e checkpoint do melhor estado. A inferência devolve original, reconstrução, mapa de erro, score e rótulo segundo um threshold. Erro de reconstrução não é, por si só, um defeito.

### Módulo 04 — PatchCore

Backbone pré-treinado, características intermediárias das imagens normais, memória de patches, score e mapa de anomalia. Configuráveis: backbone, camadas, resolução, amostragem da memória e threshold. A lógica fica atrás de um adaptador do projeto.

### Módulo 05 — Training & Experiment Tracking

Scripts e CLI escolhem categoria, algoritmo e hiperparâmetros. Registram duração, seed, versões, hardware, hash de configuração e artefatos em YAML ou JSON. O treino não roda dentro de uma requisição HTTP.

### Módulo 06 — Anomaly Detection & Visualization

Upload, escolha de categoria e modelo, score, decisão binária, heatmap, overlay com opacidade, máscara binária, caixas derivadas da máscara e download. As caixas não implicam um detector de objetos.

### Módulo 07 — Model Evaluation

Por imagem: AUROC, Average Precision, precision, recall, F1 no threshold da validação, matriz de confusão, falsos positivos e falsos negativos.

Por pixel: AUROC, AUPR, AUPRO quando aplicável, IoU e Dice na máscara binária.

Computacionais: latência, tempo de preparo, tamanho do artefato, memória máxima aproximada e throughput nas condições documentadas.

### Módulo 08 — Model Comparison Dashboard

Selecionar experimentos, comparar métricas, scores e heatmaps da mesma imagem, latência, filtros por categoria, gráficos e exemplos de erro. Exportar tabelas e figuras.

### Módulo 09 — Experiment Reports

Objetivo, hipótese, partições, algoritmos, configurações, métricas, figuras, limitações e conclusões em Markdown, CSV, JSON e PNG.

## 5. Arquitetura alvo

Três componentes:

1. **Web application** — experimentação e leitura de resultados.
2. **Inference API** — modelos já preparados, predições, imagens e metadados.
3. **Training & Evaluation Pipeline** — scripts independentes.

Não há banco na primeira versão. Metadados ficam em arquivos.

**Frontend:** Next.js, TypeScript, Tailwind CSS, componentes no estilo shadcn/ui, Recharts quando houver gráficos.

**Backend:** FastAPI, Pydantic, PyTorch, Torchvision, OpenCV, NumPy, scikit-image, Anomalib, Albumentations. PyTorch e Anomalib entram nas waves de modelo, não na fundação.

**Qualidade:** logs estruturados locais (MLflow fica opcional), YAML, Pytest, Ruff, GitHub Actions.

## 6. Interface

Painel científico em inglês, tema escuro com tons azuis, menu lateral, cards de métricas, visualizadores sincronizados, gráficos e layout para desktop e tablet.

Páginas: Overview, Datasets, Inference Lab, Experiments, Model Comparison.

## 7. Requisitos não funcionais

- Separar treino e teste. Fixar o protocolo de threshold antes do teste final.
- Validar formato, tamanho e dimensões de upload. Não executar arquivos enviados.
- Carregar checkpoints só de diretórios confiáveis do projeto.
- Suportar CPU e GPU. Usar `torch.inference_mode()` na inferência. Não recarregar o mesmo modelo a cada request sem necessidade.
- Medir latência com aquecimento e sincronização de GPU quando houver GPU.
- Registrar seeds, versões e artefatos associados a cada resultado.

## 8. Experimentos científicos

**A — Autoencoder vs PatchCore.** Mesmas três categorias e o mesmo teste. Classificação, localização, memória e velocidade.

**B — Robustez sob degradação.** Mesmas imagens-base com ruído gaussiano (σ = 10, 20, 30 em pixels de 8 bits), desfoque gaussiano (σ = 1, 2, 3) e luminosidade (fatores 0,6; 0,8; 1,2).

**C — Tamanho do conjunto normal.** Subconjuntos de 25%, 50%, 75% e 100% das imagens normais de treino, com amostragens repetidas quando possível.

Se a validação tiver só imagens normais, o threshold pode mirar uma taxa-alvo de falsos positivos. O protocolo fica documentado e estável.

## 9. Configuração inicial de hardware

Ponto de partida para uma RTX 3050 de 6 GB, não um conjunto ótimo:

| Item | Valor |
| --- | --- |
| Resolução | 256 × 256 |
| Batch do autoencoder | 8 |
| Epochs máximas | 50, com early stopping |
| Otimizador | Adam |
| Learning rate | 0,001 |
| Backbone do PatchCore | ResNet-18 |
| Categorias | Bottle, Leather, Metal Nut |

A memória de características do PatchCore começa pequena.

## 10. Critérios de aceitação do MVP

- Três categorias preparadas, com treino, validação e teste isolados.
- Autoencoder treinado produz mapas de reconstrução.
- PatchCore constrói memória e infere.
- Os dois modelos expõem score e mapa.
- Métricas por imagem e por pixel.
- Thresholds sem vazamento do teste.
- A interface carrega uma imagem e escolhe um modelo compatível.
- O dashboard mostra resultados quantitativos e visuais, com exportação.
- Módulos essenciais cobertos por testes.
- README explica como reproduzir os experimentos.
- Pelo menos três experimentos comparativos documentados.

## 11. Evolução científica sugerida

*Comparative Analysis of Reconstruction-Based and Feature-Based Methods for Industrial Visual Anomaly Detection Under Image Degradation.*

A conclusão não assume qual família será superior. Depois do MVP: PaDiM, EfficientAD, MVTec AD 2, MVTec LOCO, few-shot, quantização, ONNX, borda e vídeo.

## 12. Referências

- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [Anomalib](https://github.com/open-edge-platform/anomalib)
- [PatchCore](https://arxiv.org/abs/2106.08265)
- [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)
