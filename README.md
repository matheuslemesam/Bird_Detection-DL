# Detecção de Aves do Cerrado com Deep Learning

Este repositório documenta o desenvolvimento de um projeto de classificação de espécies de aves do bioma Cerrado utilizando Redes Neurais Convolucionais (CNNs). O objetivo é construir um pipeline completo, desde o pré-processamento e aumento de dados até o treinamento e avaliação comparativa de diferentes arquiteturas de modelos.

O projeto foi desenvolvido como parte da disciplina de Tópicos Especiais em Matemática Aplicada da Universidade de Brasília (UnB/FCTE), ministrada pelo Professor Vinicius Rispoli no semestre 2025.2.

## Contribuidores

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com)

[`@matheuslemesam`](https://github.com/matheuslemesam) • [`@phenric26`](https://github.com/phenric26)

</div>

## Visão Geral do Projeto

O notebook `Bird_Detection.ipynb` foi estruturado para ser executado no Google Colab, tirando proveito de GPUs para acelerar o treinamento. O fluxo de trabalho pode ser dividido nas seguintes etapas:

### 1. Preparação do Ambiente e Dados
- **Google Drive:** O ambiente é configurado para se conectar ao Google Drive, onde o conjunto de dados de imagens está armazenado.
- **Estrutura de Pastas:** O dataset original reside em `dataset/original`, com cada uma das 14 espécies de aves em sua própria subpasta. As imagens processadas e aumentadas são salvas em `dataset/augmentation`.

### 2. Aumento de Dados (Data Augmentation)
Para enriquecer o dataset e torná-lo mais robusto, um pipeline de aumento de dados foi implementado utilizando exclusivamente a biblioteca **PyTorch**. As seguintes transformações são aplicadas:
- Rotações aleatórias
- Inversões horizontais
- Pequenas translações

Um fluxo de trabalho incremental garante que o processo seja resiliente a interrupções. Os dados são processados espécie por espécie, salvos localmente, sincronizados com o Google Drive e depois limpos, otimizando o uso de disco.

### 3. Abordagens de Treinamento com ResNet-50

Para avaliar diferentes estratégias de treinamento, o projeto focou na arquitetura **ResNet-50** e implementou três abordagens distintas:

- **ResNet-50 (do Zero):** O modelo foi construído camada por camada, sem o uso de pesos pré-treinados. Essa abordagem testa a capacidade da rede de aprender as características das imagens de aves apenas com o dataset fornecido.

- **Transfer Learning (Feature Extraction):** Utilizou-se uma ResNet-50 pré-treinada no dataset ImageNet. As camadas convolucionais foram "congeladas" (seus pesos não foram atualizados) e apenas a camada de classificação final foi treinada. O objetivo aqui é usar os extratores de características já aprendidos pelo modelo e adaptá-los rapidamente para a nova tarefa.

- **Transfer Learning - Fine-Tuning:** Similar à abordagem anterior, esta também partiu de uma ResNet-50 pré-treinada. No entanto, além de treinar a nova camada de classificação, a layer4 também foram "descongelada" e treinada com uma taxa de aprendizado baixa. Isso permite que o modelo ajuste fino das características já aprendidas para se especializar ainda mais no dataset de aves.

### 4. Treinamento e Avaliação

- **Preparação:** Para cada modelo, o dataset aumentado é carregado usando a classe `ImageFolder` do PyTorch, que automaticamente atribui os rótulos corretos. O conjunto é então dividido em treinamento e validação, e `DataLoaders` são criados para alimentar os dados aos modelos em lotes.
- **Ciclo de Treinamento:** Cada modelo é treinado por um número definido de épocas, e o progresso da acurácia e da perda (loss) de treinamento e validação é monitorado.
- **Métricas:** Ao final do treinamento, cada modelo é avaliado com base na acurácia, matriz de confusão e um relatório de classificação detalhado (precisão, recall, F1-score).

### 5. Resultados e Benchmark Final

Após o treinamento e a avaliação dos três modelos, os resultados foram consolidados para uma análise comparativa.

| Abordagem com ResNet-50 | Acurácia de Validação | Observações |
| :--- | :--- | :--- |
| **Fine-Tuning** | **94%** | Apresenta o maior desempenho pois utilizada a ResNet com a base do ImageNet e faz ajustes para se adequar ao nosso dataset. |
| **Feature Extraction** | **83%** | Melhor que a treinada do zero mas não tão boa quanto a do fine tuning. |
| **Treinada do Zero** | **25%** | Teve o desempenho mais baixo, pois requer um dataset muito grande para aprender características complexas do zero. |

**Conclusão do Benchmark:**

A abordagem de **Fine-Tuning** alcançou a maior acurácia, demonstrando que ajustar um modelo pré-treinado é uma estratégia altamente eficaz. A técnica de **Feature Extraction** serve como um excelente ponto de partida, enquanto treinar o modelo **do zero** se mostrou menos vantajoso e mais custoso computacionalmente, ressaltando a importância do Transfer Learning em tarefas de visão computacional com datasets de tamanho limitado.


