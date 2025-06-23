# 📈 Projeto de Regressão Linear - Consumo de Cerveja

Este projeto foi desenvolvido como parte da disciplina de **Inteligência Artificial** da PUC Goiás. O objetivo foi construir um modelo de regressão linear utilizando o conjunto de dados `beer_consumption.csv`, relacionando a **temperatura** com o **consumo de cerveja**.

## 🎯 Objetivos

- Realizar análise exploratória de dados (EDA);
- Construir dois modelos de regressão:
  - Método dos Mínimos Quadrados (MMQ);
  - Gradiente Descendente (GD);
- Comparar os modelos com base em métricas estatísticas;
- Visualizar os dados com gráficos e correlações.

## 🧪 Tecnologias e Bibliotecas

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn

> ⚠️ **Importante:** O cálculo dos coeficientes de regressão foi feito **sem uso de bibliotecas externas como `sklearn`**, conforme exigido pela tarefa.

## 📁 Estrutura

- `Regressão_linear.py`: código principal com toda a implementação;
- `Relatório.pdf`: relatório textual com descrição das etapas e resultados;
- `Objetivo_Tarefa.pdf`: documento oficial da tarefa.

## 🗂️ Etapas Desenvolvidas

1. **Leitura e inspeção do dataset**
2. **Tratamento e verificação de dados faltantes**
3. **Visualização:**
   - Gráficos de barra, linha, histograma, dispersão, boxplot e heatmaps
4. **Modelagem:**
   - Regressão Linear com MMQ
   - Regressão Linear com Gradiente Descendente
5. **Avaliação:**
   - R², R² Ajustado, MSE, RMSE, MAE, MAPE, RMSLE

## 📊 Resultados

- Ambos os modelos obtiveram resultados **semelhantes**.
- O MMQ foi exato e mais rápido, enquanto o GD apresentou resultados próximos após convergência.
- A correlação entre **temperatura média** e **consumo de cerveja** foi consistente.

## 🧠 Autor

**Rodrigo F. N. Vieira**  
PUC Goiás – Ciência da Computação  
Disciplina: Inteligência Artificial  
Professor: Clarimar J. Coelho

## 📍 Local

Goiânia, 2025
