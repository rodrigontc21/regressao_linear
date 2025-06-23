# QUESTÃO (a) - IMPORTAÇÃO DE BIBLIOTECAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# QUESTÃO (b) - LEITURA DOS DADOS
df = pd.read_csv("C:/Users/didig/Downloads/Nova pasta/beer_consuption.csv")

# QUESTÕES (c) a (g) - EXPLORAÇÃO INICIAL
print(df.head())     # (c)
print(df.tail())     # (d)
print("Dimensão:", df.shape)  # (e)
print(df.isnull().sum())      # (f)
print(df.dtypes)              # (g)

# QUESTÃO (h) - CORRELAÇÃO DE PEARSON
print(df.corr(numeric_only=True))

# QUESTÃO (i) - ESTATÍSTICAS DESCRITIVAS
print(df.describe())

# QUESTÃO (j) - GRÁFICO DE FINAIS DE SEMANA 
plt.figure(figsize=(6,4))
cores = {'0': 'green', '1': 'blue'}
df['Final de Semana'] = df['Final de Semana'].astype(int)
sns.countplot(x='Final de Semana', data=df, palette=cores)
plt.title("Dias que são finais de semana")
plt.xlabel("Final de Semana (0 = Não, 1 = Sim)")
plt.ylabel("Frequência")
plt.grid(True)
plt.tight_layout()
plt.show()

# QUESTÃO (k) - GRÁFICO DAS TEMPERATURAS
df[['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)']].plot(figsize=(12,4))
plt.title("Temperaturas")
plt.show()

# QUESTÃO (l) - GRÁFICO DA PRECIPITAÇÃO
df['Precipitacao (mm)'].plot(figsize=(12,4))
plt.title("Precipitação Diária")
plt.show()

# QUESTÃO (m) - GRÁFICO DO CONSUMO 
df['Consumo de cerveja (litros)'].plot(figsize=(12,4), color='black')
plt.title("Consumo de Cerveja")
plt.xlabel("Índice")
plt.ylabel("Consumo (litros)")
plt.grid(True)
plt.tight_layout()
plt.show()

# QUESTÃO (n) - HEATMAP PEARSON 
plt.figure(figsize=(10, 6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap='RdYlGn_r',  # <- paleta invertida
    center=0,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Correlação'}
)
plt.title("Correlação de Pearson")
plt.tight_layout()
plt.show()

# QUESTÃO (o) - HEATMAP SPEARMAN 
plt.figure(figsize=(10, 6))
sns.heatmap(
    df.corr(method='spearman', numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap='RdYlGn_r',  
    center=0,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Correlação'}
)
plt.title("Correlação de Spearman")
plt.tight_layout()
plt.show()

# QUESTÃO (p) - BOXPLOTS AJUSTADOS
variaveis = ['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)',
             'Precipitacao (mm)', 'Consumo de cerveja (litros)']
plt.figure(figsize=(15, 6))
sns.set(style="whitegrid")
for i, var in enumerate(variaveis, 1):
    plt.subplot(1, 5, i)
    sns.boxplot(
        y=df[var],
        color='skyblue',
        boxprops=dict(facecolor='lightblue'),
        medianprops=dict(color='red'),
        flierprops=dict(marker='o', markersize=4, markerfacecolor='black', markeredgecolor='black')
    )
    plt.title(var)
    plt.grid(True)
plt.tight_layout()
plt.show()

# QUESTÃO (q) - HISTOGRAMAS
df[variaveis].hist(figsize=(12,8))
plt.show()

# QUESTÃO (r) - GRÁFICOS DE DISPERSÃO
for col in variaveis[:-1]:
    plt.scatter(df[col], df['Consumo de cerveja (litros)'])
    plt.xlabel(col)
    plt.ylabel("Consumo de Cerveja")
    plt.title(f"Dispersão entre {col} e Consumo")
    plt.show()

# QUESTÃO (s) - REGRESSÃO LINEAR MÚLTIPLA
print("""\n
(s) Construir um modelo de regressão linear múltipla em Python
    (a) Usando MMQ
    (b) Gradiente descendente
    (c) Comparar os resultados.
""")

X = df[['Temperatura Media (C)', 'Temperatura Maxima (C)', 'Temperatura Minima (C)',
        'Precipitacao (mm)', 'Final de Semana']].values
y = df['Consumo de cerveja (litros)'].values
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# (s.a) MMQ
beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print("\nCoeficientes (MMQ):", beta)

# (s.b) Gradiente Descendente
def gradient_descent(X, y, alpha=0.0001, epochs=1000):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(X_b.shape[1])
    for _ in range(epochs):
        gradients = 2/m * X_b.T @ (X_b @ theta - y)
        theta -= alpha * gradients
    return theta

theta = gradient_descent(X, y)
print("\nCoeficientes (Gradiente Descendente):", theta)

# QUESTÃO (t) - CÁLCULO DAS MÉTRICAS
def calcular_metricas(y_true, y_pred, X):
    n = len(y_true)
    p = X.shape[1]
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    r2_adj = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

    print(f"R²: {r2:.4f}")
    print(f"R² ajustado: {r2_adj:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSLE: {rmsle:.4f}")

print("\n--- MÉTRICAS PARA MMQ ---")
y_pred_mmq = X_b @ beta
calcular_metricas(y, y_pred_mmq, X)

print("\n--- MÉTRICAS PARA GRADIENTE DESCENDENTE ---")
y_pred_gd = X_b @ theta
calcular_metricas(y, y_pred_gd, X)

# QUESTÃO (s.c) - GRÁFICO DE REGRESSÃO USANDO TEMPERATURA MÉDIA (C)
x_plot = df['Temperatura Media (C)'].values
x_plot_reshape = x_plot.reshape(-1, 1)
x_plot_b = np.c_[np.ones((x_plot.shape[0], 1)), x_plot]

# MMQ com 1 variável
beta_simples = np.linalg.inv(x_plot_b.T @ x_plot_b) @ x_plot_b.T @ y
y_pred_mmq_simples = x_plot_b @ beta_simples

# GD com 1 variável
def gd_simples(x, y, alpha=0.0001, epochs=1000):
    m = len(y)
    x_b = np.c_[np.ones((m, 1)), x]
    theta = np.random.randn(x_b.shape[1])
    for _ in range(epochs):
        gradients = 2/m * x_b.T @ (x_b @ theta - y)
        theta -= alpha * gradients
    return theta

theta_simples = gd_simples(x_plot_reshape, y)
y_pred_gd_simples = x_plot_b @ theta_simples

plt.figure(figsize=(8, 5))
plt.scatter(x_plot, y, color='skyblue', alpha=0.7, label='Dados Reais')
plt.plot(x_plot, y_pred_mmq_simples, color='blue', label='MMQ')
plt.plot(x_plot, y_pred_gd_simples, color='red', linestyle='--', label='Gradiente Descendente')
plt.xlabel("Temperatura Média (C)")
plt.ylabel("Consumo de cerveja (litros)")
plt.title("(s) Comparativo de Modelos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
