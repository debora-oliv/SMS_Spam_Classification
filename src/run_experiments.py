import pandas as pd
import numpy as np
import os
import scipy
import time
import preprocessing as prep
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.feature_extraction.text import TfidfVectorizer # pip install scikit-learn
from sklearn.model_selection import train_test_split

import logistic_regression_iterative as iter
import logistic_regression_vect as vect

# Esta variável define quantas épocas o algoritmo percorrerá por experimento
fixed_num_iterations = 1000

# Define o caminho do arquivo CSV no Google Drive
csv_file_path = '/content/drive/My Drive/dataset/spam.csv'

# Configuração do diretório de dados (evita problemas se estiver em um diretório diferente do script)
path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, '..', 'data', 'spam.csv')

df_spam = pd.read_csv(data_path, encoding='latin-1')

# Limpeza inicial
df_spam = prep.remove_clutter(df_spam)

# Renomeia as colunas para nomes mais intuitivos: v1 vira 'label' e v2 vira 'message'
df_spam = df_spam.rename(columns={'v1': 'label', 'v2': 'message'})

# Converte os rótulos de texto ('ham' e 'spam') para números (0 e 1) para o modelo processar
df_spam['label_numeric'] = df_spam['label'].map({'ham': 0, 'spam': 1})

# Define X como a entrada (mensagens) e y como o alvo (0 ou 1)
X = df_spam['message']
y = df_spam['label_numeric']

# Divide os dados em conjuntos de Treino (80%) e Teste (20%)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)

# prepro.text_processing VEM AQUI!

# Criando vetores de números para o texto nos conjuntos fatiados
X_train_processed, X_test_processed, y_train_processed, y_test_processed = prep.createVectorizer(X_train_orig, X_test_orig, y_train_orig, y_test_orig)


""" Função que permite rodar o modelo com algumas variações de parâmetros
    para testes mais variados, evitando repetição de código"""

def run_model_by_dataset(model, n_experiments, x_tr, y_tr):

  times = []
  train_accs = []
  test_accs = []
  model_results = None

  for i in range(n_experiments):
      print(f"          Experimento {i+1}/{n_experiments}")

      # Treinamento
      d = model(x_tr, y_tr, X_test_processed, y_test_processed,
                num_iterations=fixed_num_iterations, learning_rate=fixed_learning_rate, print_cost=False)

      # Salvar métricas
      times.append(d['total_optimization_time'])
      train_accs.append(d['train_acc'])
      test_accs.append(d['test_acc'])
      print("-" * 35)

      # Atualizar resultados a cada experimento para obter o último
      model_results = d

  return times, train_accs, test_accs, model_results['costs'][-1]



##########################################
########      EXPERIMENTOS     ###########
##########################################

##### IMPLEMENTAÇÃO VETORIZADA #####

# Parâmetros para teste vetorizado
num_experiments = 5
v_times = []
v_train_accuracies = []
v_test_accuracies = []

# Hiperparâmetros
fixed_learning_rate = 0.01

print(f"\nRodando {num_experiments} experimentos (VETORIZADA)...\n")

v_times, v_train_accuracies, v_test_accuracies, v_cost_final = run_model_by_dataset(vect.model, num_experiments, X_train_processed, y_train_processed)


#----------------------------------
##### IMPLEMENTAÇÃO ITERATIVA #####

nonv_times = []
nonv_train_accuracies = []
nonv_test_accuracies = []

# Hiperparâmetros
fixed_learning_rate = 0.01

print(f"\nRodando {num_experiments} experimentos (NÃO-VETORIZADA)...\n")

nonv_times, nonv_train_accuracies, nonv_test_accuracies, nonv_cost_final = run_model_by_dataset(iter.model, num_experiments, X_train_processed, y_train_processed)


##########################################
########       RESULTADOS      ###########
##########################################

##### RESUMO PRINCIPAL #####

# 1. Calculando as métricas para a implementação Não-Vetorizada
mean_nonv = np.mean(nonv_times)
std_nonv = np.std(nonv_times)

# 2. Calculando as métricas para a implementação Vetorizada
mean_v = np.mean(v_times)
std_v = np.std(v_times)

# Sepeedup Médio
speedup = mean_nonv / mean_v

# Cálculo do speedup de cada experimento
speedups_individuais = []
for i in range(len(v_times)):
  speedups_individuais.append(nonv_times[i] / v_times[i])


print("\n" +"="* 35)
print("      Resumo dos Experimentos      ")
print("=" * 35)

print("\nIMPLEMENTAÇÃO VETORIZADA:")
print("-" * 35)
print(f"\nTempo Médio de Otimização: {mean_v:.4f} segundos")
print(f"Tempo Desvio Padrao de Tempo por Implementação : {std_v:.4f} segundos")
print(f"Acurácia Média de Treino: {np.mean(v_train_accuracies):.2f} %")
print(f"Acurácia Média de Teste: {np.mean(v_test_accuracies):.2f} %")

print("\nIMPLEMENTAÇÃO NÃO-VETORIZADA:")
print("-" * 35)
print(f"\nTempo Médio de Otimização: {mean_nonv:.4f} segundos")
print(f"Tempo Desvio Padrao de Tempo por Implementação : {std_nonv:.4f} segundos")
print(f"Acurácia Média de Treino: {np.mean(nonv_train_accuracies):.2f} %")
print(f"Acurácia Média de Teste: {np.mean(nonv_test_accuracies):.2f} %")

print("\nDADOS SPEEDUP:")
print("-" * 35)

print(f"\nSpeedup: {speedup:.2f}x")
print(f"Desvio padrão dos Speedups: ± {np.std(speedups_individuais):.2f}x")

print("\nCOMPARAÇÃO DOS CUSTOS (J):")
print("-" * 35)

print(f"\nCusto (J) NV: {nonv_cost_final}")
print(f"Custo (J) V: {v_cost_final}")



##########################################
########  TABELA COMPARATIVA   ###########
##########################################

experiment_data = {
    "Implementação": ["Não-Vetorizada (Loops)", "Vetorizada (NumPy)"],
    "Tempo Médio (s)": [f"{mean_nonv:.4f}", f"{mean_v:.4f}"],
    "Desvio Padrão (±)": [f"{std_nonv:.4f}", f"{std_v:.4f}"],
    "Speedup": ["1.00x (Base)", f"{speedup:.2f}x"]
}

# 1. Criando a instância da tabela
tabela = PrettyTable()

# 2. Definindo os nomes das colunas
tabela.field_names = ["Implementação", "Tempo Médio (s)", "Desvios Padrão (±)", "Speedup"]

# 3. Adicionando as linhas (usando os seus dados da imagem)
tabela.add_row(["Não-Vetorizada (Loops)", f"{mean_nonv:.4f}", f"{std_nonv:.4f}", "1.00x (Base)"])
tabela.add_row(["Vetorizada (NumPy)", f"{mean_v:.4f}", f"{std_v:.4f}", f"{speedup:.2f}x"])

# 4. Configurações estéticas (opcional)
tabela.align["Implementação"] = "l"  # Alinha à esquerda
tabela.padding_width = 1            # Espaçamento interno

print("\n")
print("="*70)

print("\nTabela- Análise Comparativa de Desempenho")
print(tabela)




##########################################
######    GRÁFICOS DE DESEMPENHO   #######
##########################################

print("\nExibindo gráfico de performance oor experimento (VETORIZADA)...")


plt.figure(figsize=(10, 6))
plt.plot(range(1, num_experiments + 1), v_times, marker='o', color='red', label='Tempo de Otimização')
plt.title('Tempo de Otimização por Experimento (Vetorizada)', weight='bold')
plt.xlabel('Número do Experimento')
plt.ylabel('Tempo (segundos)')
plt.xticks(range(1, num_experiments + 1))
plt.grid(True)
plt.legend()
plt.show()


print("\nExibindo gráfico de performance oor experimento (ITERATIVA)...")


plt.figure(figsize=(10, 6))
plt.plot(range(1, num_experiments + 1), nonv_times, marker='o', color='red', label='Tempo de Otimização')
plt.title('Tempo de Otimização por Experimento (Não-Vetorizada)', weight='bold')
plt.xlabel('Número do Experimento')
plt.ylabel('Tempo (segundos)')
plt.xticks(range(1, num_experiments + 1))
plt.grid(True)
plt.legend()
plt.show()


##########################################
######      GRÁFICOS DE BARRAS     #######
##########################################

print("\nExibindo gráfico de barras comparativo...")

experimentos = ['Exp 1', 'Exp 2', 'Exp 3', 'Exp 4', 'Exp 5']
x = np.arange(len(experimentos))
width = 0.35

plt.bar(x - width/2, nonv_times, width, label='Não-Vetorizada', color='coral')
plt.bar(x + width/2, v_times, width, label='Vetorizada', color='skyblue')

plt.ylabel('Tempo de Execução (segundos)')
plt.title('Comparação de Performance: 5 Experimentos', weight='bold')
plt.xticks(x, experimentos)
plt.legend()

plt.annotate(f'Speedup Médio: {speedup:.1f}x',
             xy=(0.5, 0.2), xycoords='axes fraction',
             ha='center', fontsize=12, fontweight='bold', color='darkred',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=2))

# Adicionando texto do speedup individual em cima de cada par de barras
for i in range(len(v_times)):
    plt.text(x[i], nonv_times[i] + (max(nonv_times) * 0.01), 
             f'{speedups_individuais[i]:.1f}x', 
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred')

plt.tight_layout()
plt.savefig('grafico_comparativo_experimentos.png')

print("\n")

#################################################
#  MEDIR TEMPO EM FUNÇÃO DO TAMNAHO DA AMOSTRA  #
#################################################


tamanhos_m = range(500, 4457, 500)
speedup_outcomes = []
nonv_times = []
v_times = []
v_test_accuracies = []

num_experiments_spdup = 2

print(f"Medir tempo de Execução em função do tamanho das n amostras\n")

for m in tamanhos_m:
    # Fatiando os dados
    X_fatiado = X_train_processed[:, :m]
    Y_fatiado = y_train_processed[:, :m]
    
    print("=" * 35)
    print(f"       Tamanho: {m} amostras")
    print("=" * 35)
    print("           NÃO-VETORIZADA\n")
    
    # Medir tempo para rodar dois experimentos de uma vez
    t0 = time.time()
    
    nonv_times, _, _, _ = run_model_by_dataset(iter.model, num_experiments_spdup, X_fatiado, Y_fatiado)

    tempo_nonv = time.time() - t0

    print(f"Tempo Total: {tempo_nonv:.4f}s")
    print("-" * 35)
    print('\n')
    print("            VETORIZADA\n")


    # Fazer o mesmo para a vetorizada
    t1 = time.time()
    
    v_times, _, _, _ = run_model_by_dataset(vect.model, num_experiments_spdup, X_fatiado, Y_fatiado)

    tempo_v = time.time() - t1

    print(f"Tempo Total: {tempo_v:.4f}s")
    print("-" * 35)
    print('\n')
    speedup_outcomes.append(tempo_nonv/ tempo_v)
    


##########################################
###   GRÁFICO DO EXPERIMENTO ACIMA    ####
##########################################


# --- GERANDO O GRÁFICO DE LINHA (Item h) ---
plt.figure(figsize=(12, 6))
plt.plot(tamanhos_m, speedup_outcomes, marker='o', linestyle='-', color='darkgreen', linewidth=2)

# Estilização do gráfico
plt.title('Evolução do Speedup vs. Tamanho das Amostras (m)', fontsize=14, weight='bold')
plt.xlabel('Número de Amostras (m)', fontsize=12, weight='bold')
plt.ylabel('Speedup (x vezes mais rápido)', fontsize=12, weight='bold')
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Adicionando anotações de texto nos pontos para facilitar a leitura
for i, txt in enumerate(speedup_outcomes):
    plt.annotate(f"{txt:.1f}x", (tamanhos_m[i], speedup_outcomes[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
plt.figtext(0.5, 0.01, f'Épocas por experimento: {fixed_num_iterations}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('evolucao_speedup_amostras.png')
plt.show()



##########################################
#  INFORMAÇÕES DO AMBIENTE DE EXECUÇÃO   #
##########################################


import sys
import platform
import psutil

# Informações do Sistema Operacional
os_info = platform.system()
print(f"Sistema Operacional: {os_info}")

# Informações do Processador
processor_info = platform.processor()
print(f"Processador: {processor_info}")

# Informações da Memória RAM
ram = psutil.virtual_memory()
print(f"Memória RAM Total: {ram.total / (1024**3):.2f} GB")

# Versão do Python
python_version = sys.version
print(f"Versão do Python: {python_version}")

# Versão do NumPy
numpy_version = np.__version__
print(f"Versão do NumPy: {numpy_version}")
