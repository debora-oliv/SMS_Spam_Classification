import pandas as pd
import numpy as np
import os
import scipy
import preprocessing as prep
from sklearn.feature_extraction.text import TfidfVectorizer # pip install scikit-learn
from sklearn.model_selection import train_test_split

import logistic_regression_iterative as iter
import logistic_regression_vect as vect

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

print(X_train_processed)

""" Função que permite rodar o modelo com algumas variações de parâmetros
    para testes mais variados, evitando repetição de código"""

def run_model_by_dataset(model, n_experiments, x_tr, y_tr):

  times = []
  train_accs = []
  test_accs = []
  model_results = None
  final_cost = 0.0

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
fixed_num_iterations = 50
fixed_learning_rate = 0.01

print(f"Rodando {num_experiments} experimentos (VETORIZADA)...\n")

v_times, v_train_accuracies, v_test_accuracies, v_cost_final = run_model_by_dataset(vect.model, num_experiments, X_train_processed, y_train_processed)


#----------------------------------
##### IMPLEMENTAÇÃO ITERATIVA #####

nonv_times = []
nonv_train_accuracies = []
nonv_test_accuracies = []

# Hiperparâmetros
fixed_num_iterations = 50
fixed_learning_rate = 0.01

print(f"Rodando {num_experiments} experimentos (NÃO-VETORIZADA)...\n")

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