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


