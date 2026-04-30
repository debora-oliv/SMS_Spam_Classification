import pandas as pd
import numpy as np
import os
import scipy
import preprocessing as prep
from sklearn.feature_extraction.text import TfidfVectorizer # pip install scikit-learn
from sklearn.model_selection import train_test_split

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