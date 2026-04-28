import pandas as pd
import numpy as np
import os
import scipy
from preprocessing import text_processing
from sklearn.feature_extraction.text import TfidfVectorizer # pip install scikit-learn
from sklearn.model_selection import train_test_split

# Configuração do diretório de dados (evita problemas se estiver em um diretório diferente do script)
path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, '..', 'data', 'spam.csv')

df = pd.read_csv(data_path, encoding='latin-1')

# Manter apenas as colunas relevantes
df = df[['v1', 'v2']]

df.columns = ['label', 'content']

# Convertendo labels para formato binário
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

df['content_clean'] = df['content'].apply(text_processing)


# Visualiza as primeiras 10 linhas para conferir as colunas

# print('='*15+' DADOS MAPEADOS '+'='*15)
# print(df.head(10)['content'])
# print(df.head(10)['content_clean'])


print('\nTAMANHO DATAFRAME: ' +  str(df['content_clean'].size))

# DADOS COM VOCABULÁRIO INCLUINDO SEQUÊNCIAS NUMÉRICAS ALEATÓRIAS

tfidf = TfidfVectorizer()

x_noisy = tfidf.fit_transform(df['content_clean'])
y = df['label'].values

noisy_vocab = tfidf.get_feature_names_out()

# DADOS LIMPOS (SEM SEQUÊNCIAS NUMÉRICAS)

tfidf_clean = TfidfVectorizer(token_pattern=r'\b[a-zA-Z]{3,}\b')

x_clean = tfidf_clean.fit_transform(df['content_clean'])

clean_vocab = tfidf_clean.get_feature_names_out()


print('\nTAMANHO VOCABULÁRIO (com ruído): ' +  str(noisy_vocab.size))

print('\nTAMANHO VOCABULÁRIO (sem ruído): ' +  str(clean_vocab.size) + '\n')

noisy_entries = tfidf.inverse_transform(x_noisy)
clean_entries = tfidf.inverse_transform(x_clean)

# Separar dados de treinamento e teste (com ruído)
X_train_nsy, X_test_nsy, y_train_nsy, y_test_nsy = train_test_split(x_noisy, y, test_size=0.2, random_state=42)

print(f'PRÉVIA DO VOCABULÁRIO (noisy): {noisy_vocab}')