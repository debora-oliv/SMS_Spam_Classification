import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import text_processing
from src.logistic_regression_vectorized import model as v_model
from src.logistic_regression_iterative import model as nonv_model
from src.run_experiments import print_env_info, run_all_experiments, run_speedup_vs_samples

def main():
    """
    Coordena a execução sequencial da avaliação de desempenho da Regressão Logística.
    O fluxo garante a correta passagem de parâmetros entre os módulos
    e previne o vazamento de dados (data leakage) durante a vetorização.

    O fluxo de execução está dividido em cinco etapas principais:
    
    1. Carregamento e Pré-processamento: Lê o ficheiro de dados original, 
       converte as etiquetas de texto para formato binário ('ham' = 0, 'spam' = 1) 
       e aplica o pipeline de normalização lexical.
    
    2. Divisão dos Dados: Separa as mensagens e as etiquetas em 
       conjuntos de treino (80%) e teste (20%) utilizando uma semente aleatória fixa 
       para garantir a total reprodutibilidade do experimento.
       
    3. Vetorização Matemática (TF-IDF): Converte as strings limpas numa matriz esparsa 
       de características. O modelo TF-IDF é ajustado (fit) exclusivamente nos dados 
       de treino, transformando posteriormente os dados de treino e de teste.
       
    4. Adaptação Dimensional: Transpõe as matrizes geradas para o formato 
       algébrico esperado pelas implementações do modelo (Atributos x Amostras).
       
    5. Treinamento e Avaliação: Executa a bateria de testes comparativos entre
       os algoritmos iterativo e vetorizado, extraindo tempos de execução, custo e speedup.

    Returns:
        None. A função orquestra o processo e imprime/exporta os resultados diretamente.
    """
    print_env_info()

    # --- ETAPA 1: Carregamento e Tratamento ---
    try:
        df = pd.read_csv('data/SMSSpamCollection', sep='\t', names=["label", "message"])
    except FileNotFoundError:
        print("\nERRO CRÍTICO: Arquivo 'SMSSpamCollection' não encontrado na pasta 'data/'.")
        print("Por favor, baixe o dataset e insira-o no diretório correspondente antes de executar o script.")
        sys.exit(1)

    # Tratamento de valores nulos e Label Encoding
    df = df.dropna()
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Pré-processamento
    df['cleaned_message'] = df['message'].apply(text_processing)

    # --- ETAPA 2: Divisão dos Dados (Train/Test Split) ---
    X_text_train, X_text_test, Y_train_orig, Y_test_orig = train_test_split(
        df['cleaned_message'], 
        df['label'].values, 
        test_size=0.20, 
        random_state=42
    )

    # --- ETAPA 3: Vetorização Matemática (Zero Data Leakage) ---
    vectorizer = TfidfVectorizer(max_features=5000)
    
    X_train_tfidf = vectorizer.fit_transform(X_text_train)
    X_test_tfidf = vectorizer.transform(X_text_test)

    # --- ETAPA 4: Adaptação Dimensional (Transposição) ---
    # O modelo matemático exige (atributos, amostras), mas o Scikit-Learn retorna (amostras, atributos)
    X_train_processed = X_train_tfidf.T.toarray()
    X_test_processed = X_test_tfidf.T.toarray()
    
    # Adaptando o formato do vetor alvo Y para (1, amostras)
    y_train_processed = Y_train_orig.reshape(1, -1)
    y_test_processed = Y_test_orig.reshape(1, -1)

    # --- ETAPA 5: Bateria de Experimentos Comparativos ---
    print("5. Iniciando a bateria de experimentos...")
    
    run_all_experiments(
        v_model, nonv_model, 
        X_train_processed, y_train_processed, 
        X_test_processed, y_test_processed
    )


    run_speedup_vs_samples(
        v_model, nonv_model, 
        X_train_processed, y_train_processed, 
        X_test_processed, y_test_processed
    )

if __name__ == "__main__":
    main()
