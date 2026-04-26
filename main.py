import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import text_processing

# Cloquei algumas coisas entre colchetes [] indicando que são nomes hipotéticos, vcs podem alterar conforme preferirem na hora de implementar
# Maass, se seguirem esses nomes, já facilita porque no fim basta descomentar tudo sem precisar alterar nome de função/variável :)

# from src.logistic_regression_iterative import [regressao_logistica_iterativa]
# from src.logistic_regression_vectorized import [regressao_logistica_vetorizada]
# from src.run_experiments import *

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
       
    4. Treino e Otimização: Inicializa as classes de Regressão Logística
       e regista os tempos de convergência.
       
    5. Avaliação de Resultados: Processa as métricas 
       de custo, calcula o ganho de desempenho (Speedup) e plota os gráficos comparativos.

    Returns:
        None. A função orquestra o processo e imprime/exporta os resultados diretamente.
    """
    # --- ETAPA 1 ---
    df = pd.read_csv('data/SMSSpamCollection', sep='\t', names=["label", "message"])

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    df['cleaned_message'] = df['message'].apply(text_processing)

    # --- ETAPA 2 ---
    X_text_train, X_text_test, Y_train, Y_test = train_test_split(
        df['cleaned_message'], 
        df['label'].values, 
        test_size=0.20, 
        random_state=42
    )

    # --- ETAPA 3 ---
    vectorizer = TfidfVectorizer()
    
    X_train_matrix = vectorizer.fit_transform(X_text_train)
    
    X_test_matrix = vectorizer.transform(X_text_test)

    # --- ETAPA 4 ---
    
    # modelo_iterativo = [regressao_logistica_iterativa](learning_rate=0.01, epochs=1000)
    # tempo_iterativo = modelo_iterativo.fit_and_time(X_train_matrix, Y_train)
    
    # modelo_vetorizado = [regressao_logistica_vetorizada](learning_rate=0.01, epochs=1000)
    # tempo_vetorizado = modelo_vetorizado.fit_and_time(X_train_matrix, Y_train)

    # --- ETAPA 5 ---


if __name__ == "__main__":
    main()