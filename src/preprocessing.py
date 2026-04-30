import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer 

nltk.download('stopwords', quiet=True)

stop_words_english = set(stopwords.words('english')) # Converter para set faz a complexidade de busca cair de O(N) para O(1)

def text_processing(mess):
    """
    Recebe uma mensagem em formato de string e então realiza:
    1. Normalização Lexical (padronizar o texto em um formato canônico por meio do lowercasing, reduzindo a dimensionalidade e a esparsidade dos dados)
    2. Remoção de Ruído (eliminação de caracteres que não agregam valor semântico para a tarefa de classificação)
    3. Tokenização (divisão da frase em tokens, nesse caso palavras individuais)
    4. Filtragem / Remoção de Stop Words (remoção de palavras funcionais, muito frequentes ou que carregam pouca carga informacional)

    Returns:
        String única para vetorização
    """
    mess = mess.lower()

    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    cleaned_words = [word for word in nopunc.split() if word not in stop_words_english]
    
    return ' '.join(cleaned_words)

def remove_clutter(df):
    
    # Remove colunas extras vazias que costumam aparecer nesse dataset específico
    df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')

    # Renomeia as colunas para nomes mais intuitivos: v1 vira 'label' e v2 vira 'message'
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})

    return df

def createVectorizer(X_train_orig, X_test_orig, y_train_orig, y_test_orig):
    # Cria o vetorizador TF-IDF para transformar texto em números baseados na importância das palavras
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    # 'Aprende' o vocabulário do treino e transforma as mensagens em matrizes numéricas
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_orig)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_orig)

    # Transpõe as matrizes (T) para ficarem no formato (features, amostras) exigido pela nossa implementação manual
    
    X_train_processed = X_train_tfidf.T.toarray()
    X_test_processed = X_test_tfidf.T.toarray()

    # Ajusta o formato das etiquetas (y) para serem vetores linha (1, amostras)
    y_train_processed = y_train_orig.values.reshape(1, -1)
    y_test_processed = y_test_orig.values.reshape(1, -1)
    
    return X_train_processed, X_test_processed, y_train_processed, y_test_processed