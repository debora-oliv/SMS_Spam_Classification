import string
import nltk
from nltk.corpus import stopwords

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