# Sobre o Projeto 

O principal objetivo é analisar a complexidade computacional e o ganho de desempenho ao utilizar operações vetorizadas em comparação com uma abordagem iterativa tradicional (processar cada exemplo de treinamento individualmente utilizando
laços for explícitos) no treinamento do modelo de Regressão Logística para classificação binária de mensagens SMS.

# Base de Dados

Foi utilizado o [SMS Spam Collection Dataset](!https://archive.ics.uci.edu/dataset/228/sms+spam+collection), uma base de classificação binária que contém mensagens em inglês, categorizadas como:

- ```Ham```: Mensagens legítimas.

- ```Spam```: Mensagens indesejadas.

A coleção possui um total de 5.574 mensagens, onde 4.827 (86,6%) são mensagens legítimas e 747 (13,4%) são mensagens de spam.

# Tecnologias
O projeto foi desenvolvido em Python utilizando as seguintes bibliotecas:

```NLTK```: Pré-processamento de texto (limpeza de stop words).

```Scikit-Learn```: Vetorização TF-IDF e divisão de treino/teste.

```NumPy```: Para cálculos matemáticos e implementação vetorizada.

```Pandas```: Manipulação e carregamento do dataset.

```Matplotlib/Seaborn```: Geração de gráficos comparativos.

```
.
├── data/
│   └── SMSSpamCollection                       # Base de dados original
├── src/
│   ├── preprocessing.py                        # Limpeza e normalização (TF-IDF)
│   ├── logistic_regression_iterative.py        # Implementação regressão logística iterativa
│   ├── logistic_regression_iterative.py        # Implementação regressão logística vetorizada
│   └── run_experiments.py                      # Script principal de experimentação
├── .gitignore        
├── requirements.txt                
└── README.md
```