# Sobre o Projeto 
O principal objetivo é analisar a complexidade computacional e o ganho de desempenho ao utilizar operações vetorizadas em comparação com uma abordagem iterativa tradicional (processar cada exemplo de treinamento individualmente utilizando
laços for explícitos) no treinamento do modelo de Regressão Logística para classificação binária de mensagens SMS.

## Base de Dados
Foi utilizado o [SMS Spam Collection Dataset](!https://archive.ics.uci.edu/dataset/228/sms+spam+collection), uma base de classificação binária que contém mensagens em inglês, categorizadas como:

- ```Ham```: Mensagens legítimas.

- ```Spam```: Mensagens indesejadas.

A coleção possui um total de 5.574 mensagens, onde 4.827 (86,6%) são mensagens legítimas e 747 (13,4%) são mensagens de spam.

## Tecnologias
O projeto foi desenvolvido em Python utilizando as seguintes bibliotecas:

```NLTK```: Pré-processamento de texto (limpeza de stop words).

```Scikit-Learn```: Vetorização TF-IDF e divisão de treino/teste.

```NumPy```: Para cálculos matemáticos e implementação vetorizada.

```Pandas```: Manipulação e carregamento do dataset.

```Matplotlib/Seaborn```: Geração de gráficos comparativos.

## Estrutura
```
.
├── data/
│   └── SMSSpamCollection                       # Base de dados original
├── src/
│   ├── preprocessing.py                        # Limpeza e normalização (TF-IDF)
│   ├── logistic_regression_iterative.py        # Implementação regressão logística iterativa
│   ├── logistic_regression_vectorized.py       # Implementação regressão logística vetorizada
│   └── run_experiments.py                      # Script principal de experimentação
├── main.py                                     # Centraliza o fluxo de execução
├── .gitignore        
├── requirements.txt                
└── README.md
```

# Teste e Execução
### Pré-Requisitos
- Python 3.x
- Git

## Execução Local

**1. Clone o repositório:**
   ```bash
   git clone https://github.com/debora-oliv/SMS_Spam_Classification.git
   ```

**2. Faça o download do [dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) e adicone-o em ./data**

**3. Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

**4. Execute o orquestrador:**
   ```bash
   python main.py
   ```
   
## Execução via Google Colab
**1. Abra o Google Colab.**

**2. Crie um novo Notebook em branco.**

**3. Em uma célula de código, baixe o repositório clonando-o diretamente do GitHub:**
   ```bash
   !git clone https://github.com/debora-oliv/SMS_Spam_Classification.git
   ```
**4. Mude o diretório de execução do Colab para dentro da pasta clonada:**
   ```bash
   import os
   os.chdir('nome-do-repositorio')
   ```

**5. Faça o download do [dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) e adicone-o em ./data**

**6. Instale as dependências do projeto:**
   ```bash
   !pip install -r requirements.txt
   ```

**7. Execute o script principal:**
   ```bash
   !python main.py
   ```

  *Os gráficos gerados aparecerão na aba de "Arquivos" na barra lateral esquerda do Colab, podendo ser visualizados e baixados.*



