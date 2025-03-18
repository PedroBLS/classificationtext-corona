*Classificação de Sentimentos sobre o COVID-19*

Este repositório contém um projeto de classificação de sentimentos em tweets relacionados ao COVID-19. O modelo de Random Forest é treinado para classificar sentimentos com base em dados de texto.

📂 *Estrutura do Projeto*

Corona_NLP_train.csv: Conjunto de dados de treino.

Corona_NLP_test.csv: Conjunto de dados de teste.

classificacao_covid.py: Script principal que carrega, processa e treina o modelo.

🚀 *Tecnologias Utilizadas*

Python 3

Pandas (manipulação de dados)

Seaborn e Matplotlib (visualização de dados)

Scikit-learn (modelo de machine learning)

NLTK (processamento de linguagem natural)

📊 *Fluxo do Projeto*

Carregamento dos dados: O dataset é carregado a partir dos arquivos CSV.

Pré-processamento: Remoção de pontuação, stopwords e lematização das palavras.

Vetorização com TF-IDF: Conversão do texto para uma representação numérica.

Divisão dos dados: Separação em treino e validação.

Treinamento do modelo: Modelo Random Forest é treinado.

Avaliação do modelo: Exibição de métricas como precision, recall e F1-score.

Visualização: Gráfico da distribuição dos sentimentos.

📌 *Resultados*

Após a execução do script, serão gerados:

Relatórios de classificação no terminal.
