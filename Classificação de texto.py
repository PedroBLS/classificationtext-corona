import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer

# Carregar os dados de treino e teste
train_data = pd.read_csv(r"D:\Program Files (x86)\Projetos VSCode\Portifólio\Corona virus classification\archive\Corona_NLP_train.csv", encoding='latin1')
test_data = pd.read_csv(r"D:\Program Files (x86)\Projetos VSCode\Portifólio\Corona virus classification\archive\Corona_NLP_test.csv", encoding='latin1')

# Pré-processamento dos dados
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

train_data['OriginalTweet'] = train_data['OriginalTweet'].apply(preprocess_text)
test_data['OriginalTweet'] = test_data['OriginalTweet'].apply(preprocess_text)

# Vetorização do texto usando TF-IDF
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data['OriginalTweet'])
test_vectors = vectorizer.transform(test_data['OriginalTweet'])

# Dividir os dados de treino em treino e validação
X_train, X_val, y_train, y_val = train_test_split(train_vectors, train_data['Sentiment'], test_size=0.2, random_state=42)

# Treinar o modelo Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Avaliar o modelo no conjunto de validação
y_pred = model.predict(X_val)
print("Relatório de Classificação (Validação):")
print(classification_report(y_val, y_pred))

# Avaliar o modelo no conjunto de teste
y_pred_test = model.predict(test_vectors)
print("\nRelatório de Classificação (Teste):")
print(classification_report(test_data['Sentiment'], y_pred_test))

# Visualizações (opcional)
sns.countplot(x='Sentiment', data=train_data)
plt.title('Distribuição de Sentimentos nos Dados de Treino')
plt.savefig("distribuicao_sentimentos_treino.png")
plt.show()