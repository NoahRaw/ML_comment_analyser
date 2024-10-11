# ----------------------------------------------------------------------- Etape 1 ------------------------------------------------------------
import pandas as pd

# Charger les données
df = pd.read_csv('comments.csv')

# Afficher les premières lignes
print(df.head())

# ----------------------------------------------------------------------- Etape 2 ------------------------------------------------------------
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Télécharger les ressources nécessaires
nltk.download('stopwords')
nltk.download('wordnet')

# Initialiser le lemmatizer et les stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Mettre en minuscules
    text = text.lower()
    # Supprimer les URL, les mentions, etc.
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    # Supprimer les caractères spéciaux et les chiffres
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenization
    tokens = text.split()
    # Supprimer les stop words et lemmatiser
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Appliquer le prétraitement
df['clean_comment'] = df['comment'].apply(preprocess)

print(df[['comment', 'clean_comment']].head())

# ----------------------------------------------------------------------- Etape 4 ------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialiser le vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Appliquer la vectorisation
X = vectorizer.fit_transform(df['clean_comment']).toarray()
y = df['sentiment'].values

# ----------------------------------------------------------------------- Etape 5 ------------------------------------------------------------
from sklearn.model_selection import train_test_split

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------- Etape 6 ------------------------------------------------------------
from sklearn.linear_model import LogisticRegression

# Initialiser le modèle
model = LogisticRegression()

# Entraîner le modèle
model.fit(X_train, y_train)

# ----------------------------------------------------------------------- Etape 7 ------------------------------------------------------------
from sklearn.metrics import accuracy_score, classification_report

# Prédictions
y_pred = model.predict(X_test)

# Évaluer la performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------------------- Etape 8 ------------------------------------------------------------
import joblib

# Sauvegarder le modèle
joblib.dump(model, 'sentiment_model.joblib')

# Sauvegarder le vectorizer
joblib.dump(vectorizer, 'vectorizer.joblib')
