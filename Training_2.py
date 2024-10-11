import pandas as pd
import re
import spacy
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Assurer la reproductibilité de la détection de langue
DetectorFactory.seed = 42

# Charger les modèles spaCy pour l'anglais et le français
nlp_en = spacy.load('en_core_web_sm')
nlp_fr = spacy.load('fr_core_news_sm')

def preprocess_multilingual(text):
    try:
        # Détection de la langue
        lang = detect(text)
    except:
        # Si la détection échoue, considérer comme anglais par défaut
        lang = 'en'
    
    # Sélectionner le modèle spaCy approprié
    if lang.startswith('fr'):
        nlp = nlp_fr
        stop_words = spacy.lang.fr.stop_words.STOP_WORDS
    else:
        # Par défaut, utiliser l'anglais
        nlp = nlp_en
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
    
    # Mettre en minuscules
    text = text.lower()
    
    # Supprimer les URL, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Supprimer les caractères spéciaux et les chiffres tout en conservant les lettres avec accents
    text = re.sub(r'[^a-zA-ZÀ-ÿ]', ' ', text)
    
    # Traiter le texte avec spaCy
    doc = nlp(text)
    
    # Lemmatisation et suppression des stop words
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stop_words]
    
    return ' '.join(tokens)

# Charger les données
df = pd.read_csv('comments.csv')

# Appliquer le prétraitement multilingue
df['clean_comment'] = df['comment'].apply(preprocess_multilingual)

# Afficher les premières lignes pour vérifier
print(df[['comment', 'clean_comment']].head())

# Initialiser le vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

# Appliquer la vectorisation
X = vectorizer.fit_transform(df['clean_comment']).toarray()
y = df['sentiment'].values

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle de régression logistique avec équilibrage des classes
model = LogisticRegression(class_weight='balanced', solver='liblinear')

# Entraîner le modèle
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calcul de la précision
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

# Rapport de classification détaillé
print(classification_report(y_test, y_pred))

# Sauvegarder le modèle
joblib.dump(model, 'sentiment_model.joblib')

# Sauvegarder le vectorizer
joblib.dump(vectorizer, 'vectorizer.joblib')
