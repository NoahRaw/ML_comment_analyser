from flask import Flask, request, jsonify
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Charger le modèle et le vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Initialiser le prétraitement
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    comment = data.get('comment', '')
    if not comment:
        return jsonify({'error': 'No comment provided'}), 400
    clean_comment = preprocess(comment)
    vector = vectorizer.transform([clean_comment]).toarray()
    prediction = model.predict(vector)[0]
    sentiment = 1 if prediction == 1 else 0
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)


