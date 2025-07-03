import nltk
import re
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Chargement du modèle spaCy français
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    import subprocess
    print("🔄 Modèle spaCy 'fr_core_news_sm' non trouvé. Téléchargement en cours...")
    subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_sm"], check=True)
    nlp = spacy.load("fr_core_news_sm")

# ------------------------------
# Preprocessing complet
# ------------------------------

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_message(text):
    return word_tokenize(text, language='french')

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('french'))
    return [t for t in tokens if t not in stop_words]

def lemmatize_tokens(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

def preprocess_pipeline(text):
    text = normalize_text(text)
    tokens = tokenize_message(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)

# Analyse linguistique

def analyze_pos(text):
    doc = nlp(text)
    return [(t.text, t.pos_, t.tag_) for t in doc]

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def analyze_dependencies(text):
    doc = nlp(text)
    return [(t.text, t.dep_, t.head.text) for t in doc]