from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from nlp_processor import preprocess_pipeline

# ------------------------------
# 1. Classificateur d'intentions
# ------------------------------

class IntentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000, sublinear_tf=True)),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])

    def train(self, texts, labels):
        self.pipeline.fit(texts, labels)

    def predict(self, text):
        return self.pipeline.predict([text])[0]

    def predict_proba(self, text):
        return self.pipeline.predict_proba([text])[0]

# ------------------------------
# 2. Dataset
# ------------------------------

dataset = {
    "salutation": [
    "bonjour", "salut", "hello", "bonsoir", "coucou", "salutations",
    "yo", "bien le bonjour", "bonjour à vous", "salut les amis",
    "hey", "wesh", "bienvenue", "bonne journée", "je vous salue",
    "bonjour monsieur", "bonjour madame", "salut tout le monde", "bonjour à tous"
    ],
    "menu": [
        "je voudrais voir le menu", "pouvez-vous me montrer la carte",
        "quel est le menu aujourd'hui", "quelles sont vos spécialités",
        "affiche-moi le menu", "montrez-moi les plats",
        "qu'avez-vous à proposer", "je veux voir ce qu'on peut manger",
        "que servez-vous", "vous avez quoi à manger ?"
    ],
    "commande": [
        "je commande une pizza margherita", "je veux une pizza",
        "j'aimerais commander une boisson", "je passe une commande",
        "je souhaite commander à emporter", "je commande pour ce soir",
        "je prends deux pizzas", "je veux une commande rapide",
        "peux-tu me prendre une pizza royale", "j'aimerais commander trois pizzas"
    ],
    "horaires": [
        "quels sont vos horaires", "à quelle heure fermez-vous",
        "quand êtes-vous ouvert", "vos horaires d’ouverture svp",
        "vous ouvrez à quelle heure", "vous êtes ouverts aujourd'hui ?",
        "est-ce que vous ouvrez le dimanche", "fermez-vous à 22h ?",
        "pouvez-vous me dire vos heures d'ouverture", "c’est ouvert maintenant ?"
    ],
    "prix": [
        "combien coûte une pizza pepperoni", "quel est le tarif",
        "je veux connaître le prix du menu", "les prix svp",
        "c’est combien une pizza", "pouvez-vous me donner les tarifs",
        "quel est le prix moyen", "ça coûte combien",
        "je veux connaître les prix", "le prix d’une boisson svp"
    ],
    "au_revoir": [
        "merci beaucoup", "au revoir", "à bientôt",
        "bonne journée", "ciao", "bye",
        "merci pour votre aide", "bonne soirée",
        "merci encore", "à la prochaine"
    ]
}

texts = []
labels = []

for intent, phrases in dataset.items():
    for phrase in phrases:
        texts.append(preprocess_pipeline(phrase))
        labels.append(intent)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ------------------------------
# 3. Entraînement et évaluation
# ------------------------------

classifier = IntentClassifier()
classifier.train(X_train, y_train)

# Validation croisée
scores = cross_val_score(classifier.pipeline, texts, labels, cv=5, scoring='accuracy')
print("✅ Précision moyenne en validation croisée :", round(scores.mean() * 100, 2), "%")

# Évaluation
y_pred = [classifier.predict(x) for x in X_test]
print("ÉVALUATION DU CLASSIFICATEUR :\n")
print(classification_report(y_test, y_pred))

# Matrice de confusion
conf_mat = confusion_matrix(y_test, y_pred, labels=classifier.pipeline.classes_)
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=classifier.pipeline.classes_,
            yticklabels=classifier.pipeline.classes_)
plt.title("Matrice de confusion")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.show()

# ------------------------------
# 4. Fonction d'inférence
# ------------------------------

def classify_intent_with_preprocessing(message):
    processed_message = preprocess_pipeline(message)
    intent = classifier.predict(processed_message)
    confidence = classifier.predict_proba(processed_message)
    return intent, confidence

# ------------------------------
# 5. Tests obligatoires
# ------------------------------

if __name__ == "__main__":
    test_inputs = [
        "Bonsoir, je souhaiterais voir votre menu",
        "À quelle heure fermez-vous le dimanche ?",
        "Combien coûte une pizza Regina ?",
        "Je veux commander trois pizzas margherita"
    ]

    for msg in test_inputs:
        intent, confidence = classify_intent_with_preprocessing(msg)
        print(f"\nMessage : {msg}")
        print(f"Intent : {intent} | Confiance : {max(confidence):.2f}")
