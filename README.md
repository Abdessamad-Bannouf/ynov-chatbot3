# 🧠 Chatbot de Classification d'Intentions (NLP - FR)

Ce projet est un **prototype de chatbot** capable de détecter l'intention d’un utilisateur en français. Il repose sur du prétraitement linguistique et un modèle de classification supervisée entraîné sur un petit jeu de données d’intentions typiques.

---

## 📁 Structure du projet

```
Chatbox/
├── main.py                  # Script principal : traitement et affichage des prédictions
├── nlp_processor.py         # Prétraitement NLP (nltk + spaCy)
├── intent_classifier.py     # Entraînement et prédiction des intentions
└── README.md                # Documentation du projet
```

---

## ⚙️ Technologies utilisées

- 🐍 Python 3
- ✳️ NLTK (tokenisation, stopwords, stemming)
- 🇫🇷 spaCy (analyse linguistique française)
- 🤖 scikit-learn (TF-IDF, Logistic Regression, évaluation)
- 📊 Matplotlib & Seaborn (matrice de confusion)

---

## 🧪 Fonctionnalités

- Prétraitement du texte :
  - Nettoyage, tokenisation, suppression des mots vides, racinisation
- Analyse linguistique :
  - Part-of-speech, entités nommées, dépendances
- Classification d’intentions :
  - `commande`, `menu`, `horaires`, `prix`, `salutation`, `au_revoir`
- Précision atteinte : **86 %** ✅

---

## 📊 Résultat du classificateur

```
Accuracy: 0.86
Macro avg F1-score: 0.84
Weighted avg F1-score: 0.84
```

✅ Toutes les classes ont au moins un exemple correctement prédit.

---

## 🚀 Comment exécuter

```bash
pip install -r requirements.txt
python main.py
```

> Le script télécharge automatiquement les ressources nécessaires (`nltk`, `spaCy` modèle français).

---

## 💬 Exemple de message traité

```
Message original : Je veux commander une pizza
→ Intention détectée : commande (confiance : 98.5%)
```

---

## 🛠️ À améliorer

- Utiliser un modèle **CamemBERT fine-tuné** pour des performances proches du SOTA.
- Ajouter plus d'exemples pour chaque intention.
- Intégrer une API Flask/FastAPI ou une interface web.

---

## 👨‍💻 Auteur

Projet réalisé dans le cadre d'un TP de traitement du langage naturel.   
Étudiant : Abdessamad Bannouf