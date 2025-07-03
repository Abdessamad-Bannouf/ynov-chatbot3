from nlp_processor import (
    preprocess_pipeline,
    analyze_pos,
    extract_entities,
    analyze_dependencies
)

from intent_classifier import classify_intent_with_preprocessing

def process_message(message):
    print("\n🟡 Message original :", message)

    # Étape 1 : Prétraitement
    processed = preprocess_pipeline(message)
    print("🔵 Texte prétraité :", processed)

    # Étape 2 : Analyse linguistique
    print("🟢 Analyse morpho-syntaxique (POS) :", analyze_pos(message))
    print("🟢 Entités nommées :", extract_entities(message))
    print("🟢 Dépendances syntaxiques :", analyze_dependencies(message))

    # Étape 3 : Classification d'intention
    intent, confidence = classify_intent_with_preprocessing(message)
    print(f"🔴 Intention détectée : {intent} (confiance : {max(confidence)*100:.2f}%)")

if __name__ == "__main__":
    test_messages = [
        "Bonjour ! J'aimerais commander une pizza margherita s'il vous plaît.",
        "Pouvez-vous me dire les horaires d'ouverture de votre restaurant ?",
        "Je veux commander trois pizzas margherita",
        "À quelle heure fermez-vous le dimanche ?",
        "Merci beaucoup et au revoir !"
    ]

    for msg in test_messages:
        process_message(msg)
