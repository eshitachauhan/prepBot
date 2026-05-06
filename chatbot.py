import json
import pickle
import nltk
import random
import sklearn
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from spellchecker import SpellChecker

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
spell = SpellChecker()

custom_words = ['dsa', 'cv', 'api', 'ml', 'hr']
spell.word_frequency.load_words(custom_words)

with open('intents.json') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return " ".join(tokens)

def correct_spelling(text):
    words = text.split()
    corrected_sentence = []
    
    for i, word in enumerate(words):
        
        if word.lower() in custom_words:
            corrected_sentence.append(word)
            continue
        
        candidates = spell.candidates(word)
        if not candidates:
            corrected_sentence.append(word)
            continue

        best_word = word
        best_score = 0
        for candidate in candidates:
            temp_sentence = words.copy()
            temp_sentence[i] = candidate
            original_embed = embed_model.encode([text])
            new_embed = embed_model.encode([" ".join(temp_sentence)])
            score = cosine_similarity(original_embed, new_embed)[0][0]
            if score > best_score:
                best_score = score
                best_word = candidate
        
        corrected_sentence.append(best_word)
    return " ".join(corrected_sentence)

all_patterns = []
all_tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        all_patterns.append(preprocess(pattern))
        all_tags.append(intent['tag'])

pattern_vectors = vectorizer.transform(all_patterns)
pattern_embeddings = embed_model.encode(all_patterns)

def get_response(user_input):
    corrected_input = correct_spelling(user_input)
    print(f"[DEBUG] original: {user_input}")
    print(f"[DEBUG] corrected: {corrected_input}")
    
    clean = preprocess(corrected_input)
    input_vec = vectorizer.transform([clean])

    
    probs = model.predict_proba(input_vec)
    confidence = np.max(probs[0])
    prediction = model.predict(input_vec)[0]

    print(f"[DEBUG] Intent: {prediction}, Confidence: {confidence:.2f}")

   
    if confidence >= 0.6:
        chosen_tag = prediction

    else:
        print("[DEBUG] Using fallback (embeddings)")

        
        user_embedding = embed_model.encode([user_input])
        similarity = cosine_similarity(user_embedding, pattern_embeddings)

        best_index = np.argmax(similarity)
        best_score = np.max(similarity)

        print(f"[DEBUG] Similarity Score: {best_score:.2f}")

        if best_score >= 0.4:
            chosen_tag = all_tags[best_index]
        else:
            return "I didn’t quite get that. Try asking about resumes, DSA, or interviews."

    
    for intent in data['intents']:
        if intent['tag'] == chosen_tag:
            return random.choice(intent['responses'])

if __name__ == "__main__":
    while True:
        user = input("You: ")
        if user.lower() == "exit":
            print("Bot: Bye!")
            break
        response = get_response(user)
        print("Bot: ", response)

