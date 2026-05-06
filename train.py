import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(ngram_range=(1,2))
model = LogisticRegression()

with open('intents.json') as file:
    data = json.load(file)

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

patterns = []
tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        clean_text = preprocess(pattern)
        patterns.append(clean_text)
        tags.append(intent['tag'])

X = vectorizer.fit_transform(patterns)

model.fit(X, tags)


"""test = ["resume kaise bnau"]
test_clean = [preprocess(test[0])]
test_vec = vectorizer.transform(test_clean)
prediction = model.predict(test_vec)
print(prediction)"""

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Vectorizer saved")


#print(X.shape)
#print(patterns[:5])
#print(tags[:5])
#print(preprocess("How to make a resume?"))
#print(data['intents'][0])