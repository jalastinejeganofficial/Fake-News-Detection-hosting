import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# Add labels
df_fake["label"] = 0
df_real["label"] = 1

# Combine and shuffle
df = pd.concat([df_fake, df_real])
df = df.sample(frac=1).reset_index(drop=True)

# Use only the text column
X = df["text"]
y = df["label"]

# Text vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Sample input
sample_text = """Breaking: Government announces free electricity for all citizens starting next week."""

# Vectorize and predict
sample_vec = vectorizer.transform([sample_text])
prediction = model.predict(sample_vec)

print("REAL news ✅" if prediction[0] == 1 else "FAKE news ❌")

# Explanation function
def explain_news(text, label, api_key):
    prompt = f"""
    The following news headline was classified as {'REAL' if label == 1 else 'FAKE'}:
    "{text}"

    Please explain why this classification is appropriate. Use simple language suitable for Indian users. Highlight any signs of misinformation, exaggeration, or lack of credibility.
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",  # You can switch to Claude, Gemini, etc.
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# Call explanation
api_key = "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"  # Replace with secure method in production
explanation = explain_news(sample_text, prediction[0], api_key)
print("Explanation:", explanation)

import pickle

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)