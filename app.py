from flask import Flask, render_template, request
import pickle
import requests
import re
from deep_translator import GoogleTranslator
from twilio.twiml.messaging_response import MessagingResponse


app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# OpenRouter API key
API_KEY = "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"

# Keyword lists
SUSPICIOUS_WORDS = ["miracle", "shocking", "exclusive", "urgent", "cure", "secret", "breaking"]
EMOTIONAL_WORDS = ["panic", "fear", "disaster", "outrage", "terrifying", "amazing", "horrible"]

# Temporary in-memory store for community feedback
community_feedback = []

def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def translate_to_tamil(text):
    return GoogleTranslator(source='en', target='ta').translate(text)

def explain_news(text, label):
    prompt = f"""
    The following news headline was classified as {'REAL' if label == 1 else 'FAKE'}:
    "{text}"

    Please explain why this classification is appropriate. Use simple language suitable for Indian users. Highlight any signs of misinformation, exaggeration, or lack of credibility.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

def highlight_words(text, keywords):
    return [word for word in keywords if word.lower() in text.lower()]

def check_source(text):
    return "No source found" if not re.search(r"http|source:", text.lower()) else "Source present"

def get_risk_level(pred, suspicious, emotional, source_status):
    if pred == 0 and (suspicious or emotional or source_status == "No source found"):
        return "High Risk", "danger"
    elif pred == 0 or suspicious or emotional:
        return "Medium Risk", "warning"
    else:
        return "Low Risk", "safe"

def detect_manipulation_techniques(text):
    techniques = []
    if highlight_words(text, EMOTIONAL_WORDS):
        techniques.append("Emotional Manipulation")
    if highlight_words(text, SUSPICIOUS_WORDS):
        techniques.append("Sensational Language")
    if re.search(r"(WHO|NASA|UNESCO|Supreme Court|CBI|AIIMS)", text, re.IGNORECASE) and "source" not in text.lower():
        techniques.append("False Attribution")
    if re.search(r"only|always|never|everyone|nobody", text, re.IGNORECASE):
        techniques.append("Cherry-Picking")
    if re.search(r"(celebrity|expert|guru|leader)", text, re.IGNORECASE):
        techniques.append("Appeal to Authority")
    return techniques if techniques else ["None Detected"]

def detect_cognitive_biases(text):
    biases = []
    if re.search(r"\b(always|never|everyone|nobody|obviously|clearly)\b", text, re.IGNORECASE):
        biases.append(("Confirmation Bias", "This matches what I already believe."))
    if re.search(r"\b(expert|scientist|guru|celebrity|famous|renowned)\b", text, re.IGNORECASE):
        biases.append(("Authority Bias", "It came from a famous person."))
    if re.search(r"\b(viral|everyone is sharing|trending|must share|don’t miss)\b", text, re.IGNORECASE):
        biases.append(("Bandwagon Effect", "Everyone’s sharing it."))
    return biases if biases else [("None Detected", "No strong bias indicators found.")]

@app.route("/", methods=["GET", "POST"])
def index():
    result_en = explanation_en = result_ta = explanation_ta = ""
    suspicious = emotional = []
    source_status = ""
    original_text = ""
    risk_label = ""
    risk_color = ""
    manipulation_techniques = []
    biases_detected = []

    if request.method == "POST":
        if "submit_community" in request.form:
            feedback = {
                "text": request.form.get("original_text"),
                "flag": request.form.get("flag"),
                "user_source": request.form.get("user_source"),
                "vote": request.form.get("vote")
            }
            community_feedback.append(feedback)
        else:
            original_text = request.form.get("news")
            if not original_text:
                return render_template("index.html", error="No news text provided.")

            is_tamil = any('\u0B80' <= c <= '\u0BFF' for c in original_text)
            translated_text = translate_to_english(original_text) if is_tamil else original_text
            vec = vectorizer.transform([translated_text])
            pred = model.predict(vec)[0]

            result_en = "REAL news ✅" if pred == 1 else "FAKE news ❌"
            explanation_en = explain_news(translated_text, pred)

            suspicious = highlight_words(translated_text, SUSPICIOUS_WORDS)
            emotional = highlight_words(translated_text, EMOTIONAL_WORDS)
            source_status = check_source(translated_text)
            manipulation_techniques = detect_manipulation_techniques(translated_text)
            biases_detected = detect_cognitive_biases(translated_text)

            risk_label, risk_color = get_risk_level(pred, suspicious, emotional, source_status)

            if is_tamil:
                result_ta = translate_to_tamil(result_en)
                explanation_ta = translate_to_tamil(explanation_en)

    return render_template("index.html",
                           result_en=result_en,
                           explanation_en=explanation_en,
                           result_ta=result_ta,
                           explanation_ta=explanation_ta,
                           suspicious=suspicious,
                           emotional=emotional,
                           source_status=source_status,
                           original_text=original_text,
                           risk_label=risk_label,
                           risk_color=risk_color,
                           manipulation_techniques=manipulation_techniques,
                           biases_detected=biases_detected,
                           community_feedback=community_feedback)

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    incoming_msg = request.values.get("Body", "").strip()
    print(f"Received WhatsApp message: {incoming_msg}")

    if not incoming_msg:
        return "Error: No message body received.", 400

    is_tamil = any('\u0B80' <= c <= '\u0BFF' for c in incoming_msg)
    translated_text = translate_to_english(incoming_msg) if is_tamil else incoming_msg
    vec = vectorizer.transform([translated_text])
    pred = model.predict(vec)[0]
    result = "REAL news ✅" if pred == 1 else "FAKE news ❌"
    explanation = explain_news(translated_text, pred)

    # Format response using TwiML
    response = MessagingResponse()
    response.message(f"{result}\n\nExplanation:\n{explanation}")
    return str(response)


if __name__ == "__main__":
    app.run(debug=True)