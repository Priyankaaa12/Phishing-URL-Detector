from flask import Flask, render_template, request
import numpy as np
import joblib
import tldextract
import re
import math

# Load model & scaler
clf = joblib.load("phishing_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

# --- Feature extraction function ---
def url_entropy(url):
    prob = [url.count(c)/len(url) for c in set(url)]
    entropy = -sum([p * math.log2(p) for p in prob])
    return entropy

def extract_features(url):
    features = []
    features.append(url.count('.'))
    features.append(url.count('-'))
    features.append(url.count('@'))
    features.append(len(url))
    features.append(url.count('?'))
    features.append(url.count('='))
    features.append(url.count('%'))
    features.append(url.count('#'))

    features.append(1 if url.startswith("https") else 0)
    features.append(1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0)
    features.append(sum(c.isdigit() for c in url))

    suspicious_words = ['login','secure','update','verify','bank','account','confirm','ebay','paypal']
    features.append(1 if any(word in url.lower() for word in suspicious_words) else 0)

    ext = tldextract.extract(url)
    domain = ext.domain
    features.append(len(domain))
    features.append(1 if len(domain) > 15 else 0)
    features.append(1 if any(char.isdigit() for char in domain) else 0)
    features.append(len(ext.subdomain.split('.')) if ext.subdomain else 0)
    features.append(1 if url.count('//') > 1 else 0)
    features.append(url_entropy(domain))
    
    return np.array(features).reshape(1, -1)

# --- Prediction function ---
def predict_url(url):
    try:
        feats = extract_features(url)
        feats = scaler.transform(feats)
        pred = clf.predict(feats)[0]

        if pred == 1:  # phishing
            return {"code": "bad", "label": "Phishing URL"}
        else:          # legit
            return {"code": "good", "label": "Legit URL"}
    except:
        return {"code": "error", "label": "Invalid URL / Error"}

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    url = ""
    if request.method == "POST":
        url = request.form["url"]
        result = predict_url(url)
    return render_template("index.html", result=result, url=url)

if __name__ == "__main__":
    app.run(debug=True)
