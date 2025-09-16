import numpy as np
import joblib
import re
import tldextract

# 🔹 same extract_features function you used during training
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
    prob = [url.count(c)/len(url) for c in set(url)]
    entropy = -sum([p * np.log2(p) for p in prob])
    features.append(entropy)
    return features

# 🔹 load model & scaler
clf = joblib.load("phishing_model.pkl")
scaler = joblib.load("scaler.pkl")

# 🔹 prediction function
def predict_url(url):
    feats = np.array(extract_features(url)).reshape(1, -1)
    feats = scaler.transform(feats)
    pred = clf.predict(feats)[0]
    return "Good" if pred == 0 else "Phishing"

# 🔹 test examples
print(predict_url("http://microsoft-support.secureauth.com/login"))   # should say Phishing
print(predict_url("https://www.paypal.com"))  # likely Phishing
print(predict_url("https://www.google.com"))        # Good
