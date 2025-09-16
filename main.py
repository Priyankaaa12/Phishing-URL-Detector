import pandas as pd
import re
import tldextract
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

phish_data = pd.read_csv("phishing_sites_url.csv")
print(phish_data.columns)
phish_data.columns = phish_data.columns.str.strip()
phish_data = phish_data.dropna(subset=['Label'])
phish_data['Label'] = phish_data['Label'].map({'good':0, 'bad':1})
def url_entropy(url):
    prob = [url.count(c)/len(url) for c in set(url)]
    entropy = -sum([p * np.log2(p) for p in prob])
    return entropy


phish_data.head()

phish_data.tail()

def extract_features(url):
    features = []
    
    # Basic counts
    features.append(url.count('.'))
    features.append(url.count('-'))
    features.append(url.count('@'))
    features.append(len(url))
    features.append(url.count('?'))
    features.append(url.count('='))
    features.append(url.count('%'))
    features.append(url.count('#'))
    
    # HTTPS check
    features.append(1 if url.startswith("https") else 0)
    
    # Presence of IP address
    features.append(1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0)
    
    # Count digits
    features.append(sum(c.isdigit() for c in url))
    
    # Suspicious keywords
    suspicious_words = ['login','secure','update','verify','bank','account','confirm','ebay','paypal']
    features.append(1 if any(word in url.lower() for word in suspicious_words) else 0)
    
    # Domain parts
    ext = tldextract.extract(url)
    domain = ext.domain
    features.append(len(domain))
    features.append(1 if len(domain) > 15 else 0)      # Very long domain
    features.append(1 if any(char.isdigit() for char in domain) else 0)
    features.append(len(ext.subdomain.split('.')) if ext.subdomain else 0)
    features.append(1 if url.count('//') > 1 else 0)
    
    # Entropy of domain (detect random strings)
    features.append(url_entropy(domain))
    
    return features

X_features = np.array([extract_features(url) for url in phish_data['URL']])
y = phish_data['Label'].values

trainX, testX, trainY, testY = train_test_split(
    X_features, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(trainX, trainY)

predY = clf.predict(testX)

print("Accuracy:", accuracy_score(testY, predY))
print("\nClassification Report:\n", classification_report(testY, predY))
print("\nConfusion Matrix:\n", confusion_matrix(testY, predY))

import joblib

# Save model and scaler
joblib.dump(clf, "phishing_model.pkl")
joblib.dump(scaler, "scaler.pkl")

import joblib

clf = joblib.load("phishing_model.pkl")
scaler = joblib.load("scaler.pkl")
