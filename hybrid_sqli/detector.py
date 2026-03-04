# =====================================================
# 🔥 FULL HYBRID SQLi DETECTOR (Signature + RF + Save/Load)
# =====================================================

import re
import urllib.parse
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    roc_auc_score
)

# =====================================================
# 🔐 SIGNATURE DETECTOR
# =====================================================

class SignatureDetector:
    PATTERNS = {
        'basic_sqli': r"\b(or|and)\b\s+.+\s*=\s*.+",
        'union_based': r"\bunion\b\s+\bselect\b",
        'stacked_queries': r";\s*(drop|delete|insert|update|truncate)",
        'time_based': r"(sleep|benchmark|waitfor|pg_sleep)",
        'encoding_bypass': r"(%27|%23|%2d%2d|0x[0-9a-f]+|\\x[0-9a-f]+)"
    }

    def detect(self, text: str) -> int:
        score = 0
        for pattern in self.PATTERNS.values():
            if re.search(pattern, text):
                score += 20
        return min(score, 100)

# =====================================================
# 🤖 HYBRID SQLi RF DETECTOR
# =====================================================

class HybridSQLiRF:

    def __init__(self):
        self.signature_detector = SignatureDetector()
        self.vectorizer = None
        self.scaler = None
        self.model = None
        self.best_threshold = 50
        self.max_decode = 5

    # ---------- preprocess ----------
    def recursive_decode(self, text):
        for _ in range(self.max_decode):
            if '%' not in text:
                break
            new = urllib.parse.unquote(text)
            if new == text:
                break
            text = new
        return text

    def remove_comments(self, text):
        text = re.sub(r'(--|#).*?(\n|$)', ' ', text)
        text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.DOTALL)
        return text

    def skeletonize(self, text):
        text = re.sub(r'0x[0-9a-f]+', ' CONST_HEX ', text)
        text = re.sub(r"'[^']*'|\"[^\"]*\"", ' CONST_STR ', text)
        text = re.sub(r'\b(true|false|null)\b', ' CONST_BOOL ', text)
        text = re.sub(r'\b\d+(\.\d+)?\b', ' CONST_NUM ', text)
        return text

    def tokenize(self, text):
        return " ".join(re.findall(r"[a-zA-Z_]+|CONST_[A-Z_]+", text))

    def extract_stat_features(self, text):
        return np.array([
            len(text),
            text.count("'"),
            text.count(";"),
            text.count("="),
            text.count("--"),
            text.count("/*"),
            text.count(" or "),
            text.count(" and ")
        ])

    def transform(self, text):
        text = self.recursive_decode(text.lower())
        text = self.remove_comments(text)
        sig_score = self.signature_detector.detect(text)
        text = self.skeletonize(text)
        return (
            self.tokenize(text),
            sig_score,
            self.extract_stat_features(text)
        )

    # ---------- train ----------
    def train(self, df):
        processed = df["payload"].apply(self.transform)
        df["tokens"] = processed.apply(lambda x: x[0])
        df["sig"] = processed.apply(lambda x: x[1])
        df["stat"] = processed.apply(lambda x: x[2])

        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
        X_text = self.vectorizer.fit_transform(df["tokens"])

        X_sig = csr_matrix(df["sig"].values.reshape(-1,1))
        X_stat = csr_matrix(np.vstack(df["stat"].values))
        X = hstack([X_text, X_sig, X_stat])

        self.scaler = StandardScaler(with_mean=False)
        X = self.scaler.fit_transform(X)

        y = df["label"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2)

        self.model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1
        )
        self.model.fit(Xtr, ytr)

        probs = self.model.predict_proba(Xte)[:,1]
        sig = Xte[:, -9].toarray().flatten()
        scores = 0.6 * probs * 100 + 0.4 * sig

        p, r, t = precision_recall_curve(yte, scores)
        f1s = [f1_score(yte, scores >= th) for th in t]
        self.best_threshold = t[np.argmax(f1s)]

    # ---------- predict ----------
    def predict_single(self, payload):
        tok, sig, stat = self.transform(payload)
        X = hstack([
            self.vectorizer.transform([tok]),
            csr_matrix([[sig]]),
            csr_matrix([stat])
        ])
        X = self.scaler.transform(X)
        prob = self.model.predict_proba(X)[0][1]
        score = 0.6 * prob * 100 + 0.4 * sig

        return {
            "prediction": "MALICIOUS" if score >= self.best_threshold else "BENIGN",
            "final_score": score,
            "ml_probability": prob,
            "signature_score": sig
        }

    # ---------- save / load ----------
    def save_model(self, path):
        joblib.dump(self.__dict__, path)

    def load_model(self, path):
        self.__dict__.update(joblib.load(path))