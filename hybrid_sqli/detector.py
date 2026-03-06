# =====================================================
# 🔥 FULL HYBRID SQLi DETECTOR (Auto-Weight + RF + Signature Block)
# =====================================================

import re
import urllib.parse
import numpy as np
import pandas as pd
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
# 🔐 ADVANCED SQLi SIGNATURE DETECTOR
# =====================================================

class SignatureDetector:

    PATTERNS = {

        # ------------------------------------------------
        # 1️⃣ Authentication Bypass
        # ------------------------------------------------
        'auth_bypass': r"\b(or|and)\b\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?",

        # ------------------------------------------------
        # 2️⃣ UNION Based SQLi
        # ------------------------------------------------
        'union_based': r"\bunion\b\s+(all\s+)?\bselect\b",

        # ------------------------------------------------
        # 3️⃣ Stacked Queries
        # ------------------------------------------------
        'stacked_queries': r";\s*(drop|delete|insert|update|truncate|alter|create)",

        # ------------------------------------------------
        # 4️⃣ Time Based Blind SQLi
        # ------------------------------------------------
        'time_based': r"\b(sleep|benchmark|waitfor\s+delay|pg_sleep|dbms_lock\.sleep)\b",

        # ------------------------------------------------
        # 5️⃣ Error Based SQLi
        # ------------------------------------------------
        'error_based': r"\b(updatexml|extractvalue|floor\(|geometrycollection|multipoint|exp\(|rand\()", 

        # ------------------------------------------------
        # 6️⃣ Information Extraction
        # ------------------------------------------------
        'schema_enum': r"\b(information_schema|table_name|column_name|database\(\)|version\(\))\b",

        # ------------------------------------------------
        # 7️⃣ Boolean Blind SQLi
        # ------------------------------------------------
        'boolean_blind': r"\b(and|or)\b\s+\d+\s*=\s*\d+",

        # ------------------------------------------------
        # 8️⃣ Comment Bypass
        # ------------------------------------------------
        'comment_bypass': r"(--|#|/\*|\*/)",

        # ------------------------------------------------
        # 9️⃣ Encoding Bypass
        # ------------------------------------------------
        'encoding_bypass': r"(%27|%23|%2d%2d|0x[0-9a-f]+|\\x[0-9a-f]+)",

        # ------------------------------------------------
        # 🔟 SQL Functions often used in SQLi
        # ------------------------------------------------
        'sqli_functions': r"\b(concat|load_file|substring|ascii|char|hex)\s*\(",
    }

    def detect(self, text: str) -> int:
        score = 0

        for pattern in self.PATTERNS.values():
            if re.search(pattern, text):
                score += 10

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

        self.max_decode = 5

        self.best_threshold = 50

        # auto tuned weights
        self.w_ml = 0.6
        self.w_sig = 0.4

    # -----------------------------
    # Dataset Cleaning
    # -----------------------------
    def clean_dataset(self, df):

        df = df.dropna(subset=['payload']).drop_duplicates()
        df = df[df['payload'].str.strip() != ""]
        df = df[df['payload'].str.len() >= 7]
        df = df[~df['payload'].str.match(r'^\d+$')]

        return df

    # -----------------------------
    # Recursive Decode
    # -----------------------------
    def recursive_decode(self, text):

        count = 0

        while '%' in text and count < self.max_decode:

            new = urllib.parse.unquote(text)

            if new == text:
                break

            text = new
            count += 1

        if count >= self.max_decode:
            return None

        return text

    # -----------------------------
    # Remove Comments
    # -----------------------------
    def remove_comments(self, text):

        text = re.sub(r'(--|#).*?(\n|$)', ' ', text)
        text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.DOTALL)

        return text

    # -----------------------------
    # Skeletonization
    # -----------------------------
    def skeletonize(self, text):

        text = re.sub(r'0x[0-9a-f]+', ' CONST_HEX ', text)
        text = re.sub(r"'[^']*'", ' CONST_STR ', text)
        text = re.sub(r'"[^"]*"', ' CONST_STR ', text)
        text = re.sub(r'\b(true|false|null)\b', ' CONST_BOOL ', text)
        text = re.sub(r'\b\d+(\.\d+)?\b', ' CONST_NUM ', text)

        return text

    # -----------------------------
    # Tokenization
    # -----------------------------
    def tokenize(self, text):

        tokens = re.findall(r"[a-zA-Z_]+|CONST_[A-Z_]+", text)

        return " ".join(tokens)

    # -----------------------------
    # Statistical Features
    # -----------------------------
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

    # -----------------------------
    # Transform
    # -----------------------------
    def transform(self, text):

        text = self.recursive_decode(text)

        if text is None:
            return None

        text = text.lower()
        text = self.remove_comments(text)

        text = re.sub(r'\s+', ' ', text)

        sig_score = self.signature_detector.detect(text)

        text = self.skeletonize(text)

        tokenized = self.tokenize(text)

        stat = self.extract_stat_features(text)

        return tokenized, sig_score, stat

    # -----------------------------
    # Train
    # -----------------------------
    def train(self, df):

        df = self.clean_dataset(df)

        processed = df['payload'].apply(self.transform)

        processed = processed.dropna()

        df = df.loc[processed.index]

        df['tokens'] = processed.apply(lambda x: x[0])
        df['sig_score'] = processed.apply(lambda x: x[1])
        df['stat'] = processed.apply(lambda x: x[2])

        X_text = df['tokens']
        y = df['label'].values

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,3)
        )

        X_tfidf = self.vectorizer.fit_transform(X_text)

        X_sig = csr_matrix(df['sig_score'].values.reshape(-1,1))
        X_stat = csr_matrix(np.vstack(df['stat'].values))

        X_full = hstack([X_tfidf, X_sig, X_stat])

        self.scaler = StandardScaler(with_mean=False)

        X_scaled = self.scaler.fit_transform(X_full)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        self.model = RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        self.evaluate(X_test, y_test)

    # -----------------------------
    # Evaluation + Auto Weight
    # -----------------------------
    def evaluate(self, X_test, y_test):

        test_probs = self.model.predict_proba(X_test)[:,1]

        sig_feature_test = X_test[:, -9].toarray().flatten()

        best_f1 = 0
        best_w_ml = 0.6
        best_w_sig = 0.4
        best_t = 50

        # auto weight search
        for w_ml in np.arange(0.1,1.0,0.1):

            w_sig = 1 - w_ml

            final_scores = (w_ml * test_probs * 100) + (w_sig * sig_feature_test)

            precision, recall, thresholds = precision_recall_curve(
                y_test,
                final_scores
            )

            for t in thresholds:

                preds = (final_scores >= t).astype(int)

                f1 = f1_score(y_test, preds)

                if f1 > best_f1:

                    best_f1 = f1
                    best_w_ml = w_ml
                    best_w_sig = w_sig
                    best_t = t

        self.w_ml = best_w_ml
        self.w_sig = best_w_sig
        self.best_threshold = best_t

        final_scores = (self.w_ml * test_probs * 100) + (self.w_sig * sig_feature_test)

        preds = (final_scores >= self.best_threshold).astype(int)

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, preds))

        print("ROC-AUC:", roc_auc_score(y_test, final_scores))

        print("\nBest ML Weight:", self.w_ml)
        print("Best Signature Weight:", self.w_sig)
        print("Best Threshold:", self.best_threshold)
        print("Best F1:", best_f1)

    # -----------------------------
    # Predict
    # -----------------------------
    def predict_single(self, payload):

        result = self.transform(payload)

        if result is None:
            return {"prediction":"MALICIOUS","reason":"Excessive Encoding"}

        tokenized, sig_score, stat = result

        # HARD BLOCK
        if sig_score >= 80:

            return {
                "prediction":"MALICIOUS",
                "reason":"High confidence SQLi signature",
                "signature_score":sig_score
            }

        X_tfidf = self.vectorizer.transform([tokenized])

        X_sig = csr_matrix([[sig_score]])

        X_stat = csr_matrix([stat])

        X_full = hstack([X_tfidf,X_sig,X_stat])

        X_scaled = self.scaler.transform(X_full)

        prob = self.model.predict_proba(X_scaled)[0][1]

        final_score = (self.w_ml * prob * 100) + (self.w_sig * sig_score)

        prediction = "MALICIOUS" if final_score >= self.best_threshold else "BENIGN"

        return {
            "prediction":prediction,
            "ml_probability":prob,
            "signature_score":sig_score,
            "final_score":final_score
        }

    # -----------------------------
    # Save Model
    # -----------------------------
    def save_model(self,path="hybrid_sqli_model.pkl"):

        joblib.dump({
            "model":self.model,
            "vectorizer":self.vectorizer,
            "scaler":self.scaler,
            "threshold":self.best_threshold,
            "w_ml":self.w_ml,
            "w_sig":self.w_sig
        },path)

    # -----------------------------
    # Load Model
    # -----------------------------
    def load_model(self,path="hybrid_sqli_model.pkl"):

        data = joblib.load(path)

        self.model = data["model"]
        self.vectorizer = data["vectorizer"]
        self.scaler = data["scaler"]
        self.best_threshold = data["threshold"]
        self.w_ml = data["w_ml"]
        self.w_sig = data["w_sig"]