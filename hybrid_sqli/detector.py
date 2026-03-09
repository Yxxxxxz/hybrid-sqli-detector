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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# =====================================================
# Stage 2 : Signature Detector (Regex Only)
# =====================================================

class SignatureDetector:

    REGEX_PATTERNS = {

        "union_based": r"\bunion\s+(all\s+)?select\b",

        "error_based": r"\b(updatexml|extractvalue|floor\(|geometrycollection|multipoint|exp\(|rand\()", 

        "time_based": r"\b(sleep|benchmark|waitfor\s+delay|pg_sleep|dbms_lock\.sleep)\b",

        "boolean_based": r"\b(and|or)\b\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?"
    }

    def detect(self, text):

        for name, pattern in self.REGEX_PATTERNS.items():

            if re.search(pattern, text):

                return True, name

        return False, None


# =====================================================
# Hybrid SQLi Detector
# =====================================================

class SQLiDetector:

    def __init__(self):

        self.signature_detector = SignatureDetector()

        self.vectorizer = None
        self.scaler = None
        self.model = None

        self.max_decode = 5

    # =================================================
    # Stage 1 : Normalization
    # =================================================

    def recursive_decode(self, text):

        count = 0

        while "%" in text and count < self.max_decode:

            new = urllib.parse.unquote(text)

            if new == text:
                break

            text = new
            count += 1

        return text


    def remove_comments(self, text):

        text = re.sub(r'(--|#).*?(\n|$)', ' ', text)
        text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.DOTALL)

        return text


    def normalize(self, text):

        text = self.recursive_decode(text)

        text = text.lower()

        text = self.remove_comments(text)

        text = re.sub(r'\s+', ' ', text)

        return text


    # =================================================
    # Stage 3 : ML Preprocessing
    # =================================================

    def skeletonize(self, text):

        text = re.sub(r'0x[0-9a-f]+', ' CONST_HEX ', text)

        text = re.sub(r"'[^']*'", ' CONST_STR ', text)
        text = re.sub(r'"[^"]*"', ' CONST_STR ', text)

        text = re.sub(r'\b(true|false|null)\b', ' CONST_BOOL ', text)

        text = re.sub(r'\b\d+(\.\d+)?\b', ' CONST_NUM ', text)

        return text


    def tokenize(self, text):

        tokens = re.findall(r"[a-zA-Z_]+|CONST_[A-Z_]+", text)

        return " ".join(tokens)


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


    # =================================================
    # Transform
    # =================================================

    def transform(self, text):

        normalized = self.normalize(text)

        skeleton = self.skeletonize(normalized)

        tokens = self.tokenize(skeleton)

        stat = self.extract_stat_features(skeleton)

        return tokens, stat


    # =================================================
    # Dataset Cleaning
    # =================================================

    def clean_dataset(self, df):

        df = df.dropna(subset=['payload'])

        df = df.drop_duplicates()

        df = df[df['payload'].str.strip() != ""]

        df = df[df['payload'].str.len() >= 7]

        df = df[~df['payload'].str.match(r'^\d+$')]

        return df


    # =================================================
    # Train
    # =================================================

    def train(self, df):

        df = self.clean_dataset(df)

        processed = df['payload'].apply(self.transform)

        df['tokens'] = processed.apply(lambda x: x[0])
        df['stat'] = processed.apply(lambda x: x[1])

        X_text = df['tokens']

        y = df['label'].values


        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,3)
        )


        X_tfidf = self.vectorizer.fit_transform(X_text)

        X_stat = csr_matrix(np.vstack(df['stat'].values))

        X_full = hstack([X_tfidf, X_stat])


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


    # =================================================
    # Evaluation
    # =================================================

    def evaluate(self, X_test, y_test):

        preds = self.model.predict(X_test)

        cm = confusion_matrix(y_test, preds)

        accuracy = accuracy_score(y_test, preds)

        precision = precision_score(y_test, preds)

        recall = recall_score(y_test, preds)

        f1 = f1_score(y_test, preds)


        print("\n==============================")
        print("MODEL PERFORMANCE SUMMARY")
        print("==============================")

        print("\nConfusion Matrix")
        print(cm)

        print("\nAccuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)

        print("\nDetailed Report")
        print(classification_report(y_test, preds))


    # =================================================
    # Prediction
    # =================================================

    def predict_single(self, payload):

        normalized = self.normalize(payload)

        # Stage 2: Regex Signature
        sig_detected, sig_type = self.signature_detector.detect(normalized)

        if sig_detected:

            return {

                "prediction": "MALICIOUS",
                "stage": "Regex Signature",
                "reason": sig_type,
                "payload": normalized

            }


        # Stage 3: ML Detection

        tokens, stat = self.transform(payload)

        X_tfidf = self.vectorizer.transform([tokens])

        X_stat = csr_matrix([stat])

        X_full = hstack([X_tfidf, X_stat])

        X_scaled = self.scaler.transform(X_full)

        prob = self.model.predict_proba(X_scaled)[0][1]

        prediction = "MALICIOUS" if prob >= 0.5 else "BENIGN"


        return {

            "prediction": prediction,
            "stage": "ML Detection",
            "ml_probability": prob,
            "payload": normalized

        }


    # =================================================
    # Save / Load
    # =================================================

    def save_model(self, path="sqli_detector.pkl"):

        joblib.dump({

            "model": self.model,
            "vectorizer": self.vectorizer,
            "scaler": self.scaler

        }, path)


    def load_model(self, path="sqli_detector.pkl"):

        data = joblib.load(path)

        self.model = data["model"]
        self.vectorizer = data["vectorizer"]
        self.scaler = data["scaler"]