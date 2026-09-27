"""
Run this ONCE locally (or in Colab) before deploying, to produce the two
files app.py needs: chatbot_data.csv and question_embeddings.npy.

This mirrors Tahap 1-7 from your notebook, condensed into a script.
"""
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

CSV_PATH = "fixed_dataset.csv"   # your cleaned CSV from the notebook
QUESTION_COL = "Patient"
ANSWER_COL = "Doctor"
MODEL_NAME = "all-MiniLM-L6-v2"
MIN_ANSWER_WORDS = 20             # drop stub/referral-only answers
SAMPLE_SIZE = 20000               # keep this modest for free-tier hosting; None = use all rows

df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")
df = df[[QUESTION_COL, ANSWER_COL]].dropna()
df = df.drop_duplicates(subset=[QUESTION_COL]).reset_index(drop=True)
df = df[df[ANSWER_COL].str.split().str.len() >= MIN_ANSWER_WORDS].reset_index(drop=True)
if SAMPLE_SIZE:
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_tokens(tokens):
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


df["normalized_question"] = df[QUESTION_COL].apply(normalize_text)
df["tokens"] = df["normalized_question"].apply(word_tokenize)
df["clean_tokens"] = df["tokens"].apply(clean_tokens)
df["processed_question"] = df["clean_tokens"].apply(lambda toks: " ".join(toks))

model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    df["processed_question"].tolist(), batch_size=64, show_progress_bar=True, convert_to_numpy=True
)

df[["Patient", "Doctor", "processed_question"]].to_csv("chatbot_data.csv", index=False)
np.save("question_embeddings.npy", embeddings)

print("Saved chatbot_data.csv and question_embeddings.npy")
print("Rows:", len(df), "| Embedding shape:", embeddings.shape)
