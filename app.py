import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---- config: these files are produced by precompute_embeddings.py ----
DATA_PATH = "chatbot_data.csv"                 # columns: Patient, Doctor, processed_question
EMBEDDINGS_PATH = "question_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"
ANSWER_SIM_THRESHOLD = 0.35

st.set_page_config(page_title="Medical Chatbot (BERT)", page_icon="🩺")


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_data
def load_data_and_embeddings():
    df = pd.read_csv(DATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    return df, embeddings


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


def preprocess_query(text):
    text = normalize_text(text)
    tokens = word_tokenize(text)
    tokens = clean_tokens(tokens)
    return " ".join(tokens)


def chatbot_response(user_query, model, df, question_embeddings, threshold=ANSWER_SIM_THRESHOLD):
    processed = preprocess_query(user_query)
    query_embedding = model.encode([processed], convert_to_numpy=True)
    sims = cosine_similarity(query_embedding, question_embeddings)[0]
    best_idx = int(np.argmax(sims))
    score = float(sims[best_idx])

    if score < threshold:
        return "Maaf, saya belum menemukan jawaban yang cukup relevan untuk pertanyaan ini. Silakan konsultasikan ke dokter.", score

    answer = df.iloc[best_idx]["Doctor"]
    return answer + "\n\n*(Catatan: ini bukan pengganti diagnosis dokter.)*", score


# ---- UI ----
st.title("🩺 Medical Chatbot")
st.caption("Retrieval-based chatbot menggunakan BERT sentence embeddings. Bukan pengganti konsultasi medis. Use English for better results.")

model = load_model()
df, question_embeddings = load_data_and_embeddings()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Tanyakan sesuatu tentang kesehatan...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    answer, score = chatbot_response(user_input, model, df, question_embeddings)

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(f"Match score: {score:.2f}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
