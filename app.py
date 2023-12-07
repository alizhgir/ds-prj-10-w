import streamlit as st
import pandas as pd
import streamlit as st
import pickle
import time
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import transformers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import re
import string
import numpy as np
import torch.nn as nn
import json
import gensim
import torch.nn.functional as F

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


st.title('10-я неделя DS. Классификация отзывов, определение токсичности и генерация текста')

st.sidebar.header('Выберите страницу')
page = st.sidebar.radio("Выберите страницу", ["Вводная информация", "Классификация отзывов", "Определение токсичности", "Генерация текста"])

if page == "Вводная информация":
        
        st.subheader('*Задача №1*: Классификация отзывов на медицинские учреждения')
        st.write('Задача в двух словах: необходимо дать классификацию отзыва тремя моделями, время, за которое происходит классификаци отзыва, а также таблицу сравнения моделей по F-1 macro для моделей')

        st.subheader('*Задача №2*: Определение токсичности')
        st.write('Задача в двух словах: Оценка степени токсичности пользовательского сообщения ')

        st.subheader('*Задача №3*: Генерация текста')
        st.write('Задача в двух словах: Генерация текста GPT-моделью по пользовательскому prompt')

        st.subheader('☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️☀️')

        st.subheader('Выполнила команда "BERT": Алексей А., Светлана, Алиса')


if page == "Классификация отзывов":
    # Загрузка tf-idf модели и векторайзера
    with open('tf-idf/tf-idf.pkl', 'rb') as f:
        model_tf = pickle.load(f)

    with open('tf-idf/tf-idf_vectorizer.pkl', 'rb') as f:
        vectorizer_tf = pickle.load(f)

    # Загрузка словаря vocab_to_int и Word2Vec модели
    with open('lstm/vocab_to_int.json', 'r') as f:
        vocab_to_int = json.load(f)

    word2vec_model = gensim.models.Word2Vec.load("lstm/word2vec.model")
    
    def data_preprocessing(text: str) -> str:
        text = text.lower()
        text = re.sub('<.*?>', '', text) # html tags
        text = ''.join([c for c in text if c not in string.punctuation])# Remove punctuation
        text = ' '.join([word for word in text.split() if word not in stop_words])
        text = [word for word in text.split() if not word.isdigit()]
        text = ' '.join(text)
        return text

    # Функция для предсказания класса отзыва
    def classify_review_tf(review):
        # Векторизация отзыва
        review_vector = vectorizer_tf.transform([review])
        # Предсказание
        start_time = time.time()
        prediction = model_tf.predict(review_vector)
        end_time = time.time()
        # Время предсказания
        prediction_time = end_time - start_time
        return prediction[0], prediction_time
    
    VOCAB_SIZE = len(vocab_to_int) + 1  # add 1 for the padding token
    EMBEDDING_DIM = 32
    HIDDEN_SIZE = 32
    SEQ_LEN = 100

    class BahdanauAttention(nn.Module):
        def __init__(self, hidden_size: torch.Tensor = HIDDEN_SIZE) -> None:
            super().__init__()

            self.W_q = nn.Linear(hidden_size, hidden_size)
            self.W_k = nn.Linear(hidden_size, hidden_size)
            self.V = nn.Linear(HIDDEN_SIZE, 1)

        def forward(
            self,
            keys: torch.Tensor,
            query: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            query = self.W_q(query)
            keys = self.W_k(keys)

            energy = self.V(torch.tanh(query.unsqueeze(1) + keys)).squeeze(-1)
            weights = F.softmax(energy, -1)
            context = torch.bmm(weights.unsqueeze(1), keys)
            return context, weights
    
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    embedding_layer = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))

    class LSTMConcatAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            # self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
            self.embedding = embedding_layer
            self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True)
            self.attn = BahdanauAttention(HIDDEN_SIZE)
            self.clf = nn.Sequential(
                nn.Linear(HIDDEN_SIZE, 128),
                nn.Dropout(),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
        
        def forward(self, x):
            embeddings = self.embedding(x)
            outputs, (h_n, _) = self.lstm(embeddings)
            att_hidden, att_weights = self.attn(outputs, h_n.squeeze(0))
            out = self.clf(att_hidden)
            return out, att_weights
        
    model_lstm = LSTMConcatAttention()  # Инициализируйте с теми же параметрами, что использовались при обучении
    model_lstm.load_state_dict(torch.load("lstm/lstm_model.pth"))
    model_lstm.eval()

    def text_to_vector(text):
        words = text.split()
        vector = [vocab_to_int.get(word, vocab_to_int) for word in words]
        return np.array(vector)
    
    def classify_review_lstm(review):
        # Векторизация отзыва
        review_vector = text_to_vector(review)
        # Преобразование в тензор PyTorch и добавление размерности пакета (batch)
        review_tensor = torch.tensor(review_vector).unsqueeze(0)
        
        # Предсказание
        start_time = time.time()
        with torch.no_grad():
            prediction, _ = model_lstm(review_tensor)
        end_time = time.time()

        # Время предсказания
        prediction_time = end_time - start_time
        return prediction, prediction_time

    # Создание интерфейса Streamlit
    st.title('Классификатор отзывов на клиники')

    # Текстовое поле для ввода отзыва
    user_review = st.text_input('Введите ваш отзыв на клинику')

    if st.button('Классифицировать'):
        if user_review:
            # Классификация отзыва
            prediction_tf, pred_time_tf = classify_review_tf(user_review)
            st.write(f'Предсказанный класс TF-IDF: {prediction_tf}')
            st.write(f'Время предсказания TF-IDF: {pred_time_tf:.4f} секунд')
            prediction_lstm, pred_time_lstm = classify_review_lstm(user_review)
            st.write(f'Предсказанный класс LSTM: {prediction_tf}')
            st.write(f'Время предсказания LSTM: {pred_time_tf:.4f} секунд')
        else:
            st.write('Пожалуйста, введите отзыв')





# if page == "Определение токсичности":
    
    
# if page == "Генерация текста":


