import streamlit as st
import PyPDF2
import sklearn
import pandas as pd
from io import StringIO
from joblib import dump, load
import os
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
# Descargar las stopwords en español
nltk.download('stopwords')
nltk.download('punkt')

# Función para tokenizar y aplicar stemming
stemmer = SnowballStemmer("spanish")
spanish_stopwords = stopwords.words('spanish')
spanish_stopwords_stemmed = [stemmer.stem(word) for word in spanish_stopwords]

def tokenize_and_stem_stop(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [w for w in tokens if w not in spanish_stopwords]
    return [stemmer.stem(token) for token in tokens]

# Función para tokenizar y aplicar stemming
def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    return stem_tokens(tokens)

def normalizar_resuelve(texto):
    # Definir patrones de variaciones de "resuelve"
    patrones = [
        re.compile(r'res\s*ue\s*lve', re.IGNORECASE),  # res u e l v e
        re.compile(r're\s*s\s*u\s*e\s*l\s*v\s*e', re.IGNORECASE),  # re s u e l v e
        re.compile(r'r\s*e\s*s\s*u\s*e\s*l\s*v\s*e', re.IGNORECASE),  # r e s u e l v e
        re.compile(r'resu\s*ve', re.IGNORECASE),  # resu ve
        re.compile(r'resuelvf', re.IGNORECASE),  # resuelvf
        re.compile(r'resuelvt', re.IGNORECASE),  # resuelvt
        re.compile(r'rtsuelve', re.IGNORECASE),  # rtsuelve
        re.compile(r'r\s*e\s*s\s*u\s*e\s*l\s*v\s*e', re.IGNORECASE),  # r e s u e l 1 e
        re.compile(r'rei\s*elve', re.IGNORECASE),  # re i elve
        re.compile(r'r\s*e\s*s\s*u\s*e\s*l\s*v\s*e', re.IGNORECASE),  # r e s u e í- v e
        re.compile(r'resuelv', re.IGNORECASE),  # resuelv
        re.compile(r'res\s*lve', re.IGNORECASE),  # res lve
        re.compile(r're\s*5\s*u\s*e\s*l\s*v\s*e', re.IGNORECASE),  # re 5 u el v e
        re.compile(r'1\s*u\s*e\s*l\s*v\s*e', re.IGNORECASE),  # 1 u el v e
        re.compile(r'e\s*su\s*el\s*v\s*e', re.IGNORECASE),  # e su el v e
        re.compile(r'8\s*u\s*e\s*l\s*v\s*e', re.IGNORECASE),  # 8 u el v e
        re.compile(r'r\s*-\s*e\s*su\s*el\s*v\s*e', re.IGNORECASE),  # r-e su el v e
        re.compile(r'r\s*e\s*s\s*lj\s*e\s*l\s*v\s*e', re.IGNORECASE),  # r e s lj e l v e
        re.compile(r'rl\s*v\sees\s*ll\s*e', re.IGNORECASE),  # rl v ees ll e
        re.compile(r'r\s*e\s*s\s*lj\s*e\s*l\s*v\s*e', re.IGNORECASE),  # r e s lj e l v e 
        re.compile(r'rl\s*v\s*e', re.IGNORECASE),  # rl v e
        re.compile(r'r\s*e\s*s\s*lj\s*e\s*l\s*v\s*e', re.IGNORECASE),  # r e s lj e l v e
        re.compile(r'e\s*s\s*u\s*e\s*l\s*v\s*e', re.IGNORECASE),  # e s u e l v e
        re.compile(r's\s*u\s*e\s*l\s*v\s*e', re.IGNORECASE),  # s u e l v e
        re.compile(r're\s*l\s*v\s*e', re.IGNORECASE),  # re l v e
        re.compile(r'\s*l\s*v\s*e\s*', re.IGNORECASE),  #  l v e 
        re.compile(r'r\s*e\s*e\s*l\s*v\s*e', re.IGNORECASE),  # r e e l v e
        re.compile(r'res\s*l\s*v\s*e', re.IGNORECASE)  # res l v e
    ]
    # Buscar coincidencias y reemplazar con "resuelve"
    for patron in patrones:
        texto = patron.sub('resuelve', texto)
    return texto

# Función para extraer texto de un archivo PDF
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ''
        texto_pre_resuelve = ''
        for page in reader.pages:
            text += page.extract_text()

        text = normalizar_resuelve(text)
        posicion_ultimo_resuelve = text.rfind("resuelve")
        # Si se encuentra "resuelve", extraer el texto que sigue
        if posicion_ultimo_resuelve != -1:
            texto_pre_resuelve = text[:posicion_ultimo_resuelve];

        return texto_pre_resuelve
    except Exception as e:
        st.error(f"Error al extraer texto del PDF: {e}")
        return ""

datos_vectorizados_path = "models/datos_vectorizados.joblib"
vectorizador_path = "models/vectorizador.joblib"
modelo_path = "models/modelo.joblib"

def cargar_modelo_y_predecir(vectorizador_path, modelo_path, nuevo_texto):
    try:
        # Cargar el vectorizador
        vect = load(vectorizador_path)
        st.text("Vectorizador cargado.")

        # Cargar el modelo
        modelo = load(modelo_path)
        st.text("Modelo cargado.")

        # Vectorizar el nuevo texto
        nuevo_texto_vectorizado = vect.transform([nuevo_texto])

        # Realizar la predicción
        prediccion = modelo.predict(nuevo_texto_vectorizado)
        prediccion_prob = modelo.predict_proba(nuevo_texto_vectorizado)[:,1]  # Si es clasificación binaria

        st.text(f"Predicción: {prediccion[0]}")
        st.text(f"Probabilidad de clase 1: {prediccion_prob[0]:.2f}")
        return prediccion
    except Exception as e:
        st.error(f"Error al cargar el modelo o hacer la predicción: {e}")
        return None

# Interfaz de Streamlit
st.title('Predicción de sentencias de la Sala Penal de la Corte Suprema de Justicia del Paraguay')

# Subir archivo PDF
uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")

if uploaded_file is not None:
    # Extraer el texto del PDF
    # st.subheader('Texto extraído del PDF:')
    nuevo_texto = extract_text_from_pdf(uploaded_file)

    
    if nuevo_texto:  # Si el texto se extrajo correctamente
        # Mostrar el texto extraído (opcional)
        #st.write(nuevo_texto[:1000])  # Mostrar solo los primeros 1000 caracteres

        # Realizar la predicción
        st.subheader('Resultado de la predicción:')
        prediccion = cargar_modelo_y_predecir(vectorizador_path, modelo_path, nuevo_texto)
        respuesta = 'Positivo' if prediccion == 1 else 'Negativo'

        # Mostrar los resultados
        if prediccion is not None:
            st.markdown(f"Resultados de la predicción: <span style='color:maroon; font-weight:bold;'>{respuesta}</span>", unsafe_allow_html=True)
