import streamlit as st
import PyPDF2
import pandas as pd
from io import StringIO

# Funciones de predicción que ya has definido
# Asegúrate de que estas funciones estén incluidas en este archivo o importadas desde otros módulos
# Por ejemplo:
# from tu_codigo import vectorizar, naives_multinomial, logistic_regression, random_forest, support_vector_machine

# Función para extraer texto de un archivo PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Función para hacer la predicción con el texto extraído del PDF
def predict_from_pdf(text):
    # Asumimos que este es el parámetro que más ajusta
    # Parámetros: ngram=3, stopwords=0, stemming=1, min_df=3, CountVectorizer=True
    result = vectorizar([text], [text], None, None, 3, 0, 1, 3, 0)  # Esta llamada es un ejemplo
    return result

# Interfaz de Streamlit
st.title('Aplicación de Predicción Binaria desde PDF')

# Subir archivo PDF
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if uploaded_file is not None:
    # Extraer el texto del PDF
    text = extract_text_from_pdf(uploaded_file)
    
    # Mostrar el texto extraído (opcional)
    st.subheader('Texto extraído del PDF:')
    st.write(text[:1000])  # Mostrar solo los primeros 1000 caracteres para no sobrecargar la página
    
    # Realizar la predicción
    st.subheader('Resultado de la predicción:')
    result = predict_from_pdf(text)
    
    # Mostrar los resultados
    st.write(f"Resultados de las predicciones: {result}")

