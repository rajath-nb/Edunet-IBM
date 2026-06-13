import streamlit as st
import joblib
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
# This must be the very first Streamlit command
st.set_page_config(page_title="AGRIshield Command", page_icon="🌿", layout="centered")

# --- CUSTOM CSS FOR AESTHETICS ---
st.markdown("""
    <style>
    .main { background-color: #f9fdf9; }
    h1 { color: #1e5631; text-align: center; font-family: 'Montserrat', sans-serif; }
    .status-badge { padding: 10px; border-radius: 8px; font-weight: bold; text-align: center; margin-bottom: 20px;}
    .local-match { background-color: #cce5ff; color: #004085; }
    .cloud-match { background-color: #d4edda; color: #155724; }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
load_dotenv()

# FOOLPROOF API KEY FETCHING
# 1. Try to get it from Streamlit Cloud Secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
# 2. If that fails, try to get it from local PC environment
except (KeyError, FileNotFoundError):
    api_key = os.environ.get("GEMINI_API_KEY")

# 3. Stop the app gracefully if no key is found at all
if not api_key:
    st.error("🚨 API KEY IS MISSING! The app cannot connect to the AI. Please add it to Streamlit Secrets.")
    st.stop() 

# Configure Gemini with the correctly fetched key
genai.configure(api_key=api_key)


@st.cache_resource
def load_models():
    """Loads the ML models only once to save memory and speed up the app."""
    ml_model = joblib.load('agri_robot_model.pkl')
    le_plant = joblib.load('plant_encoder.pkl')
    le_disease = joblib.load('disease_encoder.pkl')
    return ml_model, le_plant, le_disease

@st.cache_resource
def load_gemini():
    return genai.GenerativeModel('gemini-2.5-flash')

ml_model, le_plant, le_disease = load_models()
llm_model = load_gemini()

# Base English Lists
english_plants = sorted(list(le_plant.classes_))
english_diseases = sorted(list(le_disease.classes_))

@st.cache_data
def translate_lists(language, plants, diseases):
    """Uses Gemini to translate UI lists and caches the result so it only runs once per language."""
    if language == 'English':
        return plants, diseases
        
    prompt = f"""
    Translate the following agricultural terms into {language}.
    Return ONLY a raw JSON object with two keys: "plants" (an array) and "diseases" (an array). 
    Do not use markdown backticks. Keep the exact same order as the English lists.
    Plants: {plants}
    Diseases: {diseases}
    """
    response = llm_model.generate_content(prompt)
    cleaned_text = response.text.strip().replace('```json', '').replace('```', '')
    translated_data = json.loads(cleaned_text)
    return translated_data.get("plants", plants), translated_data.get("diseases", diseases)


# --- MAIN UI ---
st.markdown("<h1>🌿 AGRIshield Command Center</h1>", unsafe_allow_html=True)
st.write("Welcome to the Hybrid AI Crop Diagnostic System. Please select your parameters below.")

# Language Selector
language = st.selectbox(
    "Select Language / भाषा चुनें / ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ:", 
    ["English", "Hindi", "Kannada", "Tamil", "Telugu", "Malayalam"]
)

# Fetch translated lists (or English if selected)
with st.spinner(f"Translating UI to {language}..." if language != 'English' else "Loading..."):
    display_plants, display_diseases = translate_lists(language, english_plants, english_diseases)

col1, col2 = st.columns(2)

with col1:
    # Use the translated lists for the dropdown display
    selected_display_plant = st.selectbox("Select Plant:", display_plants)
    # Find the matching English index so the ML model understands it!
    plant_index = display_plants.index(selected_display_plant)
    actual_english_plant = english_plants[plant_index]

with col2:
    # Option to select known disease or type a custom one
    selected_display_disease = st.selectbox("Select Symptoms/Disease:", ["Type custom symptom..."] + display_
