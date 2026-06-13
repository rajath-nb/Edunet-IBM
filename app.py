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

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6LroP4Xdk51llGF096DlQ58-6ridGUlAuQjjGxYLBuzzA"))

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
    selected_display_disease = st.selectbox("Select Symptoms/Disease:", ["Type custom symptom..."] + display_diseases)
    
    custom_disease = ""
    actual_english_disease = ""
    
    if selected_display_disease == "Type custom symptom...":
        custom_disease = st.text_input("Describe the symptom in your language:")
        actual_english_disease = custom_disease # Will be translated later if needed
    else:
        disease_index = display_diseases.index(selected_display_disease)
        actual_english_disease = english_diseases[disease_index]

# --- PREDICTION LOGIC ---
if st.button("Analyze & Generate Report", use_container_width=True):
    
    if selected_display_disease == "Type custom symptom..." and not custom_disease:
        st.warning("Please describe the symptom.")
    else:
        with st.spinner("⏳ AI is Scanning & Analyzing..."):
            try:
                # SCENARIO A: Local Match (They picked from the dropdowns)
                if selected_display_disease != "Type custom symptom...":
                    
                    # Local ML Prediction
                    plant_num = le_plant.transform([actual_english_plant])[0]
                    disease_num = le_disease.transform([actual_english_disease])[0]
                    ml_prediction = ml_model.predict([[plant_num, disease_num]])[0]
                    
                    # Gemini Translation & Formatting
                    prompt = f"""
                    You are an expert agricultural botanist. Respond entirely in {language}.
                    A farmer has a {actual_english_plant} crop suffering from {actual_english_disease}. 
                    Our local database suggests this treatment: {ml_prediction}.
                    
                    Provide:
                    1. A translation of the plant name '{actual_english_plant}', disease '{actual_english_disease}', and treatment '{ml_prediction}' into {language}.
                    2. A concise report with these sections:
                       * **Root Causes:** How it develops.
                       * **Best Herbal Practices:** Organic application.
                       * **Long-Term Prevention:** Non-chemical methods.
                    Format in Markdown. Ensure the tone is helpful for a farmer.
                    """
                    response = llm_model.generate_content(prompt)
                    
                    st.markdown(f"<div class='status-badge local-match'>✅ Fast Reflex Local Match ({language})</div>", unsafe_allow_html=True)
                    st.success(f"Prescribed Action: Generated via Local ML")
                    st.markdown(response.text)

                # SCENARIO B: Cloud Fallback (They typed a custom unknown symptom)
                else:
                    # Translate their custom symptom to English first if necessary
                    if language != 'English':
                        trans_prompt = f"Translate this agricultural symptom from {language} to English. Just return the English text: '{custom_disease}'"
                        actual_english_disease = llm_model.generate_content(trans_prompt).text.strip()
                    
                    prompt = f"""
                    You are an expert agricultural botanist. Respond entirely in {language}.
                    A farmer has inputted a plant: '{actual_english_plant}' with custom symptoms: '{actual_english_disease}'. 
                    This is an edge case not in our local database.
                    
                    Provide a comprehensive diagnostic report in {language} including:
                    * **Possible Diseases:** What are the most likely issues based on the symptoms?
                    * **Recommended Treatment:** Best organic/herbal treatments.
                    * **Long-Term Prevention:** Future safety.
                    Format in Markdown.
                    """
                    response = llm_model.generate_content(prompt)
                    
                    st.markdown(f"<div class='status-badge cloud-match'>🌐 Cloud AI Deep Analysis ({language})</div>", unsafe_allow_html=True)
                    st.info("Prescribed Action: Cloud Diagnostic Complete")
                    st.markdown(response.text)
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")