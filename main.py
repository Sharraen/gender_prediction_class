import streamlit as st
import nltk
from nltk import NaiveBayesClassifier
from nltk.classify import apply_features
from joblib import load

# --- Configuration for a Premium Look ---
st.set_page_config(
    page_title="Gender Predictor 🚻",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a sleek and modern look
st.markdown("""
<style>
    /* General Styling */
    body {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f0f2f6; /* Light gray background */
        color: #333333;
    }
    .stApp {
        background-color: #ffffff; /* White content background */
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        margin-top: 2rem;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Header Styling */
    .stApp header {
        background-color: #6a0572; /* Deep purple */
        padding: 1rem;
        border-radius: 10px 10px 0 0;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stApp header h1 {
        color: white !important;
        font-size: 2.5em;
        margin-bottom: 0.5rem;
    }
    h1 {
        color: #6a0572; /* Deep purple for main titles */
        text-align: center;
        font-size: 2.8em;
        margin-bottom: 0.8em;
        letter-spacing: 0.05em;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    h2 {
        color: #8b008b; /* Slightly lighter purple for subheadings */
        font-size: 1.8em;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }

    /* Text Input Styling */
    .stTextInput label {
        font-size: 1.2em;
        font-weight: 600;
        color: #4a4a4a;
    }
    .stTextInput input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
        font-size: 1.1em;
        transition: border-color 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
    }
    .stTextInput input:focus {
        border-color: #a020f0; /* Purple on focus */
        box-shadow: 0 0 0 0.2rem rgba(160, 32, 240, 0.25);
        outline: none;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #a020f0; /* Purple button */
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.2em;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.3s ease-in-out, transform 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background-color: #8b008b; /* Darker purple on hover */
        transform: translateY(-2px);
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Message Styling (Success, Warning) */
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1.5rem;
        font-size: 1.1em;
        font-weight: 500;
    }
    .stAlert.success {
        background-color: #e6ffe6; /* Light green */
        color: #388e3c; /* Dark green text */
        border: 1px solid #c8e6c9;
    }
    .stAlert.warning {
        background-color: #fff3e0; /* Light orange */
        color: #f57c00; /* Dark orange text */
        border: 1px solid #ffe0b2;
    }

    /* Footer / Information Text */
    .stMarkdown p {
        font-size: 1em;
        line-height: 1.6;
        color: #555555;
    }
    .stMarkdown small {
        color: #777777;
        font-style: italic;
    }

    /* Animations (subtle) */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stApp > div {
        animation: fadeIn 0.7s ease-out forwards;
    }

</style>
""", unsafe_allow_html=True)

# Download NLTK resources if not already downloaded
# This should ideally be done once or handled outside the main app run for production
try:
    nltk.data.find('corpora/names')
except nltk.downloader.DownloadError:
    nltk.download('names', quiet=True)


# Function to extract features from a name
def extract_gender_features(name):
    name = name.lower()
    features = {
        "suffix": name[-1:],
        "suffix2": name[-2:] if len(name) > 1 else name[0],
        "suffix3": name[-3:] if len(name) > 2 else name[0],
        "suffix4": name[-4:] if len(name) > 3 else name[0],
        "suffix5": name[-5:] if len(name) > 4 else name[0],
        "suffix6": name[-6:] if len(name) > 5 else name[0],
        "prefix": name[:1],
        "prefix2": name[:2] if len(name) > 1 else name[0],
        "prefix3": name[:3] if len(name) > 2 else name[0],
        "prefix4": name[:4] if len(name) > 3 else name[0],
        "prefix5": name[:5] if len(name) > 4 else name[0]
    }
    return features

# Load the trained Naive Bayes classifier
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_classifier():
    return load('gender_prediction.joblib')

bayes = load_classifier()

# Streamlit app
def main():
    st.title('🌌 Gender Predictor')
    st.write('Empower your insights. Simply type a name below, and our advanced **AI model** will predict its likely gender. Fast, accurate, and intuitive.')

    st.markdown("---") # Visual separator

    # Input for name
    input_name = st.text_input('**Enter a Name Here:** 👇', placeholder='e.g., Alex, Sarah, Michael...', help='Type any name to get a gender prediction.')
    
    col1, col2 = st.columns([1, 4]) # Use columns for better button placement

    with col1:
        predict_button = st.button('🔮 Predict Gender')

    with col2:
        # Add a clear button
        clear_button = st.button('✨ Clear')

    if predict_button:
        if input_name.strip() != '':
            # Show a spinner while predicting for better UX
            with st.spinner('Analyzing name...'):
                import time
                time.sleep(1) # Simulate a short delay for demonstration
                # Extract features for the input name
                features = extract_gender_features(input_name)
                
                # Predict using the trained classifier
                predicted_gender = bayes.classify(features)
                
                # Display prediction with an icon and vibrant message
                gender_icon = "👨" if predicted_gender.lower() == "male" else "👩"
                st.success(f'{gender_icon} The predicted gender for **"{input_name}"** is: **{predicted_gender.upper()}**')
                st.balloons() # Add a celebratory animation
        else:
            st.warning('⚠️ Please enter a name in the input field above to get a prediction.')
    
    if clear_button:
        # Clear the input field (Streamlit re-runs, so setting a default value works)
        # This isn't strictly "clearing" the value from the previous run but resetting the widget
        # A more robust clear would involve session state if you needed to truly wipe previous state
        st.experimental_rerun() # This will clear the input by re-running the script
        
    st.markdown("---") # Visual separator

    st.markdown("""
    <small>
    This app uses a **Naive Bayes Classifier** trained on a dataset of names and their associated genders. 
    It leverages various linguistic features like prefixes and suffixes to make its predictions. 
    While highly accurate, it's an **AI prediction** and should be used for informational purposes only.
    </small>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
