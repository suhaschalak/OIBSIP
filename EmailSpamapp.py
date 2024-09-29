import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to preprocess the input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app customization with the specified background image
st.markdown(
    """
    <style>
    body {
        background-image: url('https://t4.ftcdn.net/jpg/09/54/87/61/360_F_954876141_hslPXJqCefNdocyASKZaYvgiAuBcPeLo.jpg');
        background-size: cover;
        font-family: 'Arial', sans-serif;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
    }
    h1 {
        color: #4a7a8c;
        font-family: 'Helvetica', sans-serif;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4a7a8c;
        color: white;
        border-radius: 10px;
        font-size: 1rem;
        padding: 0.5rem 1.5rem;
    }
    .stTextArea textarea {
        border: 1px solid #4a7a8c;
        border-radius: 10px;
    }
    .stHeader {
        text-align: center;
        font-size: 1.5rem;
        color: #ffffff;
        background-color: #4a7a8c;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown("<h1 style='text-align: center;'>📧 Email Spam Classifier</h1>", unsafe_allow_html=True)
st.write("This app helps classify an email message as **spam** or **not spam** using machine learning.")

# Input section
input_email = st.text_area("✍️ Enter the email message to analyze", height=200)

if st.button('Predict'):
    # Preprocess the email
    transformed_email = transform_text(input_email)
    # Vectorize the email
    vector_input = tfidf.transform([transformed_email])
    # Predict the result
    result = model.predict(vector_input)[0]
    # Get prediction probabilities (confidence)
    confidence = model.predict_proba(vector_input)[0][result]
    
    # Display result with confidence score
    if result == 1:
        st.markdown("<div class='stHeader'>🚨 This is likely Spam</div>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence*100:.2f}%")
        st.image("https://cdn-icons-png.flaticon.com/512/753/753345.png", width=100)  # Spam icon
    else:
        st.markdown("<div class='stHeader'>✅ This is Not Spam</div>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence*100:.2f}%")
        st.image("https://cdn-icons-png.flaticon.com/512/753/753318.png", width=100)  # Not spam icon
