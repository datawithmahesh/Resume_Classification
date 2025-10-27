# 🧠 Resume Classification App -  by Mahesh Thapa

import streamlit as st
import pandas as pd
import pickle as pk
import re
import nltk
import io
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from PyPDF2 import PdfReader

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Resume Classification", page_icon="🧾", layout="centered")

# ---------------- HEADER ----------------
st.markdown("""
    <div style='background-color:#2563EB;padding:20px;border-radius:10px;margin-bottom:25px;'>
        <h1 style='color:white;text-align:center;'>🧾 Resume Classification Model</h1>
        <p style='color:white;text-align:center;font-size:17px;'>Predict your resume category using machine learning</p>
    </div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
load_model = pk.load(open("Resume_Data.pickle", "rb"))


# ---------------- NLTK SETUP ----------------
nltk.download('stopwords')
words = stopwords.words("english")
stemmer = PorterStemmer()

# ---------------- INPUT SECTION ----------------
st.subheader("Enter Resume Text or Upload File")

input_mode = st.radio("Choose input method:", ["📝 Type Text", "📄 Upload File"], horizontal=True)

resume_text = ""

if input_mode == "📝 Type Text":
    resume_text = st.text_area(
        "Paste your resume description below:",
        placeholder="e.g. Experienced data analyst skilled in Python, SQL, and data visualization...",
        height=200
    )

else:
    uploaded_file = st.file_uploader("Upload your Resume (PDF or TXT)", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            resume_text = text
        elif uploaded_file.type == "text/plain":
            resume_text = uploaded_file.read().decode("utf-8")
        st.info("✅ File uploaded successfully! You can now click 'Predict'.")

# ---------------- PREDICTION BUTTON ----------------
if st.button("🔍 Predict Category", use_container_width=True):
    if resume_text.strip() == "":
        st.warning("⚠️ Please enter or upload some text first.")
    else:
        # DataFrame creation
        resume_data = {'predict_resume_type': [resume_text]}
        resume_data_df = pd.DataFrame(resume_data)

        # Text cleaning
        resume_data_df['predict_resume_type'] = list(map(
            lambda x: " ".join([i for i in x.lower().split() if i not in words]),
            resume_data_df['predict_resume_type']
        ))
        resume_data_df['predict_resume_type'] = resume_data_df['predict_resume_type'].apply(
            lambda x: " ".join([
                stemmer.stem(i)
                for i in re.sub("[^a-zA-Z]", " ", x).split()
                if i not in words
            ]).lower()
        )

        # Prediction
        result = load_model.predict(resume_data_df['predict_resume_type'])

        # Display results
        st.balloons()
        st.markdown(f"""
            <div style='background-color:#DCFCE7;padding:25px;border-radius:10px;margin-top:20px;text-align:center;'>
                <h3 style='color:#166534;'>🎯 Predicted Resume Category</h3>
                <h2 style='color:#047857;font-size:28px;'>{result[0]}</h2>
                <p style='color:#16A34A;'>Model used: <b>{model_choice}</b></p>
            </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align:center;color:gray;'>
        🚀 Developed by <b>Mahesh Thapa</b> | Streamlit x Machine Learning Project
    </p>
""", unsafe_allow_html=True)







# #Resume_data__categories
# #pip install nltk
# import pandas as pd
# import pickle as pk
# import streamlit as st
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# st.write("Resume classification Model")

# load_model = pk.load(open("Resume_Data.pickle", 'rb'))

# nltk.download('stopwords')
# words = stopwords.words("english")
# stemmer = PorterStemmer()

# resume = st.text_area("Enter your resume:--")

# if st.button("predict"):
#    # df = pd.DataFrame({
#    #    'cleaned':[text]
#    #    })  we can write the code of 3 lines above and continue from else: line  or simply use if else condtion to modify it
#       #  sentiment = input("Enter text = ") which is already given just before the dataframe
#    if resume.strip() == "":
#       st.write("⚠️ Please enter some text")
#    else:
#       # Put text in dataframe
#       resume_data = {'predict_resume_type':[resume]}
#       resume_data_df = pd.DataFrame(resume_data)

#       # Clean text
#       resume_data_df['predict_resume_type'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]), resume_data_df['predict_resume_type']))
#       resume_data_df['predict_resume_type'] = resume_data_df['predict_resume_type'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

#       # Predicticting result 
#       # predict_news_cat = load_model.predict(sentiment_data_df['predict_sentiments'])
#       result = load_model.predict(resume_data_df['predict_resume_type'])

#       # Show result
#       #  st.write("Predicted sentiment category = ",predict_news_cat[0])
#       st.write("Predicted Resume Category = ",result[0])




# st.write("This project is done by Mahesh Thapa")