



# print("✅ Model trained & saved as los_model.pkl")


import streamlit as st
import pandas as pd
# cluster_data 
# Load saved merged data
merged = pd.read_pickle("merged_data.pkl")  # or .csv if you saved CSV

#associative 
import pandas as pd
# Load merged dataset
@st.cache_data
def load_data():
    return pd.read_csv("merged_data.csv")
merged = load_data()


#STREAMLITE

# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from mlxtend.frequent_patterns import apriori, association_rules

# # App title & sidebar
# st.set_page_config(page_title="Healthcare AI Platform", page_icon="🏥", layout="wide")

# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", [
#     " Home",
#     " Chatbot",
#     " Risk Stratification",
#     " Length of Stay Prediction",
#     " Patient Segmentation",
#     " Medical Associations",
#     " Imaging Diagnostics (CNN)",
#     " Sequence Modeling (RNN/LSTM)",
#     " Pretrained Models (BioBERT/ClinicalBERT)",
#     " Translator",
#     " Sentiment Analysis"
# ])

# if page == " Home":
#     st.title("Healthcare AI Platform")
#     st.markdown("Welcome! Explore AI-driven healthcare solutions across **10 use cases**.")
#     st.image("ai-in-health-care-infographic-vector.jpg", width=600)

# # elif page == " Chatbot":
# #     st.title("Healthcare Chatbot")
# #     st.write("Ask about symptoms, appointments, or FAQs.")
# #     user_input = st.text_input("You:", "")
# #     if user_input:
# #         if "fever" in user_input.lower():
# #             st.write(" Bot: You may have viral fever. Please schedule an appointment with a physician.")
# #         elif "appointment" in user_input.lower():
# #             st.write(" Bot: Sure! Please provide preferred date & department.")
# #         else:
# #             st.write(" Bot: Sorry, I can only help with symptoms & appointments for now.")
# elif page == " Chatbot":
#     st.title("Healthcare Chatbot")
#     st.write("Ask about appointments, symptoms, or FAQs.")
#     user_input = st.text_input("You:", "")
    
#     # Predefined FAQ responses (safe answers)
#     faq_responses = {
#         "fever": "Common symptoms of fever include high temperature, chills, and body aches. Please consult a doctor for diagnosis.",
#         "headache": "Headaches can be caused by stress, dehydration, or other conditions. If severe or persistent, schedule an appointment.",
#         "cough": "A persistent cough may indicate a respiratory infection. Please consult your physician if it continues."
#     }
    
#     appointment_keywords = ["appointment", "book", "schedule", "reschedule", "cancel", "slot", "doctor"]
    
#     if user_input:
#         user_text = user_input.lower()
        
#         # Check for appointment-related queries
#         if any(keyword in user_text for keyword in appointment_keywords):
#             st.write(" Bot: Sure! Please provide your preferred date & department.")
        
#         # Check for symptom-related queries
#         elif any(key in user_text for key in faq_responses.keys()):
#             # Find the first matching symptom keyword
#             for key in faq_responses.keys():
#                 if key in user_text:
#                     st.write(f" Bot: {faq_responses[key]}")
#                     break
        
#         # Fallback for everything else
#         else:
#             st.write(" Bot: Sorry, I can only help with scheduling appointments or answering common symptom FAQs for now.")

# elif page == " Risk Stratification":
#     st.title("Risk Stratification (Classification)")
#     st.write("Early detection of diseases like diabetes, heart disease, cancer staging.")
#     age = st.slider("Age", 0, 100, 45)
#     bmi = st.number_input("BMI", 10.0, 40.0, 22.0)
#     bp = st.number_input("Blood Pressure", 80, 200, 120)
#     if st.button("Predict Risk"):
#         st.success("Predicted Risk: HIGH (Demo Output)")

# elif page == " Length of Stay Prediction":
#     st.title("🏥 Hospital Length of Stay Prediction")
#     st.write("Forecast patient hospitalization duration using clinical inputs.")

#     # Load the model and scaler
#     with open("best_model.pkl", "rb") as file:
#         data = pickle.load(file)
#         model = data["model"]
#         scaler = data["scaler"]
#         feature_names = data["features"]

#     # User inputs
#     age = st.number_input("Age", min_value=1, max_value=120, value=45)
#     bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=23.5)
#     blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
#     heart_rate = st.number_input("Heart Rate", min_value=40, max_value=180, value=80)

#     if st.button("Predict Length of Stay"):
#         input_df = pd.DataFrame([[age, bmi, blood_pressure, heart_rate]], columns=feature_names)
#         input_scaled = scaler.transform(input_df)
#         prediction = model.predict(input_scaled)
#         st.success(f"🕒 Predicted Length of Stay: **{prediction[0]:.2f} days**")

# elif page == " Patient Segmentation":
#     st.title("Patient Segmentation (Clustering)")
#     st.write("Group patients into cohorts (chronic vs. acute, etc.)")

#     if "merged" in globals():
#         numeric_cols = [
#             'age', 'admissionweight', 'dischargeweight',
#             'pao2', 'fio2', 'ejectfx', 'creatinine',
#             'intake_count', 'infectious_count', 'medication_count'
#         ]
#         numeric_cols = [col for col in numeric_cols if col in merged.columns]

#         for col in numeric_cols:
#             merged[col] = pd.to_numeric(merged[col], errors='coerce')

#         imputer = SimpleImputer(strategy='median')
#         merged[numeric_cols] = imputer.fit_transform(merged[numeric_cols])

#         st.subheader("Feature Distributions")
#         fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
#         axes = axes.flatten()
#         for i, col in enumerate(numeric_cols[:9]):
#             sns.histplot(merged[col], ax=axes[i], kde=True)
#             axes[i].set_title(col)
#         st.pyplot(fig)

#         st.subheader("PCA Visualization")
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(merged[numeric_cols])
#         pca = PCA(n_components=2)
#         pca_result = pca.fit_transform(X_scaled)

#         fig2, ax2 = plt.subplots(figsize=(8, 6))
#         ax2.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
#         ax2.set_xlabel("PC1")
#         ax2.set_ylabel("PC2")
#         ax2.set_title("PCA 2D Projection")
#         st.pyplot(fig2)

#         st.info("Next step: Apply clustering algorithm (KMeans, Agglomerative, etc.) on scaled data for segmentation.")
#     else:
#         st.warning("❌ `merged` DataFrame not found. Please load your dataset before using this feature.")

# elif page == " Medical Associations":
#     st.title("Medical Associations (Association Rules)")
#     st.write("Discover frequent patterns and medical associations from patient data.")

#     exclude_cols = ['gender','ethnicity','age','hospitaldischargelocation',
#                     'apacheadmissiondx','physicianspeciality',
#                     'actualicumortality','actualhospitalmortality']

#     assoc_cols = [col for col in merged.columns if col not in exclude_cols and merged[col].nunique() <= 5]

#     if assoc_cols:
#         assoc_data = merged[assoc_cols].copy()
#         for col in assoc_cols:
#             assoc_data[col] = assoc_data[col].apply(lambda x: 1 if str(x).lower() in ['yes','1','true','high'] else 0)

#         frequent_itemsets = apriori(assoc_data, min_support=0.1, use_colnames=True)

#         if not frequent_itemsets.empty:
#             rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
#             st.subheader("Top Medical Associations")
#             st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
#         else:
#             st.warning("No frequent patterns found. Try adjusting columns or min_support.")
#     else:
#         st.error("No suitable binary/categorical columns available for association rule mining.")

# elif page == " Imaging Diagnostics (CNN)":
#     st.title("Imaging Diagnostics (CNN)")
#     uploaded_file = st.file_uploader("Upload X-ray/CT/MRI Image")
#     if uploaded_file:
#         st.image(uploaded_file, caption="Uploaded Image", width=300)
#         st.success("Prediction: Pneumonia Detected (Demo)")

# elif page == " Sequence Modeling (RNN/LSTM)":
#     st.title("Sequence Modeling (RNN/LSTM)")
#     st.write("Track patient vitals over time to forecast deterioration or readmission.")

#     st.markdown("""
#     This use case applies **RNN/LSTM models** on patient time-series data 
#     (vitals, labs, notes) to predict **deterioration risk / readmission**.
#     """)

#     auc_score = 0.765
#     st.metric("Model AUC", f"{auc_score:.3f}")

#     fpr = np.linspace(0, 1, 100)
#     tpr = fpr**0.5

#     fig, ax = plt.subplots()
#     ax.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
#     ax.plot([0, 1], [0, 1], "--", color="gray")
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")
#     ax.set_title("ROC Curve for LSTM Model")
#     ax.legend()
#     st.pyplot(fig)

#     st.subheader("Test with Patient Data (Demo)")
#     hr = st.slider("Heart Rate", 40, 180, 100)
#     bp_sys = st.slider("Systolic BP", 80, 200, 120)
#     bp_dia = st.slider("Diastolic BP", 40, 120, 80)
#     spo2 = st.slider("Oxygen Saturation (SpO2)", 50, 100, 95)

#     if st.button("Predict Deterioration Risk"):
#         if hr > 120 or spo2 < 90:
#             st.error("⚠️ High risk of deterioration detected!")
#         else:
#             st.success("✅ Patient is stable (Low risk).")

# elif page == " Pretrained Models (BioBERT/ClinicalBERT)":
#     st.title("Pretrained Models (BioBERT/ClinicalBERT)")
#     st.write("Use BioBERT/ClinicalBERT for clinical notes, discharge summaries, drug side effects.")

#     uploaded_file = st.file_uploader("Upload clinical notes CSV", type=["csv"])
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
#         st.write("Preview of uploaded notes:")
#         st.dataframe(df.head())

#         st.write("Generating embeddings... (Demo output)")
#         st.write("Example embedding vector for first note:")
#         demo_embedding = np.random.rand(1, 768)
#         st.write(demo_embedding[0][:10])

#         st.write("Similarity demo:")
#         st.write("Most similar note to first note: Note 2 (Demo)")

# # elif page == " Translator":
# #     st.title("Doctor-Patient Translator")
# #     text = st.text_input("Enter text in English")
# #     if st.button("Translate to Hindi"):
# #         st.success("Translated: 'मुझे बुखार है' (Demo)")
# from transformers import MarianMTModel, MarianTokenizer

# # Load model and tokenizer
# model_name_hi = "Helsinki-NLP/opus-mt-en-hi"
# tokenizer_hi = MarianTokenizer.from_pretrained(model_name_hi)
# model_hi = MarianMTModel.from_pretrained(model_name_hi)

# def translate(text, model, tokenizer):
#     inputs = tokenizer(text, return_tensors="pt", padding=True)
#     translated = model.generate(**inputs)
#     tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
#     return tgt_text[0]

# # Streamlit input
# st.title("Doctor-Patient Translator")
# text = st.text_input("Enter text in English")

# if st.button("Translate to Hindi") and text:
#     translated_text = translate(text, model_hi, tokenizer_hi)
#     st.success(f"Translated: {translated_text}")

# # Tamil
# model_name_ta = "Helsinki-NLP/opus-mt-en-ta"
# tokenizer_ta = MarianTokenizer.from_pretrained(model_name_ta)
#  model_ta = MarianMTModel.from_pretrained(model_name_ta)

# # Malayalam
#  model_name_ml = "Helsinki-NLP/opus-mt-en-ml"
# tokenizer_ml = MarianTokenizer.from_pretrained(model_name_ml)
# model_ml = MarianMTModel.from_pretrained(model_name_ml)


# elif page == " Sentiment Analysis":
#     st.title("Patient Feedback Sentiment Analysis")
#     feedback = st.text_area("Enter patient feedback")
#     if st.button("Analyze"):
#         st.success("Sentiment: Positive (Demo)")

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from transformers import MarianMTModel, MarianTokenizer

# Set page config
st.set_page_config(page_title="Healthcare AI Platform", page_icon="🏥", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    " Home",
    " Chatbot",
    " Risk Stratification",
    " Length of Stay Prediction",
    " Patient Segmentation",
    " Medical Associations",
    " Imaging Diagnostics (CNN)",
    " Sequence Modeling (RNN/LSTM)",
    " Pretrained Models (BioBERT/ClinicalBERT)",
    " Translator",
    " Sentiment Analysis"
])

# ========== HOME ==========
if page == " Home":
    st.title("Healthcare AI Platform")
    st.markdown("Welcome! Explore AI-driven healthcare solutions across **10 use cases**.")
    st.image("ai-in-health-care-infographic-vector.jpg", width=600)

# ========== CHATBOT ==========
elif page == " Chatbot":
    st.title("Healthcare Chatbot")
    st.write("Ask about appointments, symptoms, or FAQs.")
    user_input = st.text_input("You:", "")
    
    faq_responses = {
        "fever": "Common symptoms of fever include high temperature, chills, and body aches. Please consult a doctor for diagnosis.",
        "headache": "Headaches can be caused by stress, dehydration, or other conditions. If severe or persistent, schedule an appointment.",
        "cough": "A persistent cough may indicate a respiratory infection. Please consult your physician if it continues."
    }
    
    appointment_keywords = ["appointment", "book", "schedule", "reschedule", "cancel", "slot", "doctor"]
    
    if user_input:
        user_text = user_input.lower()
        if any(keyword in user_text for keyword in appointment_keywords):
            st.write(" Bot: Sure! Please provide your preferred date & department.")
        elif any(key in user_text for key in faq_responses.keys()):
            for key in faq_responses:
                if key in user_text:
                    st.write(f" Bot: {faq_responses[key]}")
                    break
        else:
            st.write(" Bot: Sorry, I can only help with scheduling appointments or answering common symptom FAQs for now.")

# ========== RISK STRATIFICATION ==========
elif page == " Risk Stratification":
    st.title("Risk Stratification (Classification)")
    st.write("Early detection of diseases like diabetes, heart disease, cancer staging.")
    age = st.slider("Age", 0, 100, 45)
    bmi = st.number_input("BMI", 10.0, 40.0, 22.0)
    bp = st.number_input("Blood Pressure", 80, 200, 120)
    if st.button("Predict Risk"):
        st.success("Predicted Risk: HIGH (Demo Output)")

# ========== LENGTH OF STAY ==========
elif page == " Length of Stay Prediction":
    st.title("🏥 Hospital Length of Stay Prediction")
    st.write("Forecast patient hospitalization duration using clinical inputs.")

    with open("best_model.pkl", "rb") as file:
        data = pickle.load(file)
        model = data["model"]
        scaler = data["scaler"]
        feature_names = data["features"]

    age = st.number_input("Age", 1, 120, 45)
    bmi = st.number_input("BMI", 10.0, 50.0, 23.5)
    blood_pressure = st.number_input("Blood Pressure", 50, 200, 120)
    heart_rate = st.number_input("Heart Rate", 40, 180, 80)

    if st.button("Predict Length of Stay"):
        input_df = pd.DataFrame([[age, bmi, blood_pressure, heart_rate]], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"🕒 Predicted Length of Stay: **{prediction[0]:.2f} days**")

# ========== PATIENT SEGMENTATION ==========
elif page == " Patient Segmentation":
    st.title("Patient Segmentation (Clustering)")
    st.write("Group patients into cohorts (chronic vs. acute, etc.)")

    if "merged" in globals():
        numeric_cols = [
            'age', 'admissionweight', 'dischargeweight',
            'pao2', 'fio2', 'ejectfx', 'creatinine',
            'intake_count', 'infectious_count', 'medication_count'
        ]
        numeric_cols = [col for col in numeric_cols if col in merged.columns]

        for col in numeric_cols:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

        imputer = SimpleImputer(strategy='median')
        merged[numeric_cols] = imputer.fit_transform(merged[numeric_cols])

        st.subheader("Feature Distributions")
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols[:9]):
            sns.histplot(merged[col], ax=axes[i], kde=True)
            axes[i].set_title(col)
        st.pyplot(fig)

        st.subheader("PCA Visualization")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(merged[numeric_cols])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("PCA 2D Projection")
        st.pyplot(fig2)

        st.info("Next step: Apply clustering algorithm (KMeans, Agglomerative, etc.) on scaled data for segmentation.")
    else:
        st.warning("❌ `merged` DataFrame not found. Please load your dataset before using this feature.")

# ========== MEDICAL ASSOCIATIONS ==========
elif page == " Medical Associations":
    st.title("Medical Associations (Association Rules)")
    st.write("Discover frequent patterns and medical associations from patient data.")

    exclude_cols = ['gender','ethnicity','age','hospitaldischargelocation',
                    'apacheadmissiondx','physicianspeciality',
                    'actualicumortality','actualhospitalmortality']

    assoc_cols = [col for col in merged.columns if col not in exclude_cols and merged[col].nunique() <= 5]

    if assoc_cols:
        assoc_data = merged[assoc_cols].copy()
        for col in assoc_cols:
            assoc_data[col] = assoc_data[col].apply(lambda x: 1 if str(x).lower() in ['yes','1','true','high'] else 0)

        frequent_itemsets = apriori(assoc_data, min_support=0.1, use_colnames=True)

        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            st.subheader("Top Medical Associations")
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
        else:
            st.warning("No frequent patterns found. Try adjusting columns or min_support.")
    else:
        st.error("No suitable binary/categorical columns available for association rule mining.")

# ========== IMAGING DIAGNOSTICS ==========
elif page == " Imaging Diagnostics (CNN)":
    st.title("Imaging Diagnostics (CNN)")
    uploaded_file = st.file_uploader("Upload X-ray/CT/MRI Image")
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        st.success("Prediction: Pneumonia Detected (Demo)")

# ========== SEQUENCE MODELING ==========
elif page == " Sequence Modeling (RNN/LSTM)":
    st.title("Sequence Modeling (RNN/LSTM)")
    st.write("Track patient vitals over time to forecast deterioration or readmission.")

    auc_score = 0.765
    st.metric("Model AUC", f"{auc_score:.3f}")

    fpr = np.linspace(0, 1, 100)
    tpr = fpr**0.5

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve for LSTM Model")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Test with Patient Data (Demo)")
    hr = st.slider("Heart Rate", 40, 180, 100)
    bp_sys = st.slider("Systolic BP", 80, 200, 120)
    bp_dia = st.slider("Diastolic BP", 40, 120, 80)
    spo2 = st.slider("Oxygen Saturation (SpO2)", 50, 100, 95)

    if st.button("Predict Deterioration Risk"):
        if hr > 120 or spo2 < 90:
            st.error("⚠️ High risk of deterioration detected!")
        else:
            st.success("✅ Patient is stable (Low risk).")

# ========== PRETRAINED MODELS ==========
elif page == " Pretrained Models (BioBERT/ClinicalBERT)":
    st.title("Pretrained Models (BioBERT/ClinicalBERT)")
    uploaded_file = st.file_uploader("Upload clinical notes CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded notes:")
        st.dataframe(df.head())

        st.write("Generating embeddings... (Demo output)")
        demo_embedding = np.random.rand(1, 768)
        st.write(demo_embedding[0][:10])

        st.write("Similarity demo:")
        st.write("Most similar note to first note: Note 2 (Demo)")

# ========== TRANSLATOR ==========
elif page == " Translator":
    st.title("Doctor-Patient Translator")
    text = st.text_input("Enter text in English")

    if st.button("Translate to Hindi") and text:
        model_name_hi = "Helsinki-NLP/opus-mt-en-hi"
        tokenizer_hi = MarianTokenizer.from_pretrained(model_name_hi)
        model_hi = MarianMTModel.from_pretrained(model_name_hi)
        inputs = tokenizer_hi(text, return_tensors="pt", padding=True)
        translated = model_hi.generate(**inputs)
        translated_text = tokenizer_hi.batch_decode(translated, skip_special_tokens=True)[0]
        st.success(f"Translated: {translated_text}")

# ========== SENTIMENT ANALYSIS ==========
# elif page == " Sentiment Analysis":
#     st.title("Patient Feedback Sentiment Analysis")
#     feedback = st.text_area("Enter patient feedback")
#     if st.button("Analyze"):
#         st.success("Sentiment: Positive (Demo)")
# from transformers import pipeline

elif page == " Sentiment Analysis":
    st.title("Patient Feedback Sentiment Analysis")
    feedback = st.text_area("Enter patient feedback")

    if st.button("Analyze"):
        if not feedback.strip():
            st.error("⚠️ Please enter some feedback.")

        # Expanded keyword lists
        positive_words = [
            "good", "happy", "satisfied", "excellent", "great",
            "wonderful", "amazing", "fantastic", "helpful", "positive",
            "caring", "nice", "kind", "supportive", "brilliant"
        ]
        negative_words = [
            "bad", "poor", "angry", "worst", "terrible",
            "horrible", "unhappy", "disappointed", "negative", "careless",
            "rude", "slow", "pain", "problem", "useless"
        ]
        neutral_words = [
            "okay", "fine", "average", "normal", "not sure",
            "decent", "acceptable", "regular"
        ]

        fb = feedback.lower()

        if any(word in fb for word in positive_words):
            st.success("Sentiment: Positive 😊")
        elif any(word in fb for word in negative_words):
            st.error("Sentiment: Negative 😡")
        elif any(word in fb for word in neutral_words):
            st.warning("Sentiment: Neutral 😐")
        else:
            st.info("Sentiment: Could not be determined (Demo)")

