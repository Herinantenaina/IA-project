import streamlit as st
import pandas as pd
import joblib as jb

model = jb.load('model.pkl')

st.title('Prédiction du Churn Client')
st.write("Veuillez entrer les informations du client: ")

genre = st.selectbox("Genre", ["Male", "Female"])
senior = st.selectbox("Client senior", [0, 1])
partenaire = st.selectbox("Partenaire", ["Yes", "No"])
dependants = st.selectbox("Personnes à charge", ["Yes", "No"])
anciennete = st.slider("Ancienneté (mois)", 0, 72, 12)
telephone = st.selectbox("Service téléphonique", ["Yes", "No"])
facturation = st.selectbox("Facturation électronique", ["Yes", "No"])
charges_mensuelles = st.number_input("Charges mensuelles", 0.0, 200.0, 50.0)
charges_totales = st.number_input("Charges totales", 0.0, 10000.0, 500.0)
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Support technique", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Sauvegarde en ligne", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Protection des appareils", ["Yes", "No", "No internet service"])
MultipleLines = st.selectbox("Lignes multiples", ["Yes", "No", "No phone service"])
OnlineSecurity = st.selectbox("Sécurité en ligne", ["Yes", "No", "No internet service"])
contrat = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
paiement = st.selectbox("Méthode de paiement", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

data = pd.DataFrame({
    "gender": [genre],
    "SeniorCitizen": [senior],
    "Partner": [partenaire],
    "Dependents": [dependants],
    "tenure": [anciennete],
    "PhoneService": [telephone],
    "PaperlessBilling": [facturation],
    "MonthlyCharges": [charges_mensuelles],
    "TotalCharges": [charges_totales],
    "PaymentMethod": [paiement],
    "Contract": [contrat],
    "InternetService": [internet],
    "StreamingMovies": [StreamingMovies],
    "TechSupport": [TechSupport],
    "OnlineBackup": [OnlineBackup],
    "StreamingTV": [StreamingTV],
    "DeviceProtection": [DeviceProtection],
    "MultipleLines": [MultipleLines],
    "OnlineSecurity": [OnlineSecurity],
})

if  st.button('Prédire'):
    prediction= model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.error(f' Le client risque de partir ({proba:.2%})')
    else :
        st.success(f'Le client restera ({1 - proba:.2%})')