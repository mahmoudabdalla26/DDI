import streamlit as st
import joblib
import numpy as np

# Load model and matrices
clf = joblib.load('DDI_rf_model.pkl')
u = np.load('u_matrix.npy')
vt = np.load('vt_matrix.npy')
drug_index = np.load('drug_index.npy', allow_pickle=True).item()

# Mapping numeric severity predictions to user-friendly messages
severity_messages = {
    3: "Major interaction: These drugs can have serious side effects when taken together. Consult a healthcare provider immediately.",
    2: "Moderate interaction: These drugs may interact and cause noticeable effects. Use caution and seek medical advice if necessary.",
    1: "Minor interaction: The interaction between these drugs is minimal, but you may still experience mild effects.",
    0: "No known interaction: There are no reported interactions between these drugs."
}

# Function to preprocess input drugs
def preprocess_input(drug_a, drug_b):
    # Convert drug names to indices using `drug_index`
    idx1 = drug_index.get(drug_a)
    idx2 = drug_index.get(drug_b)
    if idx1 is None or idx2 is None:
        return None
    # Create feature vector by combining corresponding u and vt vectors
    features = np.concatenate([u[idx1], vt[idx2]])
    return features

# Streamlit application layout
st.title("Drug Interaction Prediction")
st.write("Enter two drugs to predict the interaction severity between them.")

# User input for drugs
drug_a = st.text_input("Enter Drug A")
drug_b = st.text_input("Enter Drug B")

# Prediction button
if st.button("Predict Interaction"):
    if drug_a and drug_b:
        # Preprocess the input
        features = preprocess_input(drug_a, drug_b)
        if features is None:
            st.error("One or both drugs not found in the index.")
        else:
            # Make prediction
            severity_prediction = clf.predict([features])[0]
            severity_message = severity_messages.get(severity_prediction, "Unknown interaction level.")
            st.write(severity_message)
    else:
        st.error("Please enter both Drug A and Drug B.")
