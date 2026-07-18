
import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")

model_path = os.path.join(os.path.dirname(__file__), "decision_tree_model.pkl")
project = joblib.load(model_path)

model = project["model"]
feature_names = project["feature_names"]
accuracy = project.get("accuracy", 0)
precision = project.get("precision", 0)
recall = project.get("recall", 0)
f1 = project.get("f1_score", 0)

st.title("🩺 Breast Cancer Prediction using Decision Tree")
st.write("Enter the measurements below and click **Predict Diagnosis**.")

st.sidebar.header("Model Performance")
st.sidebar.write(f"Accuracy: {accuracy:.2%}")
st.sidebar.write(f"Precision: {precision:.2%}")
st.sidebar.write(f"Recall: {recall:.2%}")
st.sidebar.write(f"F1 Score: {f1:.2%}")

defaults = {
'mean radius':14.1,'mean texture':19.3,'mean perimeter':91.9,'mean area':654.9,'mean smoothness':0.096,
'mean compactness':0.104,'mean concavity':0.089,'mean concave points':0.049,'mean symmetry':0.181,'mean fractal dimension':0.062,
'radius error':0.405,'texture error':1.216,'perimeter error':2.866,'area error':40.34,'smoothness error':0.007,
'compactness error':0.025,'concavity error':0.032,'concave points error':0.012,'symmetry error':0.021,'fractal dimension error':0.0038,
'worst radius':16.27,'worst texture':25.68,'worst perimeter':107.3,'worst area':880.6,'worst smoothness':0.132,
'worst compactness':0.254,'worst concavity':0.273,'worst concave points':0.114,'worst symmetry':0.290,'worst fractal dimension':0.084
}

vals=[]
sections=[("Mean Features",feature_names[:10]),("Error Features",feature_names[10:20]),("Worst Features",feature_names[20:])]
for title,names in sections:
    with st.expander(title, expanded=(title=="Mean Features")):
        c1,c2=st.columns(2)
        for i,n in enumerate(names):
            col=c1 if i%2==0 else c2
            with col:
                vals.append(st.number_input(n.title(), value=float(defaults.get(n,0.0)), format="%.6f"))

if st.button("🔍 Predict Diagnosis", use_container_width=True):
    pred=model.predict(np.array(vals).reshape(1,-1))[0]
    if pred==1:
        st.success("🟢 Prediction: Benign (Non-Cancerous)")
    else:
        st.error("🔴 Prediction: Malignant (Cancerous)")
