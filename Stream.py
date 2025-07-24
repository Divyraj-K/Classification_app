import streamlit as st
import pandas as pd
import numpy as np
#import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.write("hello")

import warnings
warnings.filterwarnings("ignore")

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

# loading the data from sklearn
@st.cache_data
def get_data():
    breast_cancer_dataset = load_breast_cancer()
    data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
    return data_frame
data_frame = get_data()

show_data = st.checkbox("Show Data")
if show_data:
    st.dataframe(data_frame.head())
    st.write(data_frame.shape)

#---------------------------------------------------------------------------------------------------------------------------------
#Train Mode
st.header("🤖 Train Classification Model")
for_train_model = st.radio("Train Model","Use model")
if for_train_model=="Train Model":
    columns11 = list(data_frame.columns)
    target_col = st.selectbox("Select Target",columns11)
    #Encoding


    try:
        X = data_frame.drop(target_col, axis=1)
        y = data_frame[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except:
        st.error("Something is Wrong")

    ## Scale the data
    Scale = st.radio("Scale the data",("No Need","Standard Scaler","Min-Max Scaler"))
    if Scale=="Standard Scaler":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif Scale=="Min-Max Scaler":
        scaler = minmax_scale()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        pass
    selected_model = st.selectbox("Select Model For Train",("Logistic Regression", "Random Forest", "SVM", "KNN", "Decision Tree"))
    if selected_model == "Logistic Regression":
        C = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C)

    elif selected_model == "Random Forest":
        n_estimators = st.slider("No. of Trees", 10, 200, 100)
        max_depth = st.slider("Max Depth", 2, 20, 5)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    elif selected_model == "SVM":
        C = st.slider("Regularization Parameter (C)", 0.01, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
        model = SVC(C=C, kernel=kernel, probability=True)

    elif selected_model == "KNN":
        n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

    elif selected_model == "Decision Tree":
        max_depth = st.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth)

    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

        if st.button("Save Model"):
            joblib.dump(model, f"{selected_model.replace(' ', '_').lower()}_model.pkl")
            st.success("Model saved successfully!")




#---------------------------------------------------------------------------------------------------------------------------------
#Use Model
