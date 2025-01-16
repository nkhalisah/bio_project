import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

def anova_analysis(df):
    st.subheader("ANOVA F-Test")

    # Set label and feature columns
    label_column = 'diagnosis'
    feature_columns = [col for col in df.columns if col != label_column]

    X = df[feature_columns]
    y = df[label_column]

    # Perform ANOVA F-test
    f_values, p_values = f_classif(X, y)
    feature_importance = pd.DataFrame({'Feature': X.columns, 'P-Value': p_values})

    # Filter the biomarkers
    biomarkers = ['CA19-9', 'creatinine', 'LYVE1', 'REG1B', 'TFF1', 'REG1A']
    biomarker_importance = feature_importance[feature_importance['Feature'].isin(biomarkers)]

    # Top 5 biomarkers based on ANOVA
    top_5_anova_biomarkers = biomarker_importance.nsmallest(5, 'P-Value')

    # Format p-values in scientific notation (e.g., 1e-5)
    top_5_anova_biomarkers['P-Value'] = top_5_anova_biomarkers['P-Value'].apply(lambda x: f'{x:.2e}')

    st.write("Top 5 Biomarkers based on ANOVA:")
    st.write(top_5_anova_biomarkers)

    X_selected = X[top_5_anova_biomarkers["Feature"]]

    return X_selected, y

def rfe_analysis(df):
    st.subheader("Recursive Feature Elimination (RFE)")

    # Set label and feature columns
    label_column = 'diagnosis'
    feature_columns = [col for col in df.columns if col != label_column]

    X = df[feature_columns]
    y = df[label_column]

    # Scale the features for better SVM performance
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform RFE
    svm_model = SVC(kernel="linear")
    rfe = RFE(estimator=svm_model, n_features_to_select=5)
    rfe.fit(X_scaled, y)

    # Get the selected features
    selected_features = X.columns[rfe.support_]

    # Filter for biomarkers
    biomarkers = ['CA19-9', 'creatinine', 'LYVE1', 'REG1B', 'TFF1', 'REG1A']
    biomarker_rfe_importance = pd.DataFrame({'Feature': X.columns, 'Ranking': rfe.ranking_})
    biomarker_rfe_importance = biomarker_rfe_importance[biomarker_rfe_importance['Feature'].isin(biomarkers)]

    top_5_rfe_biomarkers = biomarker_rfe_importance.sort_values('Ranking').head(5)
    st.write("Top 5 Biomarkers based on RFE:")
    st.write(top_5_rfe_biomarkers)

    X_selected = X[selected_features]

    return X_selected, y

def evaluate_model(X_selected, y, test_size=0.2, method="ANOVA"):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42)

    # Train the SVM model
    svm_model = SVC(kernel="linear")
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.write(pd.DataFrame(report).transpose())

    # Print confusion matrix based on the method (ANOVA or RFE)
    if method == "ANOVA":
        st.write("Confusion Matrix (ANOVA):")
        st.write(pd.DataFrame(cm))
    elif method == "RFE":
        # For RFE, print the confusion matrix with y_test_rfe and y_pred_rfe
        st.write("Confusion Matrix (RFE):")
        y_test_rfe, y_pred_rfe = y_test, y_pred  # Rename for clarity
        st.write(pd.DataFrame(confusion_matrix(y_test_rfe, y_pred_rfe)))

def main():
    st.title("Feature Selection with ANOVA and RFE")

    # Use a fixed file path
    file_path = 'C:/Processed_Data.csv'
    try:
        df = load_data(file_path)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        # Data validation
        if df.isnull().any().any():
            st.error("The dataset contains missing values. Please handle them before proceeding.")
            return

        if 'diagnosis' not in df.columns:
            st.error("'diagnosis' column not found in the dataset.")
            return

        # Page navigation
        page = st.sidebar.radio("Select Analysis Method", ["ANOVA", "RFE"])

        if page == "ANOVA":
            X_selected, y = anova_analysis(df)
            evaluate_model(X_selected, y, method="ANOVA")
        elif page == "RFE":
            X_selected, y = rfe_analysis(df)
            evaluate_model(X_selected, y, method="RFE")

    except FileNotFoundError:
        st.error("The specified file path is not valid. Please check the path.")

if __name__ == "__main__":
    main()