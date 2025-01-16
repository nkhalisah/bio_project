# bio_project

Feature Selection with ANOVA and RFE
This Streamlit application performs feature selection and classification for datasets using ANOVA F-Test and Recursive Feature Elimination (RFE) methods. It is specifically designed for datasets involving biomarker analysis. The app provides an intuitive interface to:

Analyze Feature Importance:

Identify top biomarkers based on their statistical significance using ANOVA F-Test.
Perform Recursive Feature Elimination (RFE) to rank features using Support Vector Machines (SVM).
Train and Evaluate Models:

Select the most important features and train an SVM classifier.
Evaluate the model's performance using metrics such as accuracy, classification report, and confusion matrix.
Interactive Interface:

Load a dataset for analysis.
Handle missing values or column validation errors dynamically.
Visualize the results in an easy-to-understand format.

Requirements
Python 3.7+
Streamlit
Scikit-learn
Pandas
Numpy

How to Run
Install the required libraries:
pip install streamlit scikit-learn pandas numpy

Run the app:
streamlit run your_script_name.py
Load the dataset (ensure the dataset file is in CSV format with a diagnosis column).









