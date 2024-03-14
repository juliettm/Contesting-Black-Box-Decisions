import gradio as gr
import numpy as np
import joblib
from sklearn.tree import _tree
import pandas as pd


# Sample model function for loan prediction
# Load the Decision Tree model
model_path = 'best_decision_tree_model.joblib'
tree_model = joblib.load(model_path)
# Get the decision path for the input
feature_names = ["Sex", "Single", "Unemployed", "Age", "Credit", "LoanDuration", "PurposeOfLoan", "InstallmentRate", "Housing"]

def get_decision_path_details(tree_model, single_instance, feature_names):
    decision_path = tree_model.decision_path(single_instance).indices
    decision_features = []
    decision_thresholds = []
    decision_directions = []

    tree_ = tree_model.tree_

    for node_index in decision_path:
        if tree_.feature[node_index] != _tree.TREE_UNDEFINED:
            feature_name = feature_names[tree_.feature[node_index]]
            threshold = tree_.threshold[node_index]
            decision = single_instance.iloc[0, tree_.feature[node_index]] <= threshold
            direction = "<=" if decision else ">"

            decision_features.append(feature_name)
            decision_thresholds.append(threshold)
            decision_directions.append(f"{feature_name} {direction} {threshold:.2f}")

    decisions = " -> ".join(decision_directions)
    return decisions
def loan_prediction(Sex, CivilStatus, EmploymentStatus, age, credit_amount, duration, PurposeOfLoan, installmentRate, housing):

    # Placeholder model, replace this with your actual model
    model = tree_model
    # Sample input transformation, replace this with your actual preprocessing
    loan_purposes = {"Business":1, "Education":2, "Electronics":3, "Furniture":4, "HomeAppliances":5, "NewCar":6, "Other":7, "Repairs":8, "Retraining":9, "UsedCar":10}
    gender = {"Male": 1, "Female": 0}
    housing_type = {"Own": 1, "Rent": 2, "Other": 3}
    civil_status = {"Single": 1, "Married": 0}
    employment_status = {"Unemployed": 1, "Employed": 0}

    input_data = np.array([[gender[Sex],
                            civil_status[CivilStatus],
                            employment_status[EmploymentStatus],
                            age,
                            credit_amount,
                            duration,
                            loan_purposes[PurposeOfLoan],
                            installmentRate,
                            housing_type[housing]
                            ]])
    # Sample prediction
    prediction = model.predict(input_data)
    # Sample output, replace this with your actual output
    output = "Good Customer" if prediction == 1 else "Bad Customer"

    # Prepare data for prediction
    input_df = pd.DataFrame([input_data[0]],
                            columns=["Sex", "Single", "Unemployed", "Age", "Credit", "LoanDuration",
                                     "PurposeOfLoan", "InstallmentRate", "Housing"])


    decision_path_text = get_decision_path_details(tree_model, input_df, feature_names)

    results = "The client is classified as: " + output + ". The decision path is: " + decision_path_text

    return (output, decision_path_text)

# Define inputs for the Gradio interface
inputs = [
    gr.Radio(["Male", "Female"], label="Sex"),
    gr.Radio(label="Civil Status", choices=["Single", "Married"]),
    gr.Radio(label="Employment Status", choices=["Unemployed", "Employed"]),
    gr.Number(label="Age", minimum=15, maximum=100),
    gr.Number(label="Credit Amount", minimum=1),
    gr.Number(label="Loan Duration (months)", minimum=1),
    gr.Dropdown(["Business", "Education", "Electronics", "Furniture", "HomeAppliances", "NewCar", "Other", "Repairs", "Retraining", "UsedCar"], label="Purpose of Loan"),
    gr.Slider(minimum=1, maximum=4, label="Installment Rate"),
    gr.Radio(["Own", "Rent", "Other"], label="Housing"),
]


# Define the output component for the Gradio interface
output = [gr.Textbox(label="Client Classification"), gr.Textbox(label="Decision Path")]


# Create the Gradio interface
gr.Interface(fn=loan_prediction, inputs=inputs, outputs=output, title="Loan Prediction App").launch(share=True)

