from flask import Flask, request, render_template_string
import pandas as pd
import joblib
from sklearn.tree import _tree

app = Flask(__name__)

# Load the Decision Tree model
model_path = '/Users/juls/Documents/Repositories/Contesting-Black-Box-Decisions/interpretable_model/best_decision_tree_model.joblib'
model = joblib.load(model_path)


# Function to get decision path details for a specific instance
def get_decision_path_details(model, single_instance, feature_names):
    decision_path = model.decision_path(single_instance).indices
    decision_features = []
    decision_thresholds = []
    decision_directions = []

    tree_ = model.tree_

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


# HTML template for the web form
HTML = '''
    <html>
        <body>
            <h2>Loan Application Classifier with Decision Path</h2>
            <form method="post">
                Sex (1.0 for male, 0.0 for female): <input type="text" name="sex"><br>
                Single (1 for yes, 0 for no): <input type="text" name="single"><br>
                Unemployed (1 for yes, 0 for no): <input type="text" name="unemployed"><br>
                Age: <input type="text" name="age"><br>
                Credit amount: <input type="text" name="credit"><br>
                Loan Duration (in months): <input type="text" name="loanDuration"><br>
                Purpose Of Loan (numeric code): <input type="text" name="purposeOfLoan"><br>
                Installment Rate: <input type="text" name="installmentRate"><br>
                Housing (1 for rent, 2 for own, 3 for free): <input type="text" name="housing"><br>
                <input type="submit" value="Submit">
            </form>
            {{ prediction }}<br>
            {{ decision_path }}
        </body>
    </html>
'''


@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction_text = ''
    decision_path_text = ''
    if request.method == 'POST':
        try:
            # Collect input data
            input_data = [float(request.form.get('sex', 0)),
                          int(request.form['single']),
                          int(request.form['unemployed']),
                          int(request.form['age']),
                          int(request.form['credit']),
                          int(request.form['loanDuration']),
                          int(request.form['purposeOfLoan']),
                          int(request.form['installmentRate']),
                          float(request.form.get('housing', 1))]

            # Prepare data for prediction
            input_df = pd.DataFrame([input_data],
                                    columns=["Sex", "Single", "Unemployed", "Age", "Credit", "LoanDuration",
                                             "PurposeOfLoan", "InstallmentRate", "Housing"])
            prediction_result = model.predict(input_df)[0]
            prediction_text = 'Predicted Label: ' + str(prediction_result)

            # Get the decision path for the input
            feature_names = ["Sex", "Single", "Unemployed", "Age", "Credit", "LoanDuration", "PurposeOfLoan",
                             "InstallmentRate", "Housing"]
            decision_path_text = get_decision_path_details(model, input_df, feature_names)
        except Exception as e:
            prediction_text = f"Error in prediction: {str(e)}"

    return render_template_string(HTML, prediction=prediction_text, decision_path=decision_path_text)


if __name__ == '__main__':
    app.run(debug=True)
