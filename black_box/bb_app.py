from flask import Flask, request, render_template_string
from markupsafe import escape
from html import escape
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
import csv

app = Flask(__name__)

# Load the Decision Tree model
model_path = 'black_box/bb_model'
model = load_model(model_path)

# HTML for the web form
HTML = '''
    <html>
        <body>
            <h2>Loan Application Classifier</h2>
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
            {{ prediction }}
        </body>
    </html>
'''


@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = ''
    if request.method == 'POST':
        # Get data from the form
        input_data = [float(request.form.get('sex', 0)),
                      int(request.form['single']),
                      int(request.form['unemployed']),
                      int(request.form['age']),
                      int(request.form['credit']),
                      int(request.form['loanDuration']),
                      int(request.form['purposeOfLoan']),
                      int(request.form['installmentRate']),
                      float(request.form.get('housing', 1))]

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=["Sex", "Single", "Unemployed", "Age", "Credit", "LoanDuration",
                                                       "PurposeOfLoan", "InstallmentRate", "Housing"])


        original_input = input_df.copy()

        # Features to be scaled
        features_to_scale = ['Age', 'Credit', 'LoanDuration']

        # Load the scaler
        scaler_path = 'black_box/bb_scaler.joblib'  # Update this path
        scaler = load(scaler_path)
        input_df[features_to_scale] = scaler.fit_transform(input_df[features_to_scale])

        # Make prediction
        prediction_result = model.predict(input_df)
        binary_prediction = 1 if prediction_result[0] >= 0.5 else 0
        prediction = 'Predicted Label: ' + str(prediction_result[0]) + ', Binary: ' + str(binary_prediction)

        original_input['Prediction'] = prediction_result[0]
        original_input['BinaryPrediction'] = binary_prediction

        # Path to resutls CSV
        csv_file_path = 'data/results.csv'

        # Open the file in append mode and write the new row
        with open(csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(original_input.values[0])


    return render_template_string(HTML, prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)