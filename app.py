from flask import Flask, render_template, request
from flask import render_template
import joblib
from sklearn.preprocessing import LabelEncoder
import json

app = Flask(__name__)

model = joblib.load('random_forest_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

with open('label_encoder_mapping.json', 'r') as file:
    label_encoder_mapping = json.load(file)

designation_mapping = label_encoder_mapping.get('designation_mapping', {})
unit_mapping = label_encoder_mapping.get('unit_mapping', {})

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    # Get input data from the form
    DESIGNATION = request.form['designation']
    UNIT = request.form['unit']
    RATINGS = float(request.form['ratings'])
    TOTAL_EXP = float(request.form['total_exp'])
    EXP_IN_CURRENT_ROLE = float(request.form['exp_in_current_role'])
    LEAVE_UTILIZATION_RATIO = float(request.form['leaves_ratio'])

    DESIGNATION_encoded = designation_mapping.get(DESIGNATION, -1)
    UNIT_encoded = unit_mapping.get(UNIT, -1)

    if DESIGNATION_encoded == -1 or UNIT_encoded == -1:
        return "Invalid category encountered."

    input_data = [DESIGNATION_encoded, UNIT_encoded, RATINGS, TOTAL_EXP, EXP_IN_CURRENT_ROLE, LEAVE_UTILIZATION_RATIO]
    input_data = [float(item) if isinstance(item, (int, float)) else item for item in input_data]

    predicted_salary = model.predict([input_data])[0]

    return render_template('result.html', prediction_result=f"Predicted Salary for {UNIT} {DESIGNATION}: ${predicted_salary:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
