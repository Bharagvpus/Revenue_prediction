from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("final_shopper_model.keras")
CSV_FILE = "prediction_log.csv"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
            'Administrative': float(request.form['Administrative']),
            'Administrative_Duration': float(request.form['Administrative_Duration']),
            'Informational': float(request.form['Informational']),
            'Informational_Duration': float(request.form['Informational_Duration']),
            'ProductRelated': float(request.form['ProductRelated']),
            'ProductRelated_Duration': float(request.form['ProductRelated_Duration']),
            'BounceRates': float(request.form['BounceRates']),
            'ExitRates': float(request.form['ExitRates']),
            'PageValues': float(request.form['PageValues']),
            'SpecialDay': float(request.form['SpecialDay']),
            'OperatingSystems': float(request.form['OperatingSystems']),
            'Browser': float(request.form['Browser']),
            'Region': float(request.form['Region']),
            'TrafficType': float(request.form['TrafficType']),
            'Weekend': float(request.form['Weekend']),
            'New_Visitor': float(request.form['New_Visitor']),
            'Other': float(request.form['Other']),
            'Returning_Visitor': float(request.form['Returning_Visitor']),
        }

        # Predict
        input_array = np.array(list(data.values())).reshape(1, -1)
        prediction = model.predict(input_array)
        result = "Likely to Purchase" if prediction[0][0] >= 0.5 else "Unlikely to Purchase"

        # Save to CSV
        data['Prediction'] = result
        df_log = pd.DataFrame([data])
        if not os.path.exists(CSV_FILE):
            df_log.to_csv(CSV_FILE, index=False)
        else:
            df_log.to_csv(CSV_FILE, mode='a', header=False, index=False)

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)