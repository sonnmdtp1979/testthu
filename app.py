from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('wine_quality_model.joblib')

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collecting the inputs from the form
            fixed_acidity = float(request.form['fixed-acidity'])
            volatile_acidity = float(request.form['volatile-acidity'])
            citric_acid = float(request.form['citric-acid'])
            residual_sugar = float(request.form['residual-sugar'])
            
            # Making prediction
            features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar]])
            prediction = model.predict(features)

            # Showing the prediction result on the same page
            return render_template('index.html', prediction_text=f'Predicted Quality: {prediction[0]}')

        except Exception as e:
            # Handling exceptions
            return render_template('index.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
