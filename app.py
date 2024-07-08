# app.py

from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('models/model.pkl')  # Ensure the path is correct

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_profile', methods=['POST'])
def check_profile():
    data = request.json

    # Ensure the data has the correct feature names
    feature_names = [
        'profilePic', 'numsLengthUsername', 'fullnameWords',
        'numsLengthFullname', 'nameEqualsUsername', 'descriptionLength',
        'externalURL', 'private', 'posts', 'followers', 'follows'
    ]

    # Convert input data to DataFrame with correct feature names
    input_data = pd.DataFrame([data], columns=feature_names)

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Return the result
    result = {'fake': int(prediction)}
    return jsonify(result)

@app.route('/roc_curve')
def roc_curve():
    return send_file('static/roc_curve.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
