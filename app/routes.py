from app import app  # Import the app object from app/__init__.py
from flask import render_template, request, jsonify
import pickle
import numpy as np

from flask import render_template, request, jsonify
import csv
import os

from flask import render_template, request, jsonify, send_file
import pickle
import numpy as np

# Load the trained model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_profile', methods=['POST'])
def check_profile():
    data = request.get_json()
    feature_vector = [
        data.get('profile_pic', 0),
        data.get('nums_length_username', 0),
        data.get('fullname_words', 0),
        data.get('nums_length_fullname', 0),
        data.get('name_equals_username', 0),
        data.get('description_length', 0),
        data.get('external_URL', 0),
        data.get('private', 0),
        data.get('posts', 0),
        data.get('followers', 0),
        data.get('follows', 0)
    ]
    prediction = model.predict([feature_vector])
    fake = int(prediction[0])
    return jsonify({'fake': fake})

@app.route('/create_csv')
def create_csv_page():
    return render_template('create_csv.html')

@app.route('/create_csv', methods=['POST'])
def create_csv():
    data = request.get_json()

    # Define the CSV file path
    csv_file_path = 'data/new_data.csv'

    # Check if the CSV file exists to determine if we need headers
    file_exists = os.path.isfile(csv_file_path)

    # Write data to the CSV file
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = [
            'profile_pic', 'nums_length_username', 'fullname_words', 'nums_length_fullname', 
            'name_equals_username', 'description_length', 'external_URL', 'private', 
            'posts', 'followers', 'follows', 'fake'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'profile_pic': data.get('profilePic', 0),
            'nums_length_username': data.get('numsLengthUsername', 0),
            'fullname_words': data.get('fullnameWords', 0),
            'nums_length_fullname': data.get('numsLengthFullname', 0),
            'name_equals_username': data.get('nameEqualsUsername', 0),
            'description_length': data.get('descriptionLength', 0),
            'external_URL': data.get('externalURL', 0),
            'private': data.get('private', 0),
            'posts': data.get('posts', 0),
            'followers': data.get('followers', 0),
            'follows': data.get('follows', 0),
            'fake':data.get('isFake',0)
             
        })

    return jsonify({'message': 'CSV file updated successfully!'})

@app.route('/roc_curve')
def roc_curve():
    return send_file('static/roc_curve.png', mimetype='image/png')