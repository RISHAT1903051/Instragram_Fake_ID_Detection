# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import os

# Load the original data
original_data = pd.read_csv('data/train.csv')

# Load additional data
additional_data = pd.read_csv('data/new_data.csv')

# Concatenate original and additional data
data = pd.concat([original_data, additional_data], ignore_index=True)

# Separate features and target
X = data.drop('fake', axis=1)
y = data['fake']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
model_path = 'models/model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Generate ROC curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Ensure the 'static' directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Save the ROC curve plot in the 'static' directory
plt.savefig('static/roc_curve.png')
