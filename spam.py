import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
raw_mail_data = pd.read_csv('./mail_data.csv')

# Handle missing values
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# Normalize labels
mail_data['Category'] = mail_data['Category'].str.lower().map({'spam': 0, 'ham': 1}).astype(int)

# Visualize class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=mail_data['Category'].map({0: 'Spam', 1: 'Ham'}))
plt.title('Class Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

# Split data
X = mail_data['Message']
Y = mail_data['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# TF-IDF Vectorization (bigrams included)
feature_extraction = TfidfVectorizer(ngram_range=(1,2), min_df=1, lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Train Logistic Regression with balanced class weights
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_features, Y_train)

# Evaluate Model
train_pred = model.predict(X_train_features)
test_pred = model.predict(X_test_features)
print(f"Training Accuracy: {accuracy_score(Y_train, train_pred):.2f}")
print(f"Test Accuracy: {accuracy_score(Y_test, test_pred):.2f}")

# Confusion Matrix Visualization
cm = confusion_matrix(Y_test, test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Spam', 'Ham'], yticklabels=['Spam', 'Ham'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualize Top Features
feature_names = np.array(feature_extraction.get_feature_names_out())
coefs = model.coef_[0]
top_positive_indices = np.argsort(coefs)[-10:]
top_negative_indices = np.argsort(coefs)[:10]

top_features = np.hstack([feature_names[top_negative_indices], feature_names[top_positive_indices]])
top_coefficients = np.hstack([coefs[top_negative_indices], coefs[top_positive_indices]])

plt.figure(figsize=(10, 5))
plt.barh(top_features, top_coefficients, color=['red']*10 + ['blue']*10)
plt.title("Top 10 Spam and Ham Features")
plt.xlabel("Coefficient Value")
plt.show()

# Test with new email
input_mail = ["FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv"]
input_mail = [msg.lower().strip() for msg in input_mail]  # Normalize text
input_data_features = feature_extraction.transform(input_mail)

# Make prediction
prediction = model.predict(input_data_features)
if prediction[0]==0:
    print("Spam Mail")
else:
    print("Ham mail ")




