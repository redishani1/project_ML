import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load the data
df = pd.read_csv(r"C:/Documents/project_ML/creditcard.csv")

# Load the model
model = joblib.load(r"C:/Documents/project_ML/models/smote_xgb.joblib")

# Prepare the new transaction data
X_new = pd.read_csv(r"path/to/new_transactions.csv")  # same features & preprocessing

# Make predictions
probs = model.predict_proba(X_new)[:,1]

# Define the pipeline for training
ct = joblib.load(r"C:/Documents/project_ML/models/scalers.joblib")
X = df.drop(columns=['Class'])
y = df['Class']
pipeline = Pipeline([
    ('preprocess', ct),           # applies same scaling used to make normalized CSV
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Train the model
pipeline.fit(X, y)  # use stratified split in practice
joblib.dump(pipeline, r"C:/Documents/project_ML/models/baseline_pipeline.joblib")