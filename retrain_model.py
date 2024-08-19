
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Generate synthetic data as an example
np.random.seed(42)
n_samples = 1000
data = {
    'Age': np.random.randint(20, 60, size=n_samples),
    'MonthlyIncome': np.random.randint(3000, 15000, size=n_samples),
    'YearsAtCompany': np.random.randint(1, 30, size=n_samples),
    'JobSatisfaction': np.random.randint(1, 5, size=n_samples),
    'WorkLifeBalance': np.random.randint(1, 5, size=n_samples),
    'OverTime': np.random.choice([0, 1], size=n_samples),
    'Attrition': np.random.choice([0, 1], size=n_samples)
}
df = pd.DataFrame(data)

# Split the data
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model using joblib (or pickle)
import joblib
joblib.dump(model, 'model_current.pkl')
