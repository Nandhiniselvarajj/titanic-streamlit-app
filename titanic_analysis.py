import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Better looking plots
sns.set(style="whitegrid")

import pandas as pd


# Load dataset
df = pd.read_csv("D:/AI-Project/titanic.csv")

# View initial data
print(df.head())
print(df.info())

# ----------------------------------------
# Data Cleaning Starts Here
# ----------------------------------------

# Check missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Fill missing Age values with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Drop the Cabin column
df.drop(columns=['Cabin'], inplace=True)

# Drop rows with any remaining missing data
df.dropna(inplace=True)

# Confirm all missing values handled
print("After cleaning:")
print(df.isnull().sum())
sns.countplot(data=df, x='Survived')
plt.title("Survival Count (0 = No, 1 = Yes)")
plt.show()
sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Convert 'Sex' column to numeric (male: 0, female: 1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Select features (X) and target (y)
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Predict for a single passenger
sample = pd.DataFrame({
    'Pclass': [2],
    'Sex': [1],        # female
    'Age': [25],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [30.0]
})

prediction = model.predict(sample)
print("Prediction (1 = Survived, 0 = Not):", prediction[0])
