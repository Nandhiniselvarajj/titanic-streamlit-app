import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and clean data
df = pd.read_csv("D:/AI-Project/titanic.csv")
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].mean())

df.drop(columns=['Cabin'], inplace=True)
df.dropna(inplace=True)

# Features and labels
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 🧑‍🎤 Streamlit UI
st.title("Titanic Survival Predictor 🚢")
st.write("🚀 App is loaded successfully!")


# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 30.0)
if st.button("Predict"):
    sex_encoded = 0 if sex == "Male" else 1
    sample = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare]
    })

    prediction = model.predict(sample)
    proba = model.predict_proba(sample)[0][1]  # Probability of survival

    result = "Survived 🟢" if prediction[0] == 1 else "Did Not Survive 🔴"
    st.subheader(f"Prediction: {result}")
    st.write(f"Probability of Survival: {proba:.2%}")
