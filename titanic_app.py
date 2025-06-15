# titanic_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('train.csv')

# Preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# One-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Feature and label
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']

# Train model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🚢 Titanic Survival Prediction App")

st.sidebar.header("Passenger Information")

# Sidebar input
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
age = st.sidebar.slider("Age", 0, 80, 30)
sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 32.0)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Feature engineering
family_size = sibsp + parch + 1
sex_male = 1 if sex == 'male' else 0
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0

# Create input DataFrame
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'FamilySize': [family_size],
    'Sex_male': [sex_male],
    'Embarked_Q': [embarked_Q],
    'Embarked_S': [embarked_S]
})

# Make prediction
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0][1]

# Output
st.markdown("## 🧠 Prediction Outcome")

# Progress bar (optional visual)
st.progress(prediction_proba)

# Probability meter
st.metric(label="Survival Probability", value=f"{prediction_proba:.2%}")
st.markdown("## 👤 Passenger Summary")
st.info(f"""
- **Class**: {pclass}
- **Age**: {age}
- **Sex**: {sex}
- **Fare**: ${fare}
- **Siblings/Spouses Aboard**: {sibsp}
- **Parents/Children Aboard**: {parch}
- **Embarked**: {embarked}
""")


# Stylized result box
if prediction == 1:
    st.markdown(
        f"""
        <div style='padding: 1rem; background-color: #d1fae5; color: #065f46; border-radius: 10px; font-size: 18px;'>
        ✅ <strong>Survived</strong><br>
        Estimated Probability: <strong>{prediction_proba:.2%}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <div style='padding: 1rem; background-color: #fee2e2; color: #991b1b; border-radius: 10px; font-size: 18px;'>
        ❌ <strong>Did Not Survive</strong><br>
        Estimated Probability: <strong>{prediction_proba:.2%}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

