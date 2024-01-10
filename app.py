#!/usr/bin/env python
# coding: utf-8

# In[18]:


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv(r"diabetes_data_upload.csv")
label_encoder_gender = LabelEncoder()
label_encoder_polyuria = LabelEncoder()
label_encoder_polydipsia = LabelEncoder()
label_encoder_sudden_weight_loss = LabelEncoder()
label_encoder_weakness = LabelEncoder()
label_encoder_polyphagia = LabelEncoder()
label_encoder_genital_thrush = LabelEncoder()
label_encoder_visual_blurring = LabelEncoder()
label_encoder_itching = LabelEncoder()
label_encoder_irritability = LabelEncoder()
label_encoder_delayed_healing = LabelEncoder()
label_encoder_partial_paresis = LabelEncoder()
label_encoder_muscle_stiffness = LabelEncoder()
label_encoder_alopecia = LabelEncoder()
label_encoder_obesity = LabelEncoder()
label_encoder_class = LabelEncoder()
df['Gender_encoded'] = label_encoder_gender.fit_transform(df['Gender'])
df['Polyuria'] = label_encoder_polyuria.fit_transform(df['Polyuria'])
df['Polydipsia'] = label_encoder_polydipsia.fit_transform(df['Polydipsia'])
df['sudden weight loss'] = label_encoder_sudden_weight_loss.fit_transform(df['sudden weight loss'])
df['weakness'] = label_encoder_weakness.fit_transform(df['weakness'])
df['Polyphagia'] = label_encoder_polyphagia.fit_transform(df['Polyphagia'])
df['Genital thrush'] = label_encoder_genital_thrush.fit_transform(df['Genital thrush'])
df['visual blurring'] = label_encoder_visual_blurring.fit_transform(df['visual blurring'])
df['Itching'] = label_encoder_itching.fit_transform(df['Itching'])
df['Irritability'] = label_encoder_irritability.fit_transform(df['Irritability'])
df['delayed healing'] = label_encoder_delayed_healing.fit_transform(df['delayed healing'])
df['partial paresis'] = label_encoder_partial_paresis.fit_transform(df['partial paresis'])
df['muscle stiffness'] = label_encoder_muscle_stiffness.fit_transform(df['muscle stiffness'])
df['Alopecia'] = label_encoder_alopecia.fit_transform(df['Alopecia'])
df['Obesity'] = label_encoder_obesity.fit_transform(df['Obesity'])
df['class'] = label_encoder_class.fit_transform(df['class'])
df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
selected_columns = ['Age', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia',
                     'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing',
                     'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity', 'Gender_encoded']
X_train_selected = X_train[selected_columns]
X_test_selected = X_test[selected_columns]
clf = RandomForestClassifier()
clf.fit(X_train_selected, y_train)
age = st.number_input("Enter Age:")
polyuria = st.text_input("Polyuria: Excessive Urination (Enter 1 for Yes, 0 for No):")
polydipsia = st.text_input("Polydipsia: Excessive Thirstiness (Enter 1 for Yes, 0 for No):")
sudden_weight_loss = st.text_input("Sudden Weight Loss (Enter 1 for Yes, 0 for No):")
weakness = st.text_input("Weakness (Enter 1 for Yes, 0 for No):")
polyphagia = st.text_input("Polyphagia: Feeling of extreme hunger (Enter 1 for Yes, 0 for No):")
genital_thrush = st.text_input("Genital Thrush: Vaginal yeast Infection (Enter 1 for Yes, 0 for No):")
visual_blurring = st.text_input("Visual Blurring (Enter 1 for Yes, 0 for No):")
itching = st.text_input("Itching (Enter 1 for Yes, 0 for No):")
irritability = st.text_input("Irritability (Enter 1 for Yes, 0 for No):")
delayed_healing = st.text_input("Delayed Healing (Enter 1 for Yes, 0 for No):")
partial_paresis = st.text_input("Partial Paresis (Enter 1 for Yes, 0 for No):")
muscle_stiffness = st.text_input("Muscle Stiffness (Enter 1 for Yes, 0 for No):")
alopecia = st.text_input("Alopecia: Hair loss (Enter 1 for Yes, 0 for No):")
obesity = st.text_input("Obesity (Enter 1 for Yes, 0 for No):")
polyuria = int(polyuria) if polyuria else 0
polydipsia = int(polydipsia) if polydipsia else 0
sudden_weight_loss = int(sudden_weight_loss) if sudden_weight_loss else 0
weakness = int(weakness) if weakness else 0
polyphagia = int(polyphagia) if polyphagia else 0
genital_thrush = int(genital_thrush) if genital_thrush else 0
visual_blurring = int(visual_blurring) if visual_blurring else 0
itching = int(itching) if itching else 0
irritability = int(irritability) if irritability else 0
delayed_healing = int(delayed_healing) if delayed_healing else 0
partial_paresis = int(partial_paresis) if partial_paresis else 0
muscle_stiffness = int(muscle_stiffness) if muscle_stiffness else 0
alopecia = int(alopecia) if alopecia else 0
obesity = int(obesity) if obesity else 0
gender = st.selectbox("Select Gender:", df['Gender'].unique())
gender_encoded = label_encoder_gender.transform([gender])[0]
prediction = clf.predict([[age, polyuria, polydipsia,
                           sudden_weight_loss, weakness, polyphagia,
                           genital_thrush, visual_blurring, itching,
                           irritability, delayed_healing, partial_paresis,
                           muscle_stiffness, alopecia, obesity, gender_encoded]])
st.subheader("Prediction:")
prediction_output = st.empty()  
if st.button("Get Prediction"):
    prediction = clf.predict([[age, polyuria, polydipsia,
                               sudden_weight_loss, weakness, polyphagia,
                               genital_thrush, visual_blurring, itching,
                               irritability, delayed_healing, partial_paresis,
                               muscle_stiffness, alopecia, obesity, gender_encoded]])

    prediction_output.write(f"The predicted class is: {prediction[0]}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




