#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv("C:\\Users\\bodap\\Downloads\\diabetes_data_upload.csv")


# In[3]:


# Display the first few rows of the DataFrame
df.head()


# In[4]:


# Display information about the DataFrame
df.info()


# In[5]:


# Display the shape of the DataFrame (number of rows and columns)
df.shape


# In[6]:


# Check for missing values in the DataFrame
df.isnull().any()


# In[7]:


# Display the count of missing values for each column
df.isnull().sum()


# In[8]:


# Create a count plot for the 'class' column
sns.countplot(df['class'])


# In[10]:


# Create count plots for Polyuria with 'hue' based on the 'class' column
sns.countplot(df['Polyuria'],hue=df['class'], data=df)


# In[11]:


# Create count plots for sudden weight loss with 'hue' based on the 'class' column
sns.countplot(df['sudden weight loss'],hue=df['class'], data=df)


# In[12]:


# Create count plots for Polyphagia with 'hue' based on the 'class' column
sns.countplot(df['Polyphagia'],hue=df['class'], data=df)


# In[13]:


# Create count plots for Polydipsia with 'hue' based on the 'class' column
sns.countplot(df['Polydipsia'],hue=df['class'], data=df)


# In[19]:


# Encode categorical columns
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


# In[20]:


# Display information about the DataFrame after encoding
df.info()


# In[21]:


# Create a boxplot to compare the distribution of age for each class
sns.boxplot(x='class', y='Age', data=df)
plt.title('Distribution of age for each class')
plt.show()


# In[22]:


# Create a histogram of the 'Age' column
plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()


# In[23]:


# Create a countplot to compare the distribution of gender for each class
sns.countplot(x='class', hue='Gender', data=df)
plt.title('Distribution of gender for each class')
plt.show()


# In[24]:


# Calculating the correlation matrix and create a heatmap
x=df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(x,annot=True, cmap='YlOrRd_r',fmt='.2f',linewidth=0.5)


# In[25]:


# Create a pair plot for the DataFrame
sns.pairplot(df)
plt.show()


# In[26]:


# Convert 'Age' to standard format
df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()


# In[27]:


df.head()


# In[39]:


# Separate features and target variable
X = df.drop('class', axis=1)
y = df['class']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[46]:



import pickle
# Select only the relevant columns (including 'Gender_encoded')
selected_columns = ['Age', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia',
                     'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing',
                     'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity', 'Gender_encoded']
X_train_selected = X_train[selected_columns]
X_test_selected = X_test[selected_columns]
# Example: Use RandomForestClassifier for prediction
clf = RandomForestClassifier()
clf.fit(X_train_selected, y_train)
with open("C:\\Users\\bodap\\model_job.joblib", "wb") as model_file:
    pickle.dump(clf, model_file)


# In[41]:


from sklearn.metrics import accuracy_score
# Make predictions on the test set
y_pred = clf.predict(X_test_selected)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# Display accuracy
print(f"The accuracy of the model on the test set is: {accuracy:.2%}")


# In[42]:


# Calculate precision
precision = precision_score(y_test, y_pred, average='binary')  # For binary classification
# For multi-class classification, you can use average='weighted' or 'macro' or 'micro' based on your requirement

# Calculate recall
recall = recall_score(y_test, y_pred, average='binary')  # For binary classification
# For multi-class classification, you can use average='weighted' or 'macro' or 'micro' based on your requirement

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='binary')  # For binary classification
# For multi-class classification, you can use average='weighted' or 'macro' or 'micro' based on your requirement

# Display precision, recall, and F1 score
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")


# In[48]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder


X = df.drop("class", axis=1)
y = df["class"]

X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB()
}

f1_scores = []
model_names = []

colors = ["#FF4136", "#FF851B", "#FFDC00", "#2ECC40"]  # Specify colors for bars

for model_name, model, color in zip(models.keys(), models.values(), colors):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, pos_label='Positive')
    f1_scores.append(f1)
    model_names.append(model_name)

plt.figure(figsize=(10, 6))
plt.bar(model_names, f1_scores, color=colors)  # Assign colors to bars
plt.xlabel("Algorithm")
plt.ylabel("F1 Score")
plt.title("F1 Score Comparison of ML Algorithms")
plt.show()


# In[49]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder


X = df.drop("class", axis=1)
y = df["class"]

X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB()
}

recall_scores = []
model_names = []

colors = ["#FF4136", "#FF851B", "#FFDC00", "#2ECC40"]  # Specify colors for bars

for model_name, model, color in zip(models.keys(), models.values(), colors):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    recall = recall_score(y_test, y_pred, pos_label='Positive')
    recall_scores.append(recall)
    model_names.append(model_name)

plt.figure(figsize=(10, 6))
plt.bar(model_names, recall_scores, color=colors)  # Assign colors to bars
plt.xlabel("Algorithm")
plt.ylabel("Recall")
plt.title("Recall Comparison of ML Algorithms")
plt.show()


# In[50]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

X = df.drop("class", axis=1)
y = df["class"]

X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB()
}

accuracy_scores = []
model_names = []

colors = ["#FF4136", "#FF851B", "#FFDC00", "#2ECC40"]  # Specify colors for bars

for model_name, model, color in zip(models.keys(), models.values(), colors):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    model_names.append(model_name)

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracy_scores, color=colors)  # Assign colors to bars
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of ML Algorithms")
plt.show()


# In[ ]:




