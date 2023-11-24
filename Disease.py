import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

csv_file_path = 'C:/Users/Ncc/Desktop/py/Disease_symptom_and_patient_profile_dataset.csv'
df = pd.read_csv(csv_file_path)

for i in df.columns:
    values = df[i].value_counts()
    print(values)

import matplotlib.pyplot as plt
import seaborn as sns
for i in df.columns:
    plt.figure(figsize=(20,8))
    plt.xticks(rotation=90)  
    plt.tight_layout()
    sns.histplot(df[i])
    plt.show()

X_train = df.drop(columns=['Outcome Variable'],axis=1)
y_train= df['Outcome Variable']
X_train.shape , y_train.shape
X_train

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder ()

columns_to_encode = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Disease']

for column in columns_to_encode:
    if column != 'Age':
        X_train[column] = encoder.fit_transform(X_train[column])
X_train
encoder = LabelEncoder ()
y_train = encoder.fit_transform(y_train)
y_train
sns.heatmap(X_train.corr())

corr_matrix = X_train.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")

plt.show()

min_age = df['Age'].min()
max_age = df['Age'].max()
median_age = df['Age'].median()
print(f"min_age {min_age}")
print(f"max_age: {max_age}")
print(f"median_age: {median_age}")

mapping_fever = {'Yes': 1, 'No': 0}
df['Fever'] = df['Fever'].map(mapping_fever)

mapping_cough = {'Yes': 1, 'No': 0}
df['Cough'] = df['Cough'].map(mapping_cough)

mapping_fatigue = {'Yes': 1, 'No': 0}
df['Fatigue'] = df['Fatigue'].map(mapping_fatigue)

mapping_difficulty_breathing = {'Yes': 1, 'No': 0}
df['Difficulty Breathing'] = df['Difficulty Breathing'].map(mapping_difficulty_breathing)

mapping_outcome = {'Positive': 1, 'Negative': 0}

df['Outcome Variable'] = df['Outcome Variable'].map(mapping_outcome)

mapping = {'Low': 0, 'Normal': 0.5, 'High': 1}

df['Blood Pressure'] = df['Blood Pressure'].map(mapping)

df['Cholesterol Level'] = df['Cholesterol Level'].map(mapping)

print(df)

count_yes = df['Outcome Variable'].sum()
count_no = len(df) - count_yes

print("Yes:", count_yes)
print("No:", count_no)
median_value = df['Outcome Variable'].median()
print("median:", median_value)

