import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import export_graphviz
import graphviz
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import fitz
from PIL import Image
from sklearn.metrics import confusion_matrix

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

csv_file_path = 'C:/Users/Ncc/Desktop/py/Disease_symptom_and_patient_profile_dataset.csv'
df = pd.read_csv(csv_file_path)

for i in df.columns:
    values = df[i].value_counts()
    print(values)

# for i in df.columns:
#     plt.figure(figsize=(20,8))
#     plt.xticks(rotation=90)  
#     plt.tight_layout()
#     sns.histplot(df[i])
#     plt.show()

X_train = df.drop(columns=['Outcome Variable'],axis=1)
y_train= df['Outcome Variable']
X_train.shape , y_train.shape
X_train

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
# plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
# plt.show()

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

column_to_normalize = ['Age']
scaler = MinMaxScaler()
df[column_to_normalize] = scaler.fit_transform(df[column_to_normalize])
print(df)

count_yes = df['Outcome Variable'].sum()
count_no = len(df) - count_yes
print("Yes:", count_yes)
print("No:", count_no)
median_value = df['Outcome Variable'].median()
print("median:", median_value)

# matrix 
def display_evaluation_results(model, X_test, y_test):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print(f'Confusion Matrix:\n{conf_matrix}\n')
    print(f'Classification Report:\n{class_report}\n')


# gini ایجاد یک مدل درخت تصمیم با شاخص
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
model_gini.fit(X_train, y_train)
y_pred = model_gini.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'gini Accuracy: {accuracy}')
dot_data_gini = export_graphviz(
    model_gini, 
    out_file=None, 
    feature_names=X_train.columns, 
    class_names=['Negative', 'Positive'], 
    filled=True, 
    rounded=True, 
    special_characters=True
)
graph = graphviz.Source(dot_data_gini)
graph.render("decision_tree_gini")
pdf_path = 'decision_tree_gini.pdf'
pdf_document = fitz.open(pdf_path)
page = pdf_document[0]
pixmap = page.get_pixmap()
image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
image.save('decision_tree_gini.png')
image.show()
print("Results for Gini Criterion:")
display_evaluation_results(model_gini, X_test, y_test)

# entropy ایجاد یک مدل درخت تصمیم با شاخص 
model_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
model_entropy.fit(X_train, y_train)
y_pred_entropy = model_entropy.predict(X_test)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
print(f'entropy Accuracy : {accuracy_entropy}')
dot_data_entropy = export_graphviz(
    model_entropy, 
    out_file=None, 
    feature_names=X_train.columns, 
    class_names=['Negative', 'Positive'], 
    filled=True, 
    rounded=True, 
    special_characters=True
)
graph_entropy = graphviz.Source(dot_data_entropy)
graph_entropy.render("decision_tree_entropy")
pdf_path2 = 'decision_tree_entropy.pdf'
pdf_document = fitz.open(pdf_path2)
page2 = pdf_document[0]
pixmap2 = page2.get_pixmap()
image2 = Image.frombytes("RGB", [pixmap2.width, pixmap2.height], pixmap2.samples)
image2.save('decision_tree_entropy.png')
image2.show()
print("Results for Entropy Criterion:")
display_evaluation_results(model_entropy, X_test, y_test)


# misclassification ایجاد یک مدل درخت تصمیم با شاخص 
model_misclassification = DecisionTreeClassifier(criterion='gini', splitter='random', random_state=42)
model_misclassification.fit(X_train, y_train)
y_pred_misclassification = model_misclassification.predict(X_test)
accuracy_misclassification = accuracy_score(y_test, y_pred_misclassification)
print(f'misclassification Accuracy : {accuracy_misclassification}')
dot_data_misclassification = export_graphviz(
    model_misclassification, 
    out_file=None, 
    feature_names=X_train.columns, 
    class_names=['Negative', 'Positive'], 
    filled=True, 
    rounded=True, 
    special_characters=True
)
graph_misclassification = graphviz.Source(dot_data_misclassification)
graph_misclassification.render("decision_tree_misclassification")
graph_misclassification = graphviz.Source(dot_data_misclassification)
pdf_path3 = 'decision_tree_misclassification.pdf'
pdf_document = fitz.open(pdf_path3)
page3 = pdf_document[0]
pixmap3 = page3.get_pixmap()
image3 = Image.frombytes("RGB", [pixmap3.width, pixmap3.height], pixmap3.samples)
image3.save('decision_tree_misclassification.png')
image3.show()
print("Results for Misclassification Criterion:")
display_evaluation_results(model_misclassification, X_test, y_test)

# svm
def manual_scaling(data, mean, std):
    scaled_data = (data - mean) / std
    return scaled_data
mean_train = np.mean(X_train, axis=0)
std_train = np.std(X_train, axis=0)
X_train_scaled = manual_scaling(X_train, mean_train, std_train)
X_test_scaled = manual_scaling(X_test, mean_train, std_train)

svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train_scaled, y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f'Accuracy (Linear Kernel): {accuracy_linear}')

svm_poly = SVC(kernel='poly', degree=3, C=1)
svm_poly.fit(X_train_scaled, y_train)
y_pred_poly = svm_poly.predict(X_test_scaled)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print(f'Accuracy (Polynomial Kernel): {accuracy_poly}')

svm_rbf = SVC(kernel='rbf', C=1)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f'Accuracy (RBF Kernel): {accuracy_rbf}')

print("Results for SVM with Linear Kernel:")
display_evaluation_results(svm_linear, X_test_scaled, y_test)
print("Results for SVM with Polynomial Kernel:")
display_evaluation_results(svm_poly, X_test_scaled, y_test)
print("Results for SVM with RBF Kernel:")
display_evaluation_results(svm_rbf, X_test_scaled, y_test)


def plot_confusion_matrix(conf_matrix, kernel_name):
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {kernel_name} Kernel")
    plt.show()
plot_confusion_matrix(confusion_matrix(y_test, y_pred_linear), "Linear")
plot_confusion_matrix(confusion_matrix(y_test, y_pred_poly), "Polynomial")
plot_confusion_matrix(confusion_matrix(y_test, y_pred_rbf), "RBF")
