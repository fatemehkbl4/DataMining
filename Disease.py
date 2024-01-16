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
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


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
# image.show()
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
# image2.show()
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
# image3.show()
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
# plot_confusion_matrix(confusion_matrix(y_test, y_pred_linear), "Linear")
# plot_confusion_matrix(confusion_matrix(y_test, y_pred_poly), "Polynomial")
# plot_confusion_matrix(confusion_matrix(y_test, y_pred_rbf), "RBF")

# K-mean algoritm

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
num_clusters = range(2, 11)
sse_values = []
cluster_centers_values = []
cluster_members_count = []

for n_clusters in num_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42 , n_init=10)
    kmeans.fit(X_train_scaled)
    # SSE
    sse_values.append(kmeans.inertia_)
    # مقادیر ویژگی‌های مراکز خوشه
    cluster_centers_values.append(kmeans.cluster_centers_)
    # تعداد اعضای هر خوشه
    cluster_members_count.append(np.bincount(kmeans.labels_))

table_data = pd.DataFrame({
    'Num Clusters': num_clusters,
    'SSE': sse_values,
    'Cluster Centers Values': cluster_centers_values,
    'Cluster Members Count': cluster_members_count
})
print(table_data)
num_clusters = range(2, 11)
# نمایش الگوریتم K-Means بر روی داده‌ها برای هر تعداد خوشه
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()
for i, n_clusters in enumerate(num_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42 , n_init=10)
    kmeans.fit(X_train_scaled)
    # نمایش نقاط داده به همراه مراکز خوشه
    axes[i].scatter(X_train_scaled.iloc[:, 0], X_train_scaled.iloc[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
    axes[i].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, color='red')
    axes[i].set_title(f'Num Clusters = {n_clusters}')
plt.tight_layout()
plt.show()
# ترسیم نمودار خط SSE
plt.figure(figsize=(10, 6))
plt.plot(num_clusters, sse_values, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.show()

# DBSCAN algoritm

features = df.drop(columns=['Outcome Variable'], axis=1)
label_encoder = LabelEncoder()
for column in features.columns:
    if features[column].dtype == np.object_:
        features[column] = label_encoder.fit_transform(features[column])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

for n_clusters in range(2, 11):
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    df[f'Cluster Labels ({n_clusters} Clusters)'] = labels
# print(df)
plt.figure(figsize=(15, 3))
for n_clusters in range(2, 11):
    plt.subplot(2, 5, n_clusters - 1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df[f'Cluster Labels ({n_clusters} Clusters)'], cmap='viridis', s=50)
    plt.title(f'DBSCAN Clustering - {n_clusters} Clusters')
plt.tight_layout()
plt.show()

# fuzzy

# import skfuzzy as fuzz
# num_clusters_fuzzy = range(2, 11)
# sse_values_fuzzy = []
# cluster_info_fuzzy = []
# for n_clusters in num_clusters_fuzzy:
#     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_train_scaled.T, n_clusters, 2, error=0.005, maxiter=1000 , n_init=10 )
#     centers = cntr.T
#     print(f'Centers for {n_clusters} Clusters:\n{centers}\n')
#     sse_values_fuzzy.append(np.sum((X_train_scaled - centers[np.argmax(u, axis=0)]) ** 2))
#     cluster_info_fuzzy.append({
#         'Number of Clusters': n_clusters,
#         'Cluster Centers': centers,
#         'Membership Matrix': u,
#         'FPC': fpc
#     })
# table_data_fuzzy = pd.DataFrame(cluster_info_fuzzy)
# table_data_fuzzy['SSE'] = sse_values_fuzzy
# print(table_data_fuzzy)

print("------------------------------------------------------------------------------------------------------")

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# ایجاد یک تابع برای محاسبه و نمایش معیارهای ارزیابی
def calculate_and_display_metrics(y_true, y_pred, algorithm_name):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    metrics_dict = {
        'Algorithm': algorithm_name,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1-Score': f1
    }
    return metrics_dict
metrics_results = []
# Decision Tree with Gini
y_pred_gini = model_gini.predict(X_test)
metrics_gini = calculate_and_display_metrics(y_test, y_pred_gini, 'Decision Tree (Gini)')
metrics_results.append(metrics_gini)
# Decision Tree with Entropy
y_pred_entropy = model_entropy.predict(X_test)
metrics_entropy = calculate_and_display_metrics(y_test, y_pred_entropy, 'Decision Tree (Entropy)')
metrics_results.append(metrics_entropy)
# Decision Tree with Misclassification
y_pred_misclassification = model_misclassification.predict(X_test)
metrics_misclassification = calculate_and_display_metrics(y_test, y_pred_misclassification, 'Decision Tree (Misclassification)')
metrics_results.append(metrics_misclassification)
# SVM with Linear Kernel
y_pred_linear = svm_linear.predict(X_test_scaled)
metrics_svm_linear = calculate_and_display_metrics(y_test, y_pred_linear, 'SVM (Linear Kernel)')
metrics_results.append(metrics_svm_linear)
# SVM with Polynomial Kernel
y_pred_poly = svm_poly.predict(X_test_scaled)
metrics_svm_poly = calculate_and_display_metrics(y_test, y_pred_poly, 'SVM (Polynomial Kernel)')
metrics_results.append(metrics_svm_poly)
# SVM with RBF Kernel
y_pred_rbf = svm_rbf.predict(X_test_scaled)
metrics_svm_rbf = calculate_and_display_metrics(y_test, y_pred_rbf, 'SVM (RBF Kernel)')
metrics_results.append(metrics_svm_rbf)
# Display results in a table
metrics_table = pd.DataFrame(metrics_results)
print(metrics_table)



# ROC 
from sklearn.metrics import roc_curve, auc

# تابع برای رسم نمودار ROC و محاسبه AUC
def plot_roc_curve(y_true, y_probs, algorithm_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{algorithm_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {algorithm_name}')
    plt.legend(loc="lower right")
    plt.show()
# Decision Tree with Gini
y_probs_gini = model_gini.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_probs_gini, 'Decision Tree (Gini)')
# Decision Tree with Entropy
y_probs_entropy = model_entropy.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_probs_entropy, 'Decision Tree (Entropy)')
# Decision Tree with Misclassification
y_probs_misclassification = model_misclassification.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_probs_misclassification, 'Decision Tree (Misclassification)')
# SVM with Linear Kernel
y_probs_linear = svm_linear.decision_function(X_test_scaled)
plot_roc_curve(y_test, y_probs_linear, 'SVM (Linear Kernel)')
# SVM with Polynomial Kernel
y_probs_poly = svm_poly.decision_function(X_test_scaled)
plot_roc_curve(y_test, y_probs_poly, 'SVM (Polynomial Kernel)')
# SVM with RBF Kernel
y_probs_rbf = svm_rbf.decision_function(X_test_scaled)
plot_roc_curve(y_test, y_probs_rbf, 'SVM (RBF Kernel)')
