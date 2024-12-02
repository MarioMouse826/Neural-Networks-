# Install necessary libraries (uncomment if running in Google Colab)
# !pip install tensorflow scikit-learn seaborn matplotlib numpy pandas imbalanced-learn -q

# Importing necessary libraries
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend
import tensorflow as tf

# Avoid warnings
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
# Update the file path if running in Google Colab
# from google.colab import files
# uploaded = files.upload()
ds = pd.read_csv('Churn.csv')

# Data Overview
print("First 5 rows of the dataset:")
print(ds.head())
print("\nLast 5 rows of the dataset:")
print(ds.tail())
print("\nDataset dimensions:")
print(ds.shape)
print("\nDataset information:")
print(ds.info())
print("\nStatistical Summary:")
print(ds.describe().T)
print("\nChecking for missing values:")
print(ds.isnull().sum())
print("\nNumber of unique values in each column:")
print(ds.nunique())

# Dropping unnecessary columns
ds = ds.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Exploratory Data Analysis
# Function to plot boxplot and histogram
def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": (0.25, 0.75)}, figsize=figsize
    )
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="violet")
    sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins)
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")
    plt.show()

# Function to create labeled barplots
def labeled_barplot(data, feature, perc=False, n=None):
    total = len(data[feature])
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))
    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(data=data, x=feature, palette="Paired", order=data[feature].value_counts().index[:n])
    for p in ax.patches:
        label = "{:.1f}%".format(100 * p.get_height() / total) if perc else p.get_height()
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(label, (x, y), ha="center", va="center", size=12, xytext=(0, 5), textcoords="offset points")
    plt.show()

# Observations
histogram_boxplot(ds, 'CreditScore')
histogram_boxplot(ds, 'Age')
histogram_boxplot(ds, 'Balance')
histogram_boxplot(ds, 'EstimatedSalary')
labeled_barplot(ds, "Exited", perc=True)
labeled_barplot(ds, "Geography")
labeled_barplot(ds, "Gender")
labeled_barplot(ds, "Tenure")
labeled_barplot(ds, "NumOfProducts")
labeled_barplot(ds, "HasCrCard")
labeled_barplot(ds, "IsActiveMember")

# Correlation Heatmap
cols_list = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
plt.figure(figsize=(15, 7))
sns.heatmap(ds[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()

# Dummy Variable Creation
ds = pd.get_dummies(ds, columns=ds.select_dtypes(include=["object"]).columns.tolist(), drop_first=True)
ds = ds.astype(float)

# Train-validation-test Split
X = ds.drop(['Exited'], axis=1)
y = ds['Exited']
X_large, X_test, y_large, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_large, y_large, test_size=0.25, random_state=42, stratify=y_large, shuffle=True)

# Data Normalization
sc = StandardScaler()
X_train[cols_list] = sc.fit_transform(X_train[cols_list])
X_val[cols_list] = sc.transform(X_val[cols_list])
X_test[cols_list] = sc.transform(X_test[cols_list])

print("\nData preprocessing completed successfully!")
print("Train, validation, and test data shapes:")
print(X_train.shape, X_val.shape, X_test.shape)

# Initialize metrics tracking
train_metric_df = pd.DataFrame(columns=["recall"])
valid_metric_df = pd.DataFrame(columns=["recall"])

# Function to plot confusion matrix
def make_confusion_matrix(actual_targets, predicted_targets):
    cm = confusion_matrix(actual_targets, predicted_targets)
    labels = np.asarray(
        [["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
         for item in cm.flatten()]
    ).reshape(cm.shape[0], cm.shape[1])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

# SMOTE for balancing dataset
sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
print("After applying SMOTE:")
print(X_train_smote.shape, y_train_smote.shape)

# Define and evaluate models (examples provided for SGD and Adam optimizers)
def build_and_evaluate_model(model, history, model_name):
    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Plot recall
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Predictions
    y_train_pred = model.predict(X_train_smote) > 0.5
    y_val_pred = model.predict(X_val) > 0.5

    # Record recall
    train_metric_df.loc[model_name] = recall_score(y_train_smote, y_train_pred)
    valid_metric_df.loc[model_name] = recall_score(y_val, y_val_pred)

    # Classification report
    print(f"\nTrain Classification Report ({model_name}):")
    print(classification_report(y_train_smote, y_train_pred))
    print(f"\nValidation Classification Report ({model_name}):")
    print(classification_report(y_val, y_val_pred))

    # Confusion Matrix
    make_confusion_matrix(y_train_smote, y_train_pred)
    make_confusion_matrix(y_val, y_val_pred)

# Example model: Neural Network with Adam optimizer
backend.clear_session()
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train_smote.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
metric = keras.metrics.Recall()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[metric])
history = model.fit(X_train_smote, y_train_smote, batch_size=32, epochs=50, validation_data=(X_val, y_val))
build_and_evaluate_model(model, history, "NN with Adam & SMOTE")

# Compare performance and finalize the best model
print("\nTraining performance comparison:")
print(train_metric_df)
print("\nValidation performance comparison:")
print(valid_metric_df)
print("\nDifference between train and validation metrics:")
print(train_metric_df - valid_metric_df)

# Test set evaluation
y_test_pred = model.predict(X_test) > 0.5
print("\nTest Set Classification Report:")
print(classification_report(y_test, y_test_pred))
make_confusion_matrix(y_test, y_test_pred)
