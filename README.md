# Haochen-Mitre-Attack-data-2024

# Automated Cybersecurity Alert Classification and MITRE ATT&CK Tagging Using Explainable Machine Learning

## Project Definition
A supervised multi-class classification problem:

**"Given a cybersecurity alert description, can we automatically classify it into the correct MITRE ATT&CK technique or tactic?"**

In other words:
- Can we train a machine learning model to understand and classify natural language security alerts?
- Can the model predict the correct attack ID or label (e.g., T1040, T1059) based on the text of the alert?
- Can we do this even with imbalanced and partially labeled data, and provide explanations for the model's decisions?

## Input Attributes
The machine learning pipeline expects preprocessed features derived from the alert data:

✅ **Required Input Attribute**:
- `alert`: A free-text string that describes the alert or incident (used in TfidfVectorizer)

🧠 **Features created from alert**:
- TF-IDF vector (term importance values for each alert)
- Cosine similarity matrix (compares each alert to the training set)
- (Optional) Dimensionality-reduced vectors using TruncatedSVD or PCA
- (Optional) Additional metadata: alert severity, source, IP info (if available)

🎯 **Target Attribute**:
- `id`: A label that corresponds to a MITRE ATT&CK technique ID or custom tag (e.g., T1059, T1203, T1566)

## Installation
```bash
pip install numpy scipy matplotlib ipython scikit-learn pandas jupyter tensorflow tensorboard joblib mglearn \
attackcti stix2 nltk seaborn imblearn imbalanced-learn spacy flaml[automl] catboost lightgbm wordcloud
Usage
1. Import Required Libraries
python
from attackcti import attack_client
from stix2 import MemoryStore, AttackPattern, Identity, Relationship
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import pickle
2. Load and Prepare Data
python
from google.colab import drive
import pandas as pd

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to your CSV files
base_path = '/content/drive/MyDrive/Colab Notebooks/Data/'

# Load the CSV files
data = pd.read_csv(base_path + 'tagged_alerts.csv')
data.dropna(inplace=True)

alerts = pd.read_csv(base_path + 'alerts.csv')
alerts.dropna(inplace=True)
3. Get MITRE ATT&CK Techniques
python
lift = attack_client()
all_techniques = lift.get_techniques()

def get_mitre_techniques():
    mitre_techniques_list = []
    for technique in all_techniques:
        if technique['external_references'] and technique['description']:
            for ref in technique['external_references']:
                if 'external_id' in ref and '.' not in ref['external_id']:
                    mitre_techniques_list.append((ref['external_id'], technique['name'],technique.description))
                    break
    return mitre_techniques_list

mitre_techniques = get_mitre_techniques()
4. Train and Evaluate the Model
python
# Split data
descriptions = data['alert'].tolist()
true_labels = data['id'].tolist()
descriptions_train, descriptions_test, labels_train, labels_test = train_test_split(
    descriptions, true_labels, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english')
descriptions_tfidf = vectorizer.fit_transform(descriptions_train)

# Match alerts to MITRE techniques
def match_alert_to_mitre(alert, vectorizer, descriptions_tfidf, labels_train):
    alert_tfidf = vectorizer.transform([alert])
    cosine_sim = cosine_similarity(alert_tfidf, descriptions_tfidf)
    max_index = cosine_sim.argmax()
    return labels_train[max_index]

# Evaluate model
predictions = [match_alert_to_mitre(alert, vectorizer, descriptions_tfidf, labels_train) for alert in descriptions_test]
correct_predictions = sum(1 for true, pred in zip(labels_test, predictions) if true == pred)
accuracy = correct_predictions / len(labels_test)
print(f"Accuracy: {accuracy:.2f}")
5. Save the Model
python
model_filename = 'current_best_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump({
        'model': best_rf_model,
        'vectorizer': vectorizer
    }, model_file)
Results
The model achieves:

Accuracy: 0.62

Top 5 AUC Classes:

Class: T1559, AUC: 1.00

Class: T1123, AUC: 0.96

Class: T1083, AUC: 0.90

Class: T1132, AUC: 0.90

Class: T1197, AUC: 0.90

Auto-Tagging New Alerts
python
def auto_tag_alerts(df, vectorizer, descriptions_tfidf, labels_train):
    df['Predicted MITRE ID'] = df['alert'].apply(lambda x: match_alert_to_mitre(x, vectorizer, descriptions_tfidf, labels_train))
    return df

tagged_df = auto_tag_alerts(alerts, vectorizer, descriptions_tfidf, labels_train)
License
This project is licensed under the MIT License - see the LICENSE file for details.


This README includes:
1. Clear project definition and goals
2. Installation instructions
3. Usage examples with code blocks
4. Results section with key metrics
5. Auto-tagging functionality
6. Proper formatting for GitHub/Markdown

The code blocks are properly formatted and the overall structure makes it easy to understand and use the project.
