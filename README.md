# 🧠 Leveraging EEG Signals for Clinical Detection of Major Depressive Disorder (MDD)
### *A Machine Learning-Based Diagnostic Approach Using Python (MNE, Antropy, SciPy)*

---

## 📝 Overview

Major Depressive Disorder (MDD) remains one of the most widespread and debilitating mental health conditions globally. Despite its prevalence, MDD is still diagnosed through subjective clinical interviews and questionnaires, which are often time-consuming and prone to bias.

This project explores the potential of **Electroencephalography (EEG)** as a diagnostic aid, utilizing advanced signal processing and machine learning to uncover biomarkers that can assist in early and objective detection of depression.

---

## 🎯 Project Goals

The core objectives of this project are:

- ✅ Replicate and adapt EEG processing techniques traditionally done in **MATLAB** to **Python**
- ✅ Utilize open-source Python libraries like **MNE**, **Antropy**, and **SciPy** for EEG analysis
- ✅ Engineer both **time-domain (linear)** and **complexity-based (non-linear)** features from raw EEG data
- ✅ Develop classification models to differentiate between healthy controls and individuals with varying levels of depressive symptoms

---

## 🔧 Tools & Technologies

| Purpose                 | Library        |
|-------------------------|----------------|
| EEG signal handling     | MNE            |
| Entropy & complexity    | Antropy        |
| Signal filtering        | SciPy          |
| Data manipulation       | Pandas, NumPy  |
| Modeling & evaluation   | Scikit-learn   |
| Visualization           | Matplotlib, Seaborn |

### 📦 Installation

```bash
pip install mne antropy numpy pandas scipy matplotlib seaborn scikit-learn

📊 Dataset Information
EEG recordings were provided in two experimental conditions:

Resting State — Captures baseline brain activity (Format: .mat)

Task-Induced (ERP) — Captures evoked brain responses under stimulation (Format: .raw)

🧪 Recording Setup:

Device: 128-channel HydroCel GSN system

Reference channel E129 was removed due to lack of relevance

📥 Data Loading & Configuration
python
Copy
Edit
# Load resting state (.mat)
import scipy.io
rest_data = scipy.io.loadmat("subject_rest.mat")

# Load task/ERP state (.raw)
import mne
raw = mne.io.read_raw_egi("subject_erp.raw", preload=True)

# Apply standard montage
montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
raw.set_montage(montage)
raw.drop_channels(['E129'])  # Remove unused reference channel
🔍 Signal Preprocessing
To reduce noise and artifacts, a bandpass filter (0.5 – 40 Hz) was applied using a Hamming window. Frequencies beyond this range often include muscular or ocular disturbances.

python
Copy
Edit
raw.filter(l_freq=0.5, h_freq=40)
🧾 Before and After Filtering:


✨ Feature Extraction
➤ Linear Features (via NumPy)
These are directly derived from signal statistics:

Mean / Median / Max / Min amplitude

Bandpower in delta, theta, alpha, and beta frequencies

➤ Non-Linear Features (via Antropy)
These capture complexity and irregularity in EEG signals:

SVD Entropy – Quantifies singular value distribution complexity

Spectral Entropy – Measures power distribution randomness across frequencies

Permutation Entropy – Evaluates pattern unpredictability in time-series

📚 References:

SVD Entropy

Spectral Entropy

Permutation Entropy

🧠 Classification Pipeline
Our predictive modeling approach involves:

Data Scaling – Normalizing features using StandardScaler

Train/Test Splitting – 80/20 division or cross-validation

Model Training – Using classifiers like Random Forest, SVM, or Logistic Regression

Performance Evaluation – Using metrics such as Accuracy, F1-score, and AUC

python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))
📈 Evaluation Metrics

Metric	What It Tells Us
Accuracy	Overall correctness of the model
Precision	Proportion of positive predictions that were correct
Recall	Proportion of actual positives correctly identified
F1-score	Harmonic mean of precision and recall
ROC-AUC	Trade-off between sensitivity and specificity
🚧 Challenges
EEG data is inherently noisy and non-stationary

Small datasets increase the risk of overfitting

Cross-subject generalization remains a hurdle

High intra-class variability due to brain dynamics

🚀 Future Enhancements
Integrate deep learning models (CNNs, LSTMs) for automated feature learning

Apply transfer learning from large EEG datasets

Explore real-time EEG classification for live applications

Conduct clinical trials for real-world validation

🗂 Suggested Directory Structure
graphql
Copy
Edit
.
├── data/                # Raw EEG data files
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Core Python scripts
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── model_train.py
├── results/             # Figures, metrics, outputs
├── README.md
└── requirements.txt
