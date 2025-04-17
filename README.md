# Can EEG Data Aid in the Clinical Diagnosis of Major Depressive Disorder (MDD)?
**A Predictive Machine Learning Approach Using Python (MNE, Antropy, SciPy)**

## 🧠 Introduction

Major Depressive Disorder (MDD) is one of the most prevalent mental health conditions, contributing significantly to the global burden of disease. As of 2015, over 322 million people were diagnosed with depression worldwide. Currently, MDD diagnosis primarily relies on psychological assessments and interviews conducted by trained professionals—an approach that is time-intensive and subjective.

With the rise of AI and data-driven tools, physiological signals like EEG (Electroencephalography) are being explored as potential diagnostic indicators. EEG offers non-invasive insights into brain activity with millisecond-level resolution, making it a promising candidate for objective mental health assessments.

## 🎯 Objective

This project explores how EEG data can be leveraged to detect signs of MDD using Python-based tools and machine learning models. Traditionally, EEG signal processing is done in MATLAB; here, we demonstrate how to replicate and extend this functionality using Python libraries.

**Key goals of the project:**

1. **Port MATLAB EEG processing to Python** using libraries such as MNE, SciPy, and StatsModels.
2. **Feature Engineering:** Extract both linear (direct signal metrics) and non-linear (entropy-based) features from EEG data.
3. **Build a Classification Model** to distinguish between MDD patients and healthy individuals.

---

## ⚙️ Requirements

The project requires Python 3.7+ and the following libraries:

- `pandas`, `numpy`, `scipy` – for data manipulation and signal processing  
- `matplotlib`, `seaborn` – for data visualization  
- `scikit-learn` – for building machine learning models  
- `mne` – for EEG signal processing  
- `antropy` – for extracting entropy-based features  

### Install Dependencies

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn mne antropy
