🔐 NLP-Based Password Strength Classification
📌 Project Overview

This project builds a machine learning model to evaluate password strength using Natural Language Processing (NLP) techniques.
Passwords are treated as text sequences, and character-level features are extracted to classify passwords into weak, medium, or strong categories.

The project demonstrates:

Text preprocessing on non-traditional NLP data (password strings)

Feature extraction using TF-IDF

Supervised classification with multiple ML models

Model evaluation and comparison

🧠 Problem Statement

Weak passwords are a major cybersecurity risk. Traditional rule-based approaches (length, symbols, etc.) are limited.

Goal:
Use NLP and machine learning to learn patterns in password strings and automatically classify their strength.

📂 Dataset

Source: SQLite database (password_data.sqlite)

Table: Users

Columns include:

password

strength (target label)

Target Classes:

0 → Weak

1 → Medium

2 → Strong

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

SQLite

Matplotlib / Seaborn

NLP (TF-IDF, character n-grams)

⚙️ Methodology
1️⃣ Data Loading

Connected to a SQLite database

Loaded password records into a Pandas DataFrame

2️⃣ Exploratory Data Analysis (EDA)

Class distribution analysis

Password length patterns

Visualization of strength imbalance

3️⃣ Text Preprocessing

Passwords treated as raw text

Character-level tokenization

No stopword removal (important for password semantics)

4️⃣ Feature Engineering

TF-IDF Vectorization

Character n-grams capture:

Repetition

Character diversity

Structural complexity

5️⃣ Model Training

Models trained and compared:

Logistic Regression

Random Forest

Support Vector Machine (SVM)

6️⃣ Model Evaluation

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

📊 Results

Character-level TF-IDF performed effectively for password strength detection

Strong passwords show higher entropy and diverse character patterns

Tree-based and linear models both performed competitively

🚀 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/USERNAME/NLP-password-strength.git
cd NLP-password-strength

2️⃣ Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

3️⃣ Open Jupyter Notebook
jupyter notebook NLP_password_strength.ipynb

4️⃣ Run all cells
📈 Key Learnings

NLP techniques can be applied beyond traditional text (emails, reviews)

Character-level modeling is powerful for security-related problems

TF-IDF + classical ML models can outperform simple rule-based systems

🔮 Future Improvements

Deep learning models (LSTM / CNN)

Password entropy & rule-based hybrid features

Real-time password strength API

Class imbalance handling (SMOTE)