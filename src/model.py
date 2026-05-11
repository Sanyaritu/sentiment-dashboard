import pandas as pd
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load cleaned data ──────────────────────────────────
print("Loading cleaned data...")
df = pd.read_csv('data/reviews_clean.csv')
print(f"Shape: {df.shape}")

# ── 2. Prepare X and y ───────────────────────────────────
X = df['clean_text'].astype(str)
y = df['sentiment']

# ── 3. Train/test split ──────────────────────────────────
# 80% train, 20% test — stratify keeps class balance in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── 4. Build pipeline ────────────────────────────────────
# TF-IDF converts text → numbers
# ngram_range=(1,2) means it looks at single words AND word pairs
# "not good" as a pair is more meaningful than "not" and "good" separately
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True
)),
    ('clf', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
        random_state=42
    ))
])

# ── 5. Train ─────────────────────────────────────────────
print("\nTraining model... (takes ~1 minute)")
pipeline.fit(X_train, y_train)
print("Training complete!")

# ── 6. Evaluate ──────────────────────────────────────────
y_pred = pipeline.predict(X_test)

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred))

# ── 7. Confusion matrix ──────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=['Positive', 'Neutral', 'Negative'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Positive', 'Neutral', 'Negative'],
            yticklabels=['Positive', 'Neutral', 'Negative'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# ── 8. Save model ────────────────────────────────────────
os.makedirs('model', exist_ok=True)
with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("\n✅ Model saved to model/sentiment_model.pkl")