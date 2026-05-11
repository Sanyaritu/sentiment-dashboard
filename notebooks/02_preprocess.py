import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import clean_text

print("Loading sample dataset...")
df = pd.read_csv('../data/reviews_sample.csv')
print(f"Loaded: {df.shape}")

print("\nCleaning text... (this takes ~1-2 minutes)")
df['clean_text'] = df['Text'].apply(clean_text)

# Remove any empty rows after cleaning
df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)

print(f"After cleaning: {df.shape}")

# Preview
print("\nSample before/after cleaning:")
for i in range(3):
    print(f"\n  Raw     : {df['Text'].iloc[i][:100]}...")
    print(f"  Cleaned : {df['clean_text'].iloc[i][:100]}...")

# Save
df.to_csv('../data/reviews_clean.csv', index=False)
print("\n✅ Saved reviews_clean.csv")