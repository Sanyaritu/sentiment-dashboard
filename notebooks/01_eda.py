import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/Reviews.csv')
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing vaues:\n", df.isnull().sum())

df['Score'].value_counts().sort_index().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Review Score Distribution')
plt.xlabel('Star Rating')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

def label_sentiment(score):
    if score <= 2:
        return 'Negative'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Positive'
    
df['sentiment'] = df['Score'].apply(label_sentiment)
print("\nSentiment counts:\n", df['sentiment'].value_counts())

colors = ['#e74c3c', '#f39c12', '#2ecc71']
df['sentiment'].value_counts().plot(kind='bar', color=colors, edgecolor='black')
plt.title('Sentiment Class Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

for sentiment in ['Positive', 'Negative', 'Neutral']:
    print(f"\n{'='*50}")
    print(f"Sample {sentiment} review:")
    print(df[df['sentiment'] == sentiment]['Text'].iloc[0])

df['review_length'] = df['Text'].astype(str).apply(lambda x: len(x.split()))
print("\nReview length by sentiment:")
print(df.groupby('sentiment')['review_length'].describe())

sample = df.groupby('sentiment', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 10000), random_state=42)
)
sample = sample[['Text', 'Score', 'sentiment']].dropna().reset_index(drop=True)
sample.to_csv('../data/reviews_sample.csv', index=False)
print(f"\n✅ Saved reviews_sample.csv — Shape: {sample.shape}")
print(sample['sentiment'].value_counts())