import re
import nltk

# Download stopwords once (safe to run multiple times)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

# These words carry sentiment meaning — don't remove them
KEEP_WORDS = {'not', 'no', 'never', 'very', 'too', 'but', 'however'}
STOPWORDS = STOPWORDS - KEEP_WORDS


def clean_text(text):
    """
    Full text cleaning pipeline.
    Input : raw review string
    Output: cleaned lowercase string
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags and entities
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'&[a-z]+;', ' ', text)

    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # 4. Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Remove stopwords (but keep sentiment words)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    return ' '.join(tokens)


if __name__ == "__main__":
    # Quick test
    test_reviews = [
        "This product is AMAZING!!! Best purchase ever <br> 5/5 stars",
        "Not good at all. Very disappointing &amp; waste of money.",
        "It's okay... nothing special but not terrible either."
    ]

    print("Testing clean_text function:\n")
    for review in test_reviews:
        print(f"  Original : {review}")
        print(f"  Cleaned  : {clean_text(review)}")
        print()