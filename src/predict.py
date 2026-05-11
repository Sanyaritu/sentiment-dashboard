import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import clean_text

# ── Load model once when module is imported ───────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'sentiment_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

EMOJI = {
    'Positive': '😊',
    'Neutral':  '😐',
    'Negative': '😠'
}

COLOR = {
    'Positive': 'green',
    'Neutral':  'orange',
    'Negative': 'red'
}


def predict_sentiment(text: str) -> dict:
    """
    Takes raw review text, returns prediction dict.

    Returns:
        {
            'label':      'Positive' / 'Neutral' / 'Negative',
            'emoji':      '😊' / '😐' / '😠',
            'color':      'green' / 'orange' / 'red',
            'confidence': 0.85,
            'scores': {
                'Positive': 0.85,
                'Neutral':  0.10,
                'Negative': 0.05
            }
        }
    """
    cleaned = clean_text(text)

    if not cleaned.strip():
        return {
            'label': 'Neutral',
            'emoji': '😐',
            'color': 'orange',
            'confidence': 0.0,
            'scores': {'Positive': 0.0, 'Neutral': 1.0, 'Negative': 0.0}
        }

    label = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    classes = model.classes_

    scores = dict(zip(classes, proba))
    confidence = max(proba)

    return {
        'label':      label,
        'emoji':      EMOJI[label],
        'color':      COLOR[label],
        'confidence': round(confidence, 4),
        'scores':     {k: round(v, 4) for k, v in scores.items()}
    }


def predict_batch(texts: list) -> list:
    """
    Takes a list of raw texts, returns list of prediction dicts.
    More efficient than calling predict_sentiment() in a loop.
    """
    cleaned = [clean_text(t) for t in texts]
    labels = model.predict(cleaned)
    probas = model.predict_proba(cleaned)
    classes = model.classes_

    results = []
    for label, proba in zip(labels, probas):
        scores = dict(zip(classes, proba))
        results.append({
            'label':      label,
            'emoji':      EMOJI[label],
            'color':      COLOR[label],
            'confidence': round(max(proba), 4),
            'scores':     {k: round(v, 4) for k, v in scores.items()}
        })
    return results


if __name__ == "__main__":
    # Quick test
    test_reviews = [
        "This is the best coffee I have ever tasted. Absolutely love it!",
        "Terrible product. Arrived broken and customer service was useless.",
        "It's decent. Nothing special but does the job I guess."
    ]

    print("Testing predict_sentiment:\n")
    for review in test_reviews:
        result = predict_sentiment(review)
        print(f"  Review    : {review[:60]}...")
        print(f"  Sentiment : {result['emoji']} {result['label']} ({result['confidence']*100:.1f}% confident)")
        print(f"  Scores    : {result['scores']}")
        print()