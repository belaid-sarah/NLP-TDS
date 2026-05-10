import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification


def make_model():
    return Pipeline(
        [
            ("count_vectorizer", CountVectorizer(min_df=5)),
            ("random_forest", RandomForestClassifier(n_estimators=100, min_samples_split=10)),
        ]
    )


def make_model_old():
    return MyModel()


class MyModel():
    def __init__(self):
        self._vectorizer = CountVectorizer()
        self._clf = RandomForestClassifier()

    def fit(self, X, y):
        X = self._vectorizer.fit_transform(X)
        self._clf.fit(X, y)

        return self

    def predict(self, X):
        X = self._vectorizer.transform(X)

        return self._clf.predict(X)


def predict_at_word_level(model, tokenizer, text: str) -> list:
    """
    Run a TokenClassification model and return word-level predictions,
    collapsing sub-word tokens back to the original words.

    Args:
        model:     AutoModelForTokenClassification (HuggingFace)
        tokenizer: matching tokenizer
        text:      raw sentence (words separated by spaces)

    Returns:
        list of {"word": str, "label": str}

    Example:
        >>> preds = predict_at_word_level(model, tokenizer, "Ask the python teacher when is the next class")
        >>> preds
        [{"word": "Ask",     "label": "O"},
         {"word": "the",     "label": "B-person"},
         {"word": "python",  "label": "I-person"},
         {"word": "teacher", "label": "I-person"},
         {"word": "when",    "label": "B-content"},
         ...]
    """
    words = text.split()
    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    word_ids = enc.word_ids()

    with torch.no_grad():
        logits = model(**enc).logits[0]

    pred_ids = logits.argmax(-1).tolist()
    id2label = model.config.id2label

    # Keep only the first sub-token prediction for each word
    word_preds: dict[int, str] = {}
    for tok_idx, wid in enumerate(word_ids):
        if wid is not None and wid not in word_preds:
            word_preds[wid] = id2label[pred_ids[tok_idx]]

    return [{"word": w, "label": word_preds.get(i, "O")} for i, w in enumerate(words)]
