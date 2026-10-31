"""Train and evaluate the intent classifier (multilingual embeddings + calibrated LinearSVC)."""

import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.nlp.normalizer import TextNormalizer  # noqa: E402

DATASET = ROOT / "data" / "annotations" / "seed_dataset.jsonl"
MODELS_DIR = ROOT / "data" / "models"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
SEED = 42


def load_dataset(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (texts, intents, languages) from a JSONL file; fails loudly on malformed rows."""
    texts, intents, langs = [], [], []
    normalizer = TextNormalizer()
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("text", "intent", "language"):
                if not row.get(key):
                    raise ValueError(f"{path}:{n}: missing '{key}'")
            texts.append(normalizer.normalize(row["text"]))
            intents.append(row["intent"])
            langs.append(row["language"])
    return texts, intents, langs


def format_confusion(matrix: np.ndarray, labels: list[str]) -> str:
    """Render a confusion matrix (rows = true, columns = predicted) as an ASCII table."""
    short = [lbl[:10] for lbl in labels]
    width = max(len(s) for s in short) + 1
    header = " " * width + "".join(f"{s:>{width}}" for s in short)
    lines = [header]
    for name, row in zip(short, matrix):
        lines.append(f"{name:<{width}}" + "".join(f"{int(v):>{width}}" for v in row))
    return "\n".join(lines)


def main() -> int:
    from sentence_transformers import SentenceTransformer

    texts, intents, langs = load_dataset(DATASET)
    print(f"Loaded {len(texts)} examples: {dict(Counter(intents))}")

    encoder = LabelEncoder().fit(intents)
    y = encoder.transform(intents)
    idx = np.arange(len(texts))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    # cv=4: with ~16 training examples per class, 5 folds would leave folds of 3 samples.
    classifier = CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=4)
    classifier.fit(embeddings[train_idx], y[train_idx])

    pred = classifier.predict(embeddings[test_idx])
    labels = list(encoder.classes_)
    accuracy = float(accuracy_score(y[test_idx], pred))
    report = classification_report(
        y[test_idx], pred, labels=range(len(labels)), target_names=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y[test_idx], pred, labels=range(len(labels)))

    per_language: dict[str, dict[str, float]] = {}
    for lang in sorted(set(langs)):
        sel = [i for i, t in enumerate(test_idx) if langs[t] == lang]
        if sel:
            hits = sum(int(pred[i] == y[test_idx][i]) for i in sel)
            per_language[lang] = {"n": len(sel), "accuracy": round(hits / len(sel), 3)}

    errors = [
        {"text": texts[t], "true": labels[y[t]], "predicted": labels[p]}
        for t, p in zip(test_idx, pred)
        if y[t] != p
    ]

    print(f"\nTrain: {len(train_idx)}  Test: {len(test_idx)}")
    print(f"Accuracy: {accuracy:.3f}\n")
    print(classification_report(y[test_idx], pred, labels=range(len(labels)), target_names=labels, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(format_confusion(matrix, labels))
    print("\nAccuracy by language:", per_language)
    print("Misclassified test examples:")
    for e in errors:
        print(f"  {e['true']:>15} -> {e['predicted']:<15} {e['text']}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "intent_evaluation.json").write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "labels": labels,
                "classification_report": report,
                "confusion_matrix": matrix.tolist(),
                "accuracy_by_language": per_language,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Final model: refit on all data so the shipped classifier uses every example.
    final = CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=4).fit(embeddings, y)
    joblib.dump(final, MODELS_DIR / "intent_classifier.pkl")
    joblib.dump(encoder, MODELS_DIR / "intent_labels.pkl")
    (MODELS_DIR / "intent_config.json").write_text(
        json.dumps({"embedding_model": EMBEDDING_MODEL, "labels": labels}, indent=2), encoding="utf-8"
    )
    print(f"\nSaved artifacts to {MODELS_DIR} (final model refit on all {len(texts)} examples;"
          " reported metrics come from the held-out split).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
