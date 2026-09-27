"""Step 9 – Evaluate the generated summary against a human reference (BLEU + ROUGE)."""
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def _tok(text):
    try:
        return nltk.word_tokenize(text.lower())
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        return nltk.word_tokenize(text.lower())


def bleu_scores(candidate: str, reference: str) -> dict:
    ref, cand = [_tok(reference)], _tok(candidate)
    sm = SmoothingFunction().method1
    weights = {"BLEU-1": (1, 0, 0, 0), "BLEU-2": (0.5, 0.5, 0, 0),
               "BLEU-3": (1/3, 1/3, 1/3, 0), "BLEU-4": (0.25, 0.25, 0.25, 0.25)}
    return {k: round(sentence_bleu(ref, cand, weights=w, smoothing_function=sm), 4)
            for k, w in weights.items()}


def rouge_scores(candidate: str, reference: str) -> dict:
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r = sc.score(reference, candidate)
    return {k: round(v.fmeasure, 4) for k, v in r.items()}


def evaluate(candidate: str, reference: str) -> dict:
    return {"bleu": bleu_scores(candidate, reference), "rouge": rouge_scores(candidate, reference),
            "candidate_words": len(candidate.split()), "reference_words": len(reference.split())}
