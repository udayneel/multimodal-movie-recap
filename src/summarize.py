"""Step 4 – Transcript -> Summary story (hierarchical / map-reduce summarization)."""
import os
import re

STORY_PROMPT = """You are given partial summaries of consecutive parts of a film, in order.
Write ONE coherent summary story of the whole film in 8-12 sentences of plain prose.
Keep chronological order, name the main characters, and cover the key turning points
and the ending. Do not use bullet points or headings.

Partial summaries:
{text}"""

CHUNK_PROMPT = """Below is part of a film's dialogue transcript (speaker names are not given).
Summarize what happens in this part in 3-5 sentences of plain prose, focusing on
events, conflicts and decisions.

Transcript:
{text}"""


FULL_PROMPT = """Below is the complete dialogue transcript of a film (speaker names are not given).
Write ONE coherent summary story of the whole film in 8-12 sentences of plain prose.
Keep chronological order, name the main characters, and cover the key turning points
and the ending. Do not use bullet points or headings.

Transcript:
{text}"""


def chunk_words(text: str, n: int) -> list:
    words = text.split()
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)]


def _gemini(prompt: str, model: str, retries: int = 8) -> str:
    import time
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return resp.text.strip()
        except Exception as e:
            # free tier: wait and retry on rate limits (429) / temporary errors (5xx)
            if attempt < retries - 1 and any(c in str(e) for c in ("429", "500", "503")):
                wait = 20 * (attempt + 1)
                print(f"[summarize] API busy/rate-limited, retrying in {wait}s")
                time.sleep(wait)
            else:
                raise


_bart = None
def _bart_summarize(text: str, model_name: str, max_len=160, min_len=50) -> str:
    """Local abstractive summarization with BART (no API needed).
    Uses the model directly (newer transformers versions dropped the
    'summarization' pipeline task)."""
    global _bart
    import torch
    if _bart is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
        _bart = (tok, model, device)
    tok, model, device = _bart
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, num_beams=4, max_length=max_len, min_length=min_len,
                             no_repeat_ngram_size=3, early_stopping=True)
    return tok.decode(ids[0], skip_special_tokens=True).strip()


def summarize(segments: list, cfg) -> dict:
    """Map step: summarize each chunk. Reduce step: merge into one story."""
    full_text = " ".join(s["text"] for s in segments)

    if cfg.SUMMARIZER == "gemini" and len(full_text.split()) <= getattr(cfg, "GEMINI_MAX_WORDS", 200000):
        # long-context LLM: the whole transcript fits in one request
        chunks, partials = [full_text], []
        story = _gemini(FULL_PROMPT.format(text=full_text), cfg.GEMINI_MODEL)
    elif cfg.SUMMARIZER == "gemini":
        # very long transcripts: map-reduce over chunks
        chunks = chunk_words(full_text, cfg.CHUNK_WORDS)
        partials = [_gemini(CHUNK_PROMPT.format(text=c), cfg.GEMINI_MODEL) for c in chunks]
        story = _gemini(STORY_PROMPT.format(text="\n\n".join(partials)), cfg.GEMINI_MODEL)
    else:
        chunks = chunk_words(full_text, cfg.BART_CHUNK_WORDS)
        partials = [_bart_summarize(c, cfg.BART_MODEL, 120, 30) for c in chunks]
        merged = " ".join(partials)
        # second pass until the text fits BART's window
        while len(merged.split()) > cfg.BART_CHUNK_WORDS:
            merged = " ".join(_bart_summarize(c, cfg.BART_MODEL, 120, 30)
                              for c in chunk_words(merged, cfg.BART_CHUNK_WORDS))
        story = _bart_summarize(merged, cfg.BART_MODEL, 250, 100)

    print(f"[summarize] {len(chunks)} chunks -> story of {len(story.split())} words")
    return {"story": story, "chunk_summaries": partials}


def split_sentences(text: str) -> list:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.split()) >= 3]
