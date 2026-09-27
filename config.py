"""Central configuration for the Multimodal Movie Recap pipeline."""
import os

# ---------- Input ----------
# A YouTube URL or a path to a local video file.
# Use a public-domain film, e.g. "Night of the Living Dead" (1968).
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "")
REFERENCE_SUMMARY = os.getenv("REFERENCE_SUMMARY", "data/reference_summary.txt")

# ---------- Paths ----------
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
CLIP_DIR = os.path.join(OUTPUT_DIR, "clips")

# ---------- Speech-to-text ----------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")   # tiny/base/small/medium/large-v3
WHISPER_LANGUAGE = "en"

# ---------- Summarization ----------
# "gemini" uses the Gemini API (set GEMINI_API_KEY); "bart" runs locally.
SUMMARIZER = os.getenv("SUMMARIZER", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
BART_MODEL = "facebook/bart-large-cnn"
GEMINI_MAX_WORDS = 200000   # Gemini long context: whole transcript in one call
CHUNK_WORDS = 1500            # transcript words per summarization chunk (Gemini)
BART_CHUNK_WORDS = 600        # BART has a 1024-token input limit

# ---------- Peak-point detection ----------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WINDOW_SECONDS = 30           # transcript segments are grouped into ~30 s windows
VISUAL_SAMPLE_FPS = 2         # frames/sec sampled for visual-activity analysis
WEIGHTS = {"text": 0.6, "audio": 0.2, "visual": 0.2}   # multimodal fusion weights

# ---------- Recap ----------
MAX_RECAP_SECONDS = 9 * 60 + 30   # hard budget, safely under 10 minutes
CLIP_PADDING = 1.0                # seconds added before/after each clip
