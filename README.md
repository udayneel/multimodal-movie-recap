# Multimodal Movie Recap Generator

**AI11 | Semester 3 | Module 08 — Multimodal AI Systems: Speech, Text, and Conversational Agents**
REVA Academy for Corporate Excellence · Assignment 1 · Mentor: Dr. J B Simha

## 1. Project overview and objective

This project takes a full-length film from YouTube and automatically produces:

1. a time-stamped **transcript** (speech-to-text),
2. a **summary story** of the film,
3. the **peak points** of the film with their **time indices**,
4. a **movie recap video of less than 10 minutes** built from the original footage, and
5. a **BLEU / ROUGE evaluation** of the generated summary against a human-written reference.

The video, audio and text streams are all used, both to understand the story and to decide which moments matter most.

## 2. Proposed solution and architecture

```
YouTube URL
   │  yt-dlp
   ▼
Input video ──────────────┬───────────────────────────────┐
   │ ffmpeg               │ OpenCV (frame sampling)       │
   ▼                      ▼                               │
Audio (16 kHz WAV)     Visual activity signal             │
   │ faster-whisper       (frame differencing)            │
   ▼                      │                               │
Transcript + timestamps   │                               │
   │ LLM (map-reduce)     │                               │
   ▼                      │                               │
Summary story             │                               │
   │                      │                               │
   ▼                      ▼                               │
Multimodal peak scoring:  text relevance + audio energy + visual activity
   ▼
Peak points ──► Time index table (summary sentence ↔ timestamps)
   ▼
Segment extraction (OpenCV frames + ffmpeg audio mux)
   ▼
Movie recap (< 10 min)            Summary ──► BLEU / ROUGE vs reference
```

### Workflow in detail

| Step | Module | What happens |
|---|---|---|
| 1. Input | `src/download.py` | `yt-dlp` downloads the film at ≤480p (enough for analysis, fast to process). |
| 2. Audio + video | `src/media.py` | `ffmpeg` extracts mono 16 kHz audio; OpenCV reads video properties. |
| 3. Speech-to-text | `src/transcribe.py` | `faster-whisper` with VAD produces segments with start/end timestamps. |
| 4. Summary story | `src/summarize.py` | Map-reduce summarization: each transcript chunk is summarized, then the partial summaries are merged into one chronological story (Gemini API, or local BART). |
| 5. Peak scoring | `src/peaks.py` | The transcript is grouped into ~30 s windows. Each window gets a **text** score (cosine similarity between its dialogue and the summary sentences, using sentence-transformer embeddings), an **audio** score (95th-percentile RMS energy) and a **visual** score (mean frame difference from OpenCV). Scores are min-max normalised and fused: `0.6·text + 0.2·audio + 0.2·visual`. |
| 6. Peak selection + time index | `src/peaks.py` | **Coverage first**: each summary sentence is mapped to its best-matching window, so every part of the story appears in the recap. **Then fill**: the highest fused-score windows are added until the 9.5-minute budget is reached. Selected windows are ordered chronologically. |
| 7–8. Extraction + recap | `src/recap.py` | OpenCV seeks to each timestamp and writes the frames of the segment; since OpenCV has no audio, the matching audio is cut from the WAV and muxed with ffmpeg; all clips are concatenated into `movie_recap.mp4`. |
| 9. Evaluation | `src/evaluate.py` | BLEU-1…4 (NLTK, with smoothing) and ROUGE-1/2/L against a reference summary. |
| Reports | `src/report.py` | CSV tables, a markdown run report and an importance-timeline plot. |

## 3. Technologies, models and libraries

| Purpose | Tool / model |
|---|---|
| Video download | yt-dlp |
| Audio extraction, muxing, concatenation | FFmpeg |
| Frame reading, visual-activity analysis, segment extraction | OpenCV |
| Speech-to-text | faster-whisper (OpenAI Whisper, CTranslate2 backend) |
| Summarization | Google Gemini (`gemini-3.8-flash`) or `facebook/bart-large-cnn` |
| Semantic similarity | `sentence-transformers/all-MiniLM-L6-v2` (TF-IDF fallback) |
| Audio features | NumPy / SciPy (RMS energy) |
| Evaluation | NLTK BLEU, `rouge-score` |
| Plots | Matplotlib |

## 4. Project structure

```
movie-recap/
├── main.py                  # end-to-end pipeline
├── config.py                # all settings (model sizes, weights, recap budget)
├── requirements.txt
├── Movie_Recap_Colab.ipynb  # run everything on Google Colab (GPU)
├── src/
│   ├── download.py  media.py  transcribe.py  summarize.py
│   ├── peaks.py     recap.py  evaluate.py    report.py
├── tests/test_components.py # unit tests for each stage (synthetic video)
├── data/reference_summary.txt
└── outputs/                 # generated results
```

## 5. Setup and running

**Recommended: Google Colab with a T4 GPU** — `Movie_Recap_Colab.ipynb` runs end to end with *Runtime → Run all*:

1. Download the film locally (YouTube blocks downloads from cloud IPs such as Colab with "Sign in to confirm you're not a bot"):
   `yt-dlp -f "bv*[height<=480]+ba/b" --merge-output-format mp4 -o film.mp4 "<YouTube URL>"`
2. In Google Drive create `My Drive/movie-recap-work/` and upload `movie-recap.zip` and `film.mp4` (film may also sit directly in My Drive).
3. Add `GEMINI_API_KEY` in Colab Secrets (Notebook access on), paste the reference summary into section 4 of the notebook, and Run all.

The notebook converts the film once to H.264 (YouTube serves AV1, which OpenCV's bundled FFmpeg cannot decode; `src/media.py` also does this automatically), and writes all outputs to `movie-recap-work/outputs` in Drive. Transcript and summary are cached there, so re-runs skip Whisper and the LLM call; a cached transcript that does not cover the whole film is detected and redone.

**Local:**

```bash
sudo apt install ffmpeg
pip install -r requirements.txt
export GEMINI_API_KEY=your_key        # or: export SUMMARIZER=bart
pytest -q                             # test each component first
python main.py --source path/to/film.mp4 --reference data/reference_summary.txt
```

## 6. Input and output

**Input:** a YouTube URL (or local video file) of a narrative film, plus a reference summary text for evaluation.

**Output (`outputs/`):**

| File | Content |
|---|---|
| `transcript.txt / .json` | Full time-stamped transcript |
| `summary_story.txt` | Generated summary story |
| `summary_to_timestamps.csv` | Each summary sentence and the time index it maps to |
| `peak_points.csv` | Selected peak points with text/audio/visual/fused scores and clip file |
| `importance_timeline.png` | Scores over the whole film with the recap segments highlighted |
| `clips/clip_XX.mp4` | Each extracted segment |
| `movie_recap.mp4` | Final recap (< 10 minutes) |
| `evaluation.json` | BLEU-1…4 and ROUGE scores |
| `report.md` | Human-readable run report |

## 7. Results (run on *Night of the Living Dead*, 1968)

| Item | Result |
|---|---|
| Input film | 95.3 min, 640×480 @ 30 fps |
| Transcript | 496 segments (faster-whisper `small`), speech up to 93.1 min |
| Summary story | 299 words, Gemini `gemini-3.8-flash`, single long-context call |
| Scored windows | 90 windows of ~30 s (text + audio + visual) |
| Peak points used | 16 |
| **Recap duration** | **9.39 min** (limit: 10 min) |
| Recap video | [movie_recap.mp4 (Google Drive)](https://drive.google.com/file/d/1dO-NQMwH3KDEArRZ-OYHV40GF8aZ1af5/view?usp=sharing) |

**Evaluation against the reference summary (549 words):**

| Metric | Score |
|---|---|
| BLEU-1 | 0.2646 |
| BLEU-2 | 0.1566 |
| BLEU-3 | 0.0878 |
| BLEU-4 | 0.0553 |
| ROUGE-1 (F1) | 0.4354 |
| ROUGE-2 (F1) | 0.1260 |
| ROUGE-L (F1) | 0.2608 |

Detailed per-run outputs (summary-to-timestamp table, peak-point scores, importance timeline, full report) are in `outputs/`.

## 8. Discussion

**Why BLEU is low but the summary is good.** BLEU counts exact n-gram overlap. An abstractive summary retells the story in its own words, so 4-gram matches are rare even when every plot point is covered; BLEU-4 of about 0.05 is typical for abstractive summaries. The generated summary is also about half the reference length (299 vs 549 words), which adds BLEU's brevity penalty. ROUGE-1 of 0.44 shows that most key content words are shared, and a manual check confirms the summary covers all major plot points in order (cemetery attack, farmhouse siege, cellar survivors, radio reports, truck fire, Karen's reanimation, Ben's death).

**Why multimodal peak detection.** Dialogue alone misses action scenes with little speech; loudness alone picks up music and noise; visual change alone picks up camera cuts. Fusing the three, and first anchoring every summary sentence to its best-matching moment, gives a recap that follows the story and still includes intense scenes.

**Limitations.**
- The film is well known, so the LLM may partly rely on prior knowledge of it rather than only the transcript, which can make the summary look better than the transcript alone justifies. Testing on a lesser-known film would give a fairer estimate.
- Whisper produces no speaker labels, so the summary relies on names mentioned in dialogue.
- Windows are fixed ~30 s groups of transcript segments, not true scene boundaries; shot-boundary detection would give cleaner cut points.
- Fusion weights (0.6 / 0.2 / 0.2) were set by judgement, not learned.
- BLEU with a single reference is a weak measure of summary quality.

**Engineering issues solved.** YouTube blocks downloads from Colab ("Sign in to confirm you're not a bot"), so the film is downloaded locally and read from Drive. YouTube serves AV1 video, which OpenCV's bundled FFmpeg cannot decode, so the input is normalised to H.264. Newer `transformers` removed the `summarization` pipeline, so BART is called directly via `generate()`. Free-tier Gemini rate limits are handled with retry and back-off, and the whole transcript is summarised in one call.

## 9. Input film and references

- **Input film:** *Night of the Living Dead* (1968), dir. George A. Romero — public domain — https://www.youtube.com/watch?v=H91BxkBXttE
- **Reference summary:** "Plot" section of *Night of the Living Dead*, Wikipedia (CC BY-SA 4.0) — https://en.wikipedia.org/wiki/Night_of_the_Living_Dead
- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper), 2022
- Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training*, 2019
- Reimers & Gurevych, *Sentence-BERT*, 2019
- Papineni et al., *BLEU: a Method for Automatic Evaluation of Machine Translation*, 2002
- Lin, *ROUGE: A Package for Automatic Evaluation of Summaries*, 2004

## Author

Uday — SRNR25TA028 — MTech AI, REVA Academy for Corporate Excellence (Batch AI11)
