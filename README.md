# Kuku AI Generalist Assignment

Microdrama promo extractor for TikTok, Instagram Reels, and YouTube Shorts.

## What It Does

- Upload one `.txt` screenplay or a `.zip` containing many episode `.txt` files
- Extract promo candidates with deterministic parsing and timeline windowing
- Score candidates with Gemini via `google-genai`
- Rank the global top 5 promos across the full batch
- Export results as JSON

## Setup

```bash
pip install -r requirements.txt
```

Create a local `.env` file from `.env.example`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
PROMO_EVAL_MODEL=gemini-3.5-flash-lite
```

## Run

```bash
streamlit run app.py
```

## Input Format

Supports screenplay text files in the project syntax, or a `.zip` containing many `.txt` episode files.

## Output

The app returns:

- top 5 global promos
- score breakdown
- source episode label
- transcript preview
- quality flags

## Files

- `app.py` - Streamlit UI
- `parser.py` - screenplay parsing
- `candidate_generator.py` - promo window generation
- `evaluator.py` - LLM scoring and fallback
- `ranker.py` - ranking and diagnostics
