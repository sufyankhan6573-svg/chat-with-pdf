# Chat with your PDF

A Streamlit app that uses a Google Gemini Flash model to work with a PDF you upload:

- **MCQs**: generate multiple-choice questions, take the quiz, get an automatic score with explanations and page references
- **Short questions**: generate practice questions with model answers
- **Chat**: ask questions about the PDF; answers cite page numbers and say so when the PDF doesn't contain the answer

You choose how many questions, the difficulty, and which page range to use. MCQs can be downloaded as a `.txt` file.

## Files

| File | Purpose |
|------|---------|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Run locally

1. Install Python 3.10+.
2. Get a free API key from https://aistudio.google.com/apikey
3. Run:

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"        # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

You can also paste the key into the sidebar instead of setting the variable.

## Configuration

| Name | Meaning | Default |
|------|---------|---------|
| `GEMINI_API_KEY` | Your Gemini API key (env var or Streamlit secret) | none |
| `GEMINI_MODEL` | Model ID to use | `gemini-3.6-flash` |

If the model name stops working, set `GEMINI_MODEL` to another current Flash model ID.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings > Secrets** and add:

```toml
GEMINI_API_KEY = "your-key"
```

5. Click **Deploy**.

**Never commit your API key to GitHub.**

## How it works

- Text is extracted page by page and labelled `[Page N]`, so the model can cite pages.
- The whole document (up to about 300,000 characters) is sent to Gemini with each request. There is no vector database, which keeps the app simple and works well for typical notes, papers and chapters. Very long books are truncated, and the app warns you.
- MCQ options are shuffled in code, so the correct answer is not always in the same position.
- Questions that fail validation (wrong number of options, duplicate options, invalid answer, duplicate question) are dropped.

## Limits

- Text-based PDFs only. Scanned/image-only PDFs are rejected because there is no text to read.
- Max file size 15 MB.
- AI-generated questions can contain mistakes. Check them against the source page before relying on them.
- PDF content is sent to Google's Gemini API; don't upload anything confidential.
