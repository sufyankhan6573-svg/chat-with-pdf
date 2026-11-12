# 📚 Advanced Chat & Study Hub

A Streamlit app that lets you upload multiple PDFs (lecture decks, notes, books) and then chat with them, generate practice quizzes, and create summaries, powered by Google Gemini (`gemini-3.8-flash`).

## Features
- **Multi-PDF upload** with total documents and character count
- **💬 Chat & Q&A** that answers only from your documents, with source and page citations
- **📝 Quiz & Practice** with 1 to 20 questions, three question types (MCQ, True/False, Open-ended) and three difficulty levels
- **📌 Summary Generator** with one-click structured summaries
- **Chat export** to `study_chat_history.txt`
- **Automatic retry** on Gemini 503 high-demand errors

## Tech Stack
- Python, Streamlit
- `google-genai` (official Google Gen AI SDK)
- `pypdf` for PDF text extraction

## Project Structure
```
├── app.py
├── requirements.txt
└── README.md
```

## Run Locally
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-api-key"
```

Start the app:
```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt` and `README.md` to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-api-key"
   ```
5. Click **Deploy**.

> Never commit `.streamlit/secrets.toml` to GitHub. Add it to `.gitignore`.

## Configuration
In `app.py`, `MAX_CONTEXT_CHARS` controls how much document text is sent to the AI (default `15000`). Increase it for long documents.

## Limitations
- Scanned or image-only PDFs have no extractable text (OCR is not included).
- Only the first `MAX_CONTEXT_CHARS` characters are used by the AI.
