# 🤖 Autonomous Data Science Co-Pilot

An AI agent that behaves like a junior data analyst: upload a CSV / Excel / JSON file, ask a
question in plain English, and it autonomously writes Python/Pandas code, runs it in a
sandbox, self-corrects errors using live RAG over the official Python & Pandas docs, and
delivers a chart + a plain-English insight.

## Project structure

```
├── Autonomous_DataScience_Copilot.ipynb   # Prototyping notebook (Colab)
├── app.py                                 # Deployable Streamlit web app (standalone)
├── requirements.txt                       # Python dependencies for app.py
└── README.md
```

The notebook and the Streamlit app are **independent implementations** of the same pipeline —
the notebook is for prototyping/demoing, the app is the deployable product. They intentionally
don't import from each other.

## How it works

1. **Load** — reads CSV / Excel / JSON into a pandas DataFrame and profiles its schema.
2. **Generate** — an LLM (OpenAI GPT-4o-mini by default) writes Python/Pandas/Matplotlib code
   to answer the user's question, given the data profile.
3. **Execute** — the code runs in a restricted namespace (limited builtins, no file/network access
   from inside the generated code).
4. **Self-heal (RAG)** — if execution fails, the agent live-searches `docs.python.org` and
   `pandas.pydata.org` for the error, fetches the top pages, TF-IDF-reranks the text against the
   error message to find the single most relevant chunk, and feeds that back to the LLM to fix
   its own code. This repeats up to 4 times.
5. **Deliver** — once code runs cleanly, the chart (`fig`) and a plain-English `insight_text`
   are shown to the user, along with the generated code for transparency.

## Setup

### Notebook (Google Colab)
1. Open `Autonomous_DataScience_Copilot.ipynb` in Colab.
2. Run cells top to bottom. Cell 3 will prompt for your OpenAI API key (not stored in the notebook).
3. Cell 4 opens a file picker — upload your CSV/Excel/JSON.
4. Cell 12 runs a sample question; edit the question string to try your own.

### Streamlit app (local or GitHub → Streamlit Cloud)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Enter your OpenAI API key in the sidebar, upload a file, type a question, click **Analyze**.

### Deploying to a public webpage
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and point it at your repo's
   `app.py`. It will build from `requirements.txt` and give you a public URL.
3. Add your OpenAI key as a Streamlit "Secret" (`OPENAI_API_KEY`) rather than typing it in each
   time, if you want the deployed app pre-configured — or leave the sidebar field as-is for
   users to bring their own key.

## Use cases covered

| # | Use case | Example question |
|---|----------|-------------------|
| 1 | Sales Dashboard | "Show revenue by region as a bar chart" |
| 2 | Data Quality Audit | "Are there missing values, outliers, or duplicates?" |
| 3 | Trend Analysis | "Is my traffic growing over time?" |
| 4 | Cohort Analysis | "Segment customers by total spend" |
| 5 | Ad-hoc Queries | "What's the average of each numeric column?" |

## Tech stack

- **UI**: Streamlit
- **Agent/LLM**: OpenAI GPT-4o-mini (swap for Anthropic/Gemini by editing the client init)
- **Code execution**: restricted `exec()` sandbox
- **RAG**: live web search (DuckDuckGo HTML) restricted to official docs + TF-IDF reranking
- **Data processing**: Pandas, NumPy, OpenPyXL
- **Visualization**: Matplotlib, Seaborn
