"""
Autonomous Data Science Co-Pilot — Streamlit Web App
------------------------------------------------------
Upload a CSV / Excel / JSON file, ask a question in plain English, and this
app will autonomously write, execute, and self-correct Python/Pandas code to
answer it — producing a chart and a plain-English insight.

This file is a standalone implementation (independent from the Colab
notebook) so the deployed web app has no dependency on the research notebook.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import contextlib
import traceback

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Data Science Co-Pilot", page_icon="🤖", layout="wide")

MAX_SELF_HEAL_RETRIES = 4
# "openrouter/free" auto-routes to whatever free model is currently available on OpenRouter —
# more robust than hardcoding a model ID since the free lineup rotates. To pin a specific
# free model instead, use e.g. "meta-llama/llama-3.3-70b-instruct:free" or "qwen/qwen3-coder:free".
LLM_MODEL = "openrouter/free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TRUSTED_DOC_SITES = ["pandas.pydata.org", "docs.python.org"]

SYSTEM_PROMPT = """You are a senior data analyst AI. You write short, correct, self-contained Python code
that analyzes a pandas DataFrame called `df` which is already loaded in memory.

Rules:
- Only use pandas, numpy, matplotlib.pyplot (as plt), and seaborn (as sns).
- Never read/write files and never re-load `df` — it already exists.
- If you produce a chart, create it on a variable named `fig` (e.g. fig, ax = plt.subplots()) and do NOT call plt.show().
- Store any text findings in a variable named `insight_text` (a plain-English string).
- Return ONLY a Python code block — no explanations outside the code."""


# --------------------------------------------------------------------------
# Data loading & profiling
# --------------------------------------------------------------------------
def load_any_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        return pd.read_json(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV, Excel, or JSON.")


def profile_dataframe(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return f"""Shape: {df.shape[0]} rows x {df.shape[1]} columns

Columns and dtypes:
{buf.getvalue()}

Missing values per column:
{df.isnull().sum().to_string()}

First 5 rows:
{df.head().to_string()}"""


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------
def build_generation_prompt(question: str, data_profile: str) -> str:
    return f"""Dataset profile:
{data_profile}

User question: "{question}"

Write Python code that answers this using the `df` DataFrame already in memory.
Produce a matplotlib/seaborn chart in `fig` where relevant, and always set `insight_text`
to a 2-4 sentence plain-English summary of what the data shows."""


def build_fix_prompt(question, data_profile, previous_code, error_message, doc_context) -> str:
    return f"""The following code failed.

Dataset profile:
{data_profile}

User question: "{question}"

Previous code:
```python
{previous_code}
```

Error raised:
{error_message}

Relevant excerpt from official Python/Pandas documentation (use this to fix the bug):
{doc_context}

Rewrite the FULL corrected code from scratch (same rules: use existing `df`, put chart in `fig`,
put summary text in `insight_text`). Return ONLY the corrected Python code."""


def extract_code_block(text: str) -> str:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            if p.strip().startswith("python"):
                return p.strip()[len("python"):].strip()
        return parts[1].strip() if len(parts) > 1 else text
    return text


def call_llm_for_code(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return extract_code_block(response.choices[0].message.content)


# --------------------------------------------------------------------------
# Sandbox execution
# --------------------------------------------------------------------------
SAFE_BUILTINS = {
    "range": range, "len": len, "min": min, "max": max, "sum": sum, "sorted": sorted,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "str": str, "int": int,
    "float": float, "bool": bool, "enumerate": enumerate, "zip": zip, "round": round,
    "print": print, "abs": abs,
}


def safe_exec(code_str: str, df: pd.DataFrame):
    local_ns = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}
    global_ns = {"__builtins__": SAFE_BUILTINS}
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code_str, global_ns, local_ns)
        return True, {
            "fig": local_ns.get("fig"),
            "insight_text": local_ns.get("insight_text", ""),
            "stdout": stdout_capture.getvalue(),
        }, ""
    except Exception:
        return False, {}, traceback.format_exc(limit=3)


# --------------------------------------------------------------------------
# RAG self-healing: live web search over official docs
# --------------------------------------------------------------------------
def web_search_docs(query: str, max_results: int = 5):
    resp = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.select("a.result__a"):
        href = a.get("href", "")
        if any(site in href for site in TRUSTED_DOC_SITES):
            links.append(href)
        if len(links) >= max_results:
            break
    return links


def fetch_page_text(url: str) -> str:
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 400):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def rerank_chunks(error_message: str, chunks: list, top_k: int = 2) -> str:
    if not chunks:
        return ""
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks + [error_message])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return "\n---\n".join(chunks[i] for i in top_idx)


def fetch_relevant_docs(error_message: str) -> str:
    query = error_message.strip().splitlines()[-1] if error_message.strip() else "pandas error"
    query = f"{query} pandas python site:pandas.pydata.org OR site:docs.python.org"
    urls = web_search_docs(query)
    all_chunks = []
    for url in urls[:3]:
        all_chunks.extend(chunk_text(fetch_page_text(url)))
    return rerank_chunks(error_message, all_chunks)


# --------------------------------------------------------------------------
# Self-healing agent loop
# --------------------------------------------------------------------------
def autonomous_analyze(client, df, question, data_profile, max_retries=MAX_SELF_HEAL_RETRIES):
    heal_log = []
    code_str = call_llm_for_code(client, build_generation_prompt(question, data_profile))

    for attempt in range(1, max_retries + 1):
        success, result, error = safe_exec(code_str, df)
        if success:
            heal_log.append(f"Attempt {attempt}: succeeded")
            return {"success": True, "code": code_str, "result": result, "attempts": attempt, "heal_log": heal_log}

        heal_log.append(f"Attempt {attempt}: failed — {error.strip().splitlines()[-1]}")
        doc_context = fetch_relevant_docs(error)
        heal_log.append(f"   RAG fetched {len(doc_context)} chars of doc context to self-correct")
        code_str = call_llm_for_code(client, build_fix_prompt(question, data_profile, code_str, error, doc_context))

    return {"success": False, "code": code_str, "result": {}, "attempts": max_retries, "heal_log": heal_log}


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🤖 Autonomous Data Science Co-Pilot")
st.caption("Upload data, ask a question in plain English — the agent writes, runs, and self-heals its own code.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenRouter API Key", type="password", help="Free, no card needed — sign up at openrouter.ai")
    st.markdown("Your key is only used for this session and is never stored.")
    st.caption("Free tier: 20 requests/min, 50/day. Get a key at [openrouter.ai](https://openrouter.ai).")

uploaded_file = st.file_uploader("Upload a data file", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    try:
        df = load_any_file(uploaded_file)
        st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]} rows x {df.shape[1]} columns")
        with st.expander("Preview data"):
            st.dataframe(df.head(20))
    except Exception as e:
        st.error(f"Could not read file: {e}")
        df = None
else:
    df = None

question = st.text_input("Ask a question about your data", placeholder="e.g. Show revenue by region as a bar chart")
run_clicked = st.button("Analyze", type="primary", disabled=not (df is not None and question and api_key))

if not api_key and (uploaded_file or question):
    st.info("Enter your OpenRouter API key in the sidebar to run the analysis.")

if run_clicked:
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    with st.spinner("Agent is writing and testing code..."):
        data_profile = profile_dataframe(df)
        outcome = autonomous_analyze(client, df, question, data_profile)

    if outcome["success"]:
        result = outcome["result"]
        st.subheader("Result")
        if result.get("fig") is not None:
            st.pyplot(result["fig"])
        if result.get("insight_text"):
            st.markdown(f"**Insight:** {result['insight_text']}")
        if result.get("stdout"):
            st.text(result["stdout"])
    else:
        st.error(f"Could not produce a working result after {outcome['attempts']} attempts.")

    with st.expander("Self-healing log"):
        for line in outcome["heal_log"]:
            st.text(line)

    with st.expander("Generated code (transparency / audit)"):
        st.code(outcome["code"], language="python")
