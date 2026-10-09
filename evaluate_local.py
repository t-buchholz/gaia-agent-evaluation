import os
import re
import sys
import time
import tempfile
import subprocess
from html import unescape
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import requests
from groq import Groq
from ddgs import DDGS

# ----------------------------------------------------------------------------
# KONFIGURATION
# ----------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_API_KEY")

BASE_URL = "https://agents-course-unit4-scoring.hf.space"
USERNAME = "XX"
AGENT_CODE = "https://huggingface.co/spaces/XX/hf-agent-course-unit4/tree/main"

MODEL = "qwen/qwen3.8-27b"
WHISPER_MODEL = "whisper-large-v3-turbo"   # bei Groq kostenlos (Free Tier)

MAX_TOKENS_THINK = 3500    # Reasoning-Modus (Reasoning wird per "hidden" ausgeblendet)
MAX_TOKENS_FAST = 300      # einfache Aufgaben ohne Reasoning
CONTEXT_CHARS = 9000       # max. Kontext-Zeichen pro Frage (Free-Tier-TPM-Limit schonen)

RUN_ATTACHED_PYTHON = True  # angehaengte .py-Dateien der Scoring-API ausfuehren (Timeout 15 s)
SEARCH_TIMEOUT_S = 15
PAUSE_BETWEEN_Q_S = 6       # Pause zwischen Fragen (Groq-Free-Tier-Limits)

UA = {"User-Agent": "hf-agent-course-unit4/1.0 (personal learning project)"}
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a precise question-answering engine for a benchmark with exact-match scoring.\n"
    "You get a QUESTION and optionally CONTEXT (tool results, transcripts, file content, web text).\n"
    "Think carefully, then reply with ONLY the bare final answer: a number, a name, a word, or a "
    "short comma-separated list. No explanation, no full sentence, no markdown, no quotes.\n"
    "Rules:\n"
    "- Never write 'Answer:' or 'Final Answer:'.\n"
    "- Apply every constraint of the question strictly (definitions, ordering such as alphabetical "
    "or ascending, requested format, number of decimals, abbreviations vs. full names).\n"
    "- Numbers: no units or thousands separators unless the question requests them.\n"
    "- Lists: comma-separated, exactly in the order/format the question asks for.\n"
    "- Results of executed code or transcripts in the CONTEXT are reliable; prefer them over guesses.\n"
    "- If the question can be solved by reasoning alone (logic, text manipulation, arithmetic), "
    "solve it yourself.\n"
    "- Never say that you cannot access something: always give your best short answer."
)

STOP = {"what", "which", "that", "this", "with", "from", "have", "were", "there", "their", "about",
        "give", "only", "your", "into", "than", "then", "when", "where", "does", "would", "could"}


# ----------------------------------------------------------------------------
# LLM-ZUGRIFF (Groq, Qwen)
# ----------------------------------------------------------------------------
class TooLarge(Exception):
    """Anfrage ueberschreitet das Token-Limit pro Minute/Request."""


def _retry_wait(msg: str, attempt: int) -> float:
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
    if m:
        return min(70.0, int(m.group(1) or 0) * 60 + float(m.group(2)) + 1.0)
    return 15.0 * (attempt + 1)


def chat(system: str, user: str, think: bool = True, max_tokens: int | None = None) -> str:
    max_tokens = max_tokens or (MAX_TOKENS_THINK if think else MAX_TOKENS_FAST)
    temperature = 0.3 if think else 0.0
    variants = [
        {"reasoning_effort": "default" if think else "none", "reasoning_format": "hidden"},
        {"reasoning_format": "hidden"},
        {},
    ]
    for extra in variants:
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                low = msg.lower()
                if "413" in low or "request too large" in low or "reduce your message size" in low:
                    raise TooLarge(msg)
                if "per day" in low and ("429" in low or "rate" in low):
                    print("   [llm] Tageslimit erreicht (TPD) - morgen erneut versuchen.")
                    return ""
                if "429" in low or "rate limit" in low:
                    wait = _retry_wait(msg, attempt)
                    print(f"   [llm] Rate-Limit, warte {wait:.0f}s")
                    time.sleep(wait)
                    continue
                print(f"   [llm] Fehler ({extra}): {type(e).__name__}: {msg[:200]}")
                break  # naechste Parameter-Variante
    return ""


# ----------------------------------------------------------------------------
# HILFSFUNKTIONEN
# ----------------------------------------------------------------------------
def clean_response(text: str) -> str:
    """Bereinigt LLM-Ausgabe fuer exakten String-Vergleich."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()
    t = re.sub(r"```[a-zA-Z]*", "", t).replace("```", "")
    t = t.replace("**", "").replace("__", "").replace("`", "")
    t = re.sub(r"^\s*(final\s*answer|answer|antwort)\s*[:\-]\s*", "", t, flags=re.IGNORECASE)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    t = " ".join(lines)
    quotes = "\"'\u201c\u201d\u2018\u2019"
    for _ in range(2):
        t = t.strip().strip(quotes).strip()
        if t.endswith(".") and not re.search(r"\b[A-Z][a-z]?\.$", t):
            t = t[:-1].strip()
    return t


def with_timeout(fn, seconds, *args, default=None):
    """Fuehrt fn mit hartem Timeout aus; bei Fehler/Timeout wird default geliefert."""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn, *args).result(timeout=seconds)
    except FutureTimeout:
        print(f"   [{fn.__name__}] Timeout")
        return default
    except Exception as e:  # noqa: BLE001
        print(f"   [{fn.__name__}] {type(e).__name__}: {str(e)[:120]}")
        return default
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def pick_relevant(text: str, question: str, budget: int) -> str:
    """Waehlt die zur Frage passendsten Textabschnitte (Keyword-Overlap)."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= budget:
        return text
    kws = set(re.findall(r"[a-z0-9]{4,}", question.lower())) - STOP
    size = 700
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    scores = [sum(1 for k in kws if k in c.lower()) for c in chunks]
    order = sorted(range(1, len(chunks)), key=lambda i: -scores[i])
    chosen, used = [0], len(chunks[0])
    for i in order:
        if scores[i] == 0 or used + len(chunks[i]) > budget:
            continue
        chosen.append(i)
        used += len(chunks[i])
    return " ... ".join(chunks[i] for i in sorted(chosen))


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", unescape(html)).strip()


def maybe_unreverse(question: str) -> str:
    """Erkennt rueckwaerts geschriebene Fragen und dreht sie zurueck."""
    common = {"the", "of", "and", "to", "in", "is", "you", "if", "this", "that", "for",
              "what", "write", "as", "a", "answer", "word", "sentence"}

    def score(t: str) -> int:
        return sum(1 for w in re.findall(r"[a-z]+", t.lower()) if w in common)

    rev = question[::-1]
    return rev if score(rev) > score(question) + 2 else ""


def run_python(code: str, timeout: int = 20) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        out = (r.stdout or "").strip()
        if r.returncode != 0:
            out += "\n[stderr] " + (r.stderr or "")[-500:]
        return out.strip()
    except subprocess.TimeoutExpired:
        return "(execution timed out)"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ----------------------------------------------------------------------------
# DATEIEN, AUDIO, YOUTUBE
# ----------------------------------------------------------------------------
def download_file(task_id: str, file_name: str):
    r = requests.get(f"{BASE_URL}/files/{task_id}", timeout=30)
    if r.status_code != 200:
        return None, None
    folder = tempfile.mkdtemp(prefix="hf_task_")
    path = os.path.join(folder, file_name or task_id)
    with open(path, "wb") as f:
        f.write(r.content)
    return path, r.content


def transcribe_audio(file_name: str, data: bytes) -> str:
    res = client.audio.transcriptions.create(
        file=(file_name, data), model=WHISPER_MODEL, response_format="text", language="en",
    )
    return res if isinstance(res, str) else getattr(res, "text", str(res))


def code_solve(question: str, path: str, preview: str) -> str:
    """LLM schreibt pandas-Code, der die Antwort aus der Datei berechnet; Code wird ausgefuehrt."""
    prompt = (
        f"QUESTION:\n{question}\n\nA data file is available at the variable FILE_PATH.\n"
        f"File preview:\n{preview}\n\n"
        "Write a complete Python script (pandas allowed; FILE_PATH is already defined) that "
        "computes the answer and prints ONLY the final answer value, formatted as the question "
        "requests (e.g. number of decimals). Output only one ```python code block."
    )
    raw = chat("You are an expert Python data analyst.", prompt, think=False, max_tokens=1200)
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, flags=re.DOTALL)
    code = m.group(1) if m else raw
    if not code.strip():
        return ""
    return run_python(f'FILE_PATH = r"{path}"\n' + code)


def handle_file(task_id: str, file_name: str, question: str) -> str:
    """Gibt Kontext-Text zur angehaengten Datei zurueck (hohe Prioritaet)."""
    try:
        path, data = download_file(task_id, file_name)
    except Exception as e:  # noqa: BLE001
        return f"(File download failed: {type(e).__name__})"
    if not path:
        return f"(Attached file '{file_name}' could not be downloaded.)"

    ext = os.path.splitext(file_name)[1].lower()
    try:
        if ext in {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}:
            text = transcribe_audio(os.path.basename(path), data)
            print(f"   [audio] {len(text)} Zeichen transkribiert")
            return f"AUDIO TRANSCRIPT of '{file_name}':\n{text}"

        if ext == ".py":
            src = data.decode("utf-8", errors="replace")
            out = ""
            if RUN_ATTACHED_PYTHON:
                out = run_python(src, timeout=15)
            print(f"   [python] Ausgabe: {out[:80]!r}")
            return (f"ATTACHED PYTHON SOURCE:\n{src[:3000]}\n\n"
                    f"OUTPUT WHEN EXECUTED (reliable):\n{out}")

        if ext in {".xlsx", ".xls", ".csv", ".tsv"}:
            import pandas as pd
            if ext == ".csv":
                sheets = {"csv": pd.read_csv(path)}
            elif ext == ".tsv":
                sheets = {"tsv": pd.read_csv(path, sep="\t")}
            else:
                sheets = pd.read_excel(path, sheet_name=None)
            preview = "\n".join(
                f"[sheet {name}] columns={list(df.columns)}\n{df.head(12).to_csv(index=False)}"
                for name, df in sheets.items()
            )[:3500]
            result = code_solve(question, path, preview)
            print(f"   [code] Ergebnis: {result[:80]!r}")
            return (f"TABLE PREVIEW:\n{preview}\n\n"
                    f"RESULT OF A SCRIPT THAT COMPUTED THE ANSWER (reliable, check format): {result}")

        if ext in {".txt", ".json", ".md", ".xml", ".html"}:
            return f"FILE '{file_name}':\n{data.decode('utf-8', errors='replace')[:6000]}"
    except Exception as e:  # noqa: BLE001
        print(f"   [file] Fehler: {type(e).__name__}: {str(e)[:150]}")
        return f"(Attached file '{file_name}' could not be processed: {type(e).__name__})"

    return f"(Attached file '{file_name}' of type {ext} cannot be read by this pipeline.)"


def get_youtube_transcript(question: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", question)
    if not m:
        return ""
    vid = m.group(1)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("   [youtube] pip install youtube-transcript-api")
        return ""
    try:
        try:
            parts = YouTubeTranscriptApi().fetch(vid)          # neue API (>= 1.0)
            text = " ".join(getattr(p, "text", "") for p in parts)
        except AttributeError:
            parts = YouTubeTranscriptApi.get_transcript(vid)   # alte API
            text = " ".join(p["text"] for p in parts)
        print(f"   [youtube] Transkript: {len(text)} Zeichen")
        return f"YOUTUBE TRANSCRIPT:\n{pick_relevant(text, question, 5000)}"
    except Exception as e:  # noqa: BLE001
        print(f"   [youtube] kein Transkript: {type(e).__name__}")
        return ""


# ----------------------------------------------------------------------------
# WEBSUCHE (Wikipedia-API + ddgs + Seitentext)
# ----------------------------------------------------------------------------
def make_queries(question: str) -> list[str]:
    raw = chat(
        "You write web search queries.",
        f"Question: {question}\n\nWrite 2 short, distinct search queries (3-8 keywords each) that "
        "would find the answer. One per line, no numbering, no quotes.",
        think=False, max_tokens=80,
    )
    qs = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in raw.splitlines() if ln.strip()]
    return qs[:2] or [question[:120]]


def wiki_search(query: str):
    api = "https://en.wikipedia.org/w/api.php"
    r = requests.get(api, headers=UA, timeout=10, params={
        "action": "query", "list": "search", "srsearch": query, "srlimit": 2, "format": "json"})
    titles = [x["title"] for x in r.json().get("query", {}).get("search", [])]
    out = []
    for t in titles:
        r = requests.get(api, headers=UA, timeout=15, params={
            "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
            "titles": t, "format": "json"})
        page = next(iter(r.json()["query"]["pages"].values()))
        out.append((t, page.get("extract", "")))
    return out


def ddg_search(query: str):
    for attempt in range(2):
        try:
            with DDGS() as d:
                return list(d.text(query, max_results=6))
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return []


def fetch_page(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=10)
    if "html" not in r.headers.get("content-type", ""):
        return ""
    return html_to_text(r.text)


def gather_web_context(question: str) -> str:
    wiki_parts, snippet_parts, page_parts = [], [], []
    seen_titles, page_urls = set(), []
    queries = with_timeout(make_queries, 40, question, default=[question[:120]]) or [question[:120]]
    print(f"   [search] Queries: {queries}")

    for q in queries:
        for title, text in with_timeout(wiki_search, 25, q, default=[]) or []:
            if title in seen_titles or not text:
                continue
            seen_titles.add(title)
            wiki_parts.append(f"[Wikipedia: {title}]\n{pick_relevant(text, question, 3000)}")
        for r in with_timeout(ddg_search, SEARCH_TIMEOUT_S, q, default=[]) or []:
            snippet_parts.append(f"- {r.get('title', '')}: {r.get('body', '')} ({r.get('href', '')})")
            href = r.get("href", "")
            if href and "wikipedia.org" not in href and href not in page_urls:
                page_urls.append(href)

    for url in page_urls[:2]:
        txt = with_timeout(fetch_page, 12, url, default="")
        if txt:
            page_parts.append(f"[Page: {url}]\n{pick_relevant(txt, question, 2000)}")

    return "\n\n".join(wiki_parts + page_parts + ["WEB SNIPPETS:\n" + "\n".join(snippet_parts[:10])])


# ----------------------------------------------------------------------------
# LOESUNG EINER FRAGE
# ----------------------------------------------------------------------------
def solve(question: str, context: str) -> str:
    budgets = [(CONTEXT_CHARS, MAX_TOKENS_THINK), (CONTEXT_CHARS // 2, 2500), (CONTEXT_CHARS // 4, 1500)]
    for ctx_chars, max_tok in budgets:
        user_msg = (f"QUESTION:\n{question}\n\nCONTEXT:\n{context[:ctx_chars] if context else '(none)'}"
                    "\n\nGive the final answer only:")
        try:
            raw = chat(SYSTEM_PROMPT, user_msg, think=True, max_tokens=max_tok)
            if not raw:  # z.B. Token-Budget im Reasoning aufgebraucht
                raw = chat(SYSTEM_PROMPT, user_msg, think=False, max_tokens=MAX_TOKENS_FAST)
            return raw
        except TooLarge:
            print(f"   [llm] Anfrage zu gross - verkleinere Kontext (war {ctx_chars} Zeichen)")
    return ""


def main():
    if GROQ_API_KEY.startswith("HIER_"):
        raise SystemExit("Bitte GROQ_API_KEY setzen (Umgebungsvariable oder im Skript).")

    print("Hole Fragen ...")
    r = requests.get(f"{BASE_URL}/questions", timeout=30)
    r.raise_for_status()
    questions = r.json()
    print(f"{len(questions)} Fragen erhalten.\n")

    answers = []
    for i, q in enumerate(questions, 1):
        task_id = q["task_id"]
        question = q["question"]
        file_name = q.get("file_name", "") or ""
        print(f"[{i}/{len(questions)}] {task_id}\n   Q: {question[:120]!r}")

        try:
            skip_search = False
            rev = maybe_unreverse(question)
            if rev:
                print(f"   [reverse] Frage zurueckgedreht: {rev[:100]!r}")
                question, skip_search = rev, True

            ctx_parts = []
            if file_name:
                ctx_parts.append(handle_file(task_id, file_name, question))
                skip_search = True

            yt = get_youtube_transcript(question)
            if yt:
                ctx_parts.append(yt)
                skip_search = True

            if not skip_search:
                ctx_parts.append(gather_web_context(question))

            context = "\n\n".join(p for p in ctx_parts if p)
            raw = solve(question, context)
            answer = clean_response(raw)
        except Exception as e:  # noqa: BLE001
            print(f"   [fehler] {type(e).__name__}: {str(e)[:150]}")
            answer = ""

        print(f"   A: {answer!r}\n")
        answers.append({"task_id": task_id, "submitted_answer": answer})
        time.sleep(PAUSE_BETWEEN_Q_S)

    payload = {"username": USERNAME, "agent_code": AGENT_CODE, "answers": answers}
    print("Sende Antworten ...")
    resp = requests.post(f"{BASE_URL}/submit", json=payload, timeout=60)
    try:
        print(resp.json())
    except Exception:  # noqa: BLE001
        print(resp.status_code, resp.text)


if __name__ == "__main__":
    main()