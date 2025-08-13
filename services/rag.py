# services/rag.py
import os
from functools import lru_cache
from typing import Optional, Tuple

from snowflake.snowpark import Session
from snowflake.core import Root

# ---- Config from env (works with zsh `export VAR=...`) ----
SF_ACCOUNT   = os.getenv("SNOWFLAKE_ACCOUNT", "")
SF_USER      = os.getenv("SNOWFLAKE_LAB_USER", "")
SF_PASSWORD  = os.getenv("SNOWFLAKE_LAB_PASSWORD", "")
SF_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "")
SF_DATABASE  = os.getenv("SNOWFLAKE_DATABASE", "")
SF_SCHEMA    = os.getenv("SNOWFLAKE_SCHEMA", "")
SF_ROLE      = "lab_role"

CORTEX_SEARCH_DB      = os.getenv("CORTEX_SEARCH_DB", SF_DATABASE)
CORTEX_SEARCH_SCHEMA  = os.getenv("CORTEX_SEARCH_SCHEMA", SF_SCHEMA)
CORTEX_SEARCH_SERVICE = os.getenv("CORTEX_SEARCH_SERVICE", "HAEBI_CORTEX_SEARCH_SERVICE")
CORTEX_COMPLETE_MODEL = os.getenv("CORTEX_COMPLETE_MODEL", "claude-3-5-sonnet")

CHUNK_LIMIT = int(os.getenv("CHUNK_LIMIT", "1"))

def _norm(name: str) -> str:
    """Uppercase unquoted identifiers to match Snowflake semantics."""
    name = (name or "").strip()
    return name if (name.startswith('"') and name.endswith('"')) else name.upper()

# ---------- Singletons ----------

@lru_cache(maxsize=1)
def _get_session() -> Session:
    """Create the Snowpark session exactly once (per process)."""
    sess = Session.builder.configs({
        "account":   SF_ACCOUNT,
        "user":      SF_USER,
        "password":  SF_PASSWORD,
        "warehouse": SF_WAREHOUSE,
        "database":  SF_DATABASE,
        "schema":    SF_SCHEMA,
        "role":      SF_ROLE,
    }).create()

    # Enforce context explicitly (defensive against defaults)
    if SF_ROLE:      sess.sql(f'USE ROLE {SF_ROLE}').collect()
    if SF_WAREHOUSE: sess.sql(f'USE WAREHOUSE {SF_WAREHOUSE}').collect()
    if SF_DATABASE:  sess.sql(f'USE DATABASE {SF_DATABASE}').collect()
    if SF_SCHEMA:    sess.sql(f'USE SCHEMA {SF_SCHEMA}').collect()

    # Quick keepalive/probe (no-op but exercises the session)
    sess.sql("SELECT 1").collect()
    return sess

@lru_cache(maxsize=1)
def _get_root() -> Root:
    """Root object built once off the singleton session."""
    return Root(_get_session())

# ---------- RAG Ops ----------

def search_top1_content(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Query Cortex Search and return (top PAGE_CONTENT (or similar), debug_info).
    """
    sess = _get_session()
    root = _get_root()

    db = _norm(CORTEX_SEARCH_DB or sess.get_current_database())
    sch = _norm(CORTEX_SEARCH_SCHEMA or sess.get_current_schema())
    svc = _norm(CORTEX_SEARCH_SERVICE)

    service = root.databases[db].schemas[sch].cortex_search_services[svc]
    res = service.search(query, columns=[], limit=max(1, CHUNK_LIMIT))

    results = getattr(res, "results", res) or []
    top = results[0] if isinstance(results, list) else results

    # Pick common content fields
    content = None
    if isinstance(top, dict):
        for k in ("PAGE_CONTENT", "CONTENT", "TEXT", "BODY"):
            if k in top:
                content = top[k]; break
    else:
        for k in ("PAGE_CONTENT", "CONTENT", "TEXT", "BODY"):
            if hasattr(top, k):
                content = getattr(top, k); break
    if content is None:
        content = str(top)

    dbg = f"role={sess.get_current_role()} db={db} schema={sch} service={svc}"
    return content, dbg

def complete_with_context(question: str, context: str) -> str:
    """
    Use the SAME Snowpark session to call SNOWFLAKE.CORTEX.COMPLETE(model, prompt).
    """
    sess = _get_session()
    # Escape single quotes for SQL string literal
    safe_q   = question.replace("'", "''")
    safe_ctx = context.replace("'", "''")
    prompt = (
        "You are a helpful assistant. Answer based ONLY on the provided context.\n\n"
        f"Question: {safe_q}\n\nContext:\n{safe_ctx}\n\nAnswer clearly and concisely:"
    )
    # Use bind parameters to keep it tidy
    df = sess.sql("SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER", [CORTEX_COMPLETE_MODEL, prompt])
    row = df.collect()[0]
    return (row["ANSWER"] or "").strip()

def ask_with_search(question: str) -> Tuple[str, dict]:
    """
    Full pipeline: Search top-1 → COMPLETE with model → return text + meta.
    Session/Root are reused via lru_cache singletons.
    """
    ctx, dbg = search_top1_content(question)
    if not ctx:
        return "I couldn't find anything relevant in search.", {"search_debug": dbg}
    ans = complete_with_context(question, ctx)
    return (ans or "(empty)"), {"search_debug": dbg}

# Optional: eager warmup you can call from app.py during startup
def warmup() -> None:
    _get_session()
    _get_root()
