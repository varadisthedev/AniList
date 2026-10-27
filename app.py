"""AniStat: anime discovery and content-based recommendations.

Expected project layout:
AniList/
  app.py
  dataset/myanimelist_cleaned.csv
  models/anistat_score_regressor.joblib          (optional)
  models/anistat_feature_columns.joblib           (optional)
  db/                                            (created automatically)

Optional JWT auth: set ANISTAT_AUTH_ENABLED=1 and ANISTAT_JWT_SECRET to a long secret.
Poster URLs are cached in SQLite. Use the sidebar button to populate missing URLs in batches.
"""
from __future__ import annotations

import os
import re
import time
import json
import hmac
import html
import base64
import hashlib
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse
from difflib import SequenceMatcher, get_close_matches

import pandas as pd
import streamlit as st
import requests
import sqlitecloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.sparse import hstack, csr_matrix

try:
    from rapidfuzz import process as rapidfuzz_process, fuzz as rapidfuzz_fuzz
except Exception:
    rapidfuzz_process = rapidfuzz_fuzz = None

try:
    import joblib
except Exception:
    joblib = None

try:
    import jwt
except Exception:
    jwt = None

# ------------------------------- Configuration -------------------------------
ROOT = Path(__file__).resolve().parent
DATASET_PATH = Path(os.getenv("ANISTAT_DATASET", str(ROOT / "dataset" / "myanimelist_cleaned.csv")))
# SQLite Cloud is the primary database. Supply the full connection string in
# environment variable SQLITECLOUD_CONNECTION_STRING or Streamlit secrets.
# Example: sqlitecloud://cuw6zxarvk.g4.sqlite.cloud:8860/auth.sqlitecloud?apikey=YOUR_API_KEY
SQLITECLOUD_CONNECTION_STRING = os.getenv("SQLITECLOUD_CONNECTION_STRING", "").strip()
if not SQLITECLOUD_CONNECTION_STRING:
    try:
        SQLITECLOUD_CONNECTION_STRING = str(st.secrets.get("SQLITECLOUD_CONNECTION_STRING", "")).strip()
    except Exception:
        SQLITECLOUD_CONNECTION_STRING = ""
MODEL_PATH = ROOT / "models" / "anistat_score_regressor.joblib"
FEATURES_PATH = ROOT / "models" / "anistat_feature_columns.joblib"
AUTH_ENABLED = True
JWT_SECRET = os.getenv("ANISTAT_JWT_SECRET", "").strip()
if not JWT_SECRET:
    try:
        JWT_SECRET = str(st.secrets.get("ANISTAT_JWT_SECRET", "")).strip()
    except Exception:
        JWT_SECRET = ""
if len(JWT_SECRET) < 32:
    JWT_SECRET = "anistat-jwt-auth-secret-key-production-myanimelist-token-2026-v1"

st.set_page_config(page_title="AniStat · Discover Anime", page_icon="◉", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #141414; color: #f5f5f5; }
[data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
[data-testid="stHeader"] { background: rgba(20,20,20,.85); backdrop-filter: blur(8px); }
.block-container { padding-top: 4.8rem !important; max-width: 1600px; }
.brand { font-size: 1.15rem; font-weight: 900; letter-spacing: .13em; color: #ff5e00; line-height: 2.2; }
div[data-testid="stPopover"] > button {
  background: #1f1f1f;
  color: #f0f0f0;
  border: 1px solid #383838;
  border-radius: 8px;
  padding: 0.35rem 0.85rem;
  font-weight: 600;
  font-size: 0.84rem;
  transition: all 0.2s ease;
}
div[data-testid="stPopover"] > button:hover {
  background: #282828;
  border-color: #ff5e00;
  color: #ff5e00;
}
div[data-testid="stPopover"] > button {
  background: transparent !important;
  border: 1px solid transparent !important;
  box-shadow: none !important;
  border-radius: 8px !important;
  padding: 0.4rem !important;
  min-height: 42px !important;
  min-width: 42px !important;
  display: flex;
  align-items: center;
  justify-content: center;
}
div[data-testid="stPopover"] > button:hover {
  background: #1f1f1f !important;
  border-color: #383838 !important;
}
div[data-testid="stPopover"] > button img {
  width: 22px !important;
  height: 22px !important;
  object-fit: contain !important;
  display: block;
}
div.stButton > button[kind="secondaryButton"] {
  background: #333;
  color: #e0e0e0;
  border: 1px solid #4a4a4a;
}
div.stButton > button[kind="secondaryButton"]:hover {
  background: #444;
  color: #fff;
  border-color: #555;
}
.hero { position: relative; overflow: hidden; border-radius: 18px; padding: 2.2rem 2.5rem 1.8rem; border: 1px solid rgba(255,255,255,.07); margin-bottom: 1.2rem; min-height: 360px; display: flex; flex-direction: column; justify-content: center; background: #141414; }
.hero-bg { position: absolute; top: 0; right: 0; bottom: 0; left: 0; width: 100%; height: 100%; object-fit: cover; object-position: center right; transform: scale(1.35); transform-origin: center right; z-index: 0; pointer-events: none; }
.hero-overlay { position: absolute; top: 0; right: 0; bottom: 0; left: 0; background-image: linear-gradient(90deg, rgba(20,20,20,.98) 0%, rgba(20,20,20,.82) 50%, rgba(20,20,20,.20) 100%), linear-gradient(0deg, #141414 0%, rgba(20,20,20,0) 65%); z-index: 1; pointer-events: none; }
.hero-content { position: relative; z-index: 2; max-width: 720px; }
.hero h1 { font-size: clamp(2rem, 4vw, 3.4rem); margin: 0; line-height: 1.1; }
.hero p { color: #d0d0d0; max-width: 720px; }
.eyebrow { color: #ff5e00; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; font-size: .78rem; }
.section-title { font-size: 1.5rem; font-weight: 800; margin: 1.5rem 0 .2rem; }
.section-sub { color: #aaa; margin-bottom: 1rem; }
.predictor-result { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; margin-top: 1rem; padding: 1rem 1.1rem; background: #292929; border: 1px solid #ff5e00; border-radius: 10px; }
.predictor-result span { color: #d0d0d0; }
.predictor-result strong { color: #ff7a2f; font-size: 1.5rem; }
.anime-card { background: #1f1f1f; border: 1px solid #303030; border-radius: 12px; overflow: hidden; height: 100%; margin-bottom: .8rem; }

@keyframes diagonal-shimmer {
  0% {
    background-position: -200% -200%;
  }
  100% {
    background-position: 200% 200%;
  }
}
.anime-card img {
  width: 100%;
  aspect-ratio: 2/3;
  object-fit: cover;
  display: block;
  background: linear-gradient(
    135deg,
    #181818 0%,
    #242424 25%,
    #383838 50%,
    #242424 75%,
    #181818 100%
  );
  background-size: 300% 300%;
  animation: diagonal-shimmer 2.2s infinite ease-in-out;
}
.poster-placeholder {
  aspect-ratio: 2/3;
  background: linear-gradient(
    135deg,
    #181818 0%,
    #242424 25%,
    #383838 50%,
    #242424 75%,
    #181818 100%
  );
  background-size: 300% 300%;
  animation: diagonal-shimmer 2.2s infinite ease-in-out;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #777;
  font-size: 3rem;
}
.card-body { padding: .75rem; }
.card-title { font-size: .95rem; line-height: 1.3; font-weight: 800; margin-bottom: .4rem; min-height: 2.4rem; }
.card-meta { color: #b8b8b8; font-size: .78rem; line-height: 1.45; }
.score { display: inline-block; color: #fff; background: #ff5e00; border-radius: 5px; font-weight: 800; padding: .12rem .4rem; margin-right: .3rem; font-size: .78rem; }
div.stButton > button { background: #ff5e00; color: white; border: 0; border-radius: 8px; font-weight: 750; }
div.stButton > button:hover { background: #e65300; color: white; border: 0; }
[data-testid="stTextInput"] input { background: #262626; color: #fff; border-color: #444; }
hr { border-color: #303030; }
.small-muted { color:#aaa; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def _hero_image_data_uri() -> str:
    path = ROOT / "assets" / "site_hero.png"
    if not path.exists():
        return ""
    try:
        from PIL import Image
        import io
        img = Image.open(path)
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=85)
        return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


_hero_image = _hero_image_data_uri()


@st.cache_data
def _icon_data_uri(name: str) -> str:
    path = ROOT / "assets" / name
    if not path.exists():
        return ""
    try:
        from PIL import Image
        import io
        with Image.open(path) as img:
            img = img.convert("RGBA")
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
            # Normalize the visible icon into a square canvas.
            img.thumbnail((96, 96), Image.Resampling.LANCZOS)
            canvas = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
            x = (128 - img.width) // 2
            y = (128 - img.height) // 2
            canvas.alpha_composite(img, (x, y))
            buf = io.BytesIO()
            canvas.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(
            buf.getvalue()
        ).decode("ascii")
    except Exception as exc:
        print(f"Could not load icon {name}: {exc}")
        return ""


_profile_icon = _icon_data_uri("Profile.png")
_history_icon = _icon_data_uri("history.png")

# ---------------------------------- SQLite Cloud -----------------------------
@contextmanager
def get_conn():
    """Open a short-lived SQLite Cloud connection and always close it.

    sqlitecloud uses autocommit mode, so do not rely on sqlite3 transaction
    context-manager semantics. Each INSERT/UPDATE is persisted immediately.
    """
    if not SQLITECLOUD_CONNECTION_STRING:
        raise RuntimeError(
            "SQLite Cloud is not configured. Set SQLITECLOUD_CONNECTION_STRING "
            "in your environment or .streamlit/secrets.toml."
        )
    conn = sqlitecloud.connect(SQLITECLOUD_CONNECTION_STRING)
    try:
        # sqlitecloud's Cursor is not a sqlite3.Cursor, so sqlite3.Row rejects it;
        # the driver special-cases its own Row class for name-based access instead.
        conn.row_factory = sqlitecloud.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
        except Exception:
            # Some managed environments control PRAGMA settings server-side.
            pass
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables with individual statements (SQLite Cloud has no executescript)."""
    statements = [
        """CREATE TABLE IF NOT EXISTS anime_catalog (
            anime_key TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS poster_cache (
            anime_key TEXT PRIMARY KEY,
            mal_id INTEGER,
            poster_url TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY(anime_key) REFERENCES anime_catalog(anime_key) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS favorites (
            user_id INTEGER NOT NULL,
            anime_key TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(user_id, anime_key),
            FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY(anime_key) REFERENCES anime_catalog(anime_key) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS user_ratings (
            user_id INTEGER NOT NULL,
            anime_key TEXT NOT NULL,
            rating REAL NOT NULL CHECK(rating >= 1 AND rating <= 10),
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(user_id, anime_key),
            FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY(anime_key) REFERENCES anime_catalog(anime_key) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS search_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            query TEXT NOT NULL,
            matched_title TEXT,
            searched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )""",
    ]
    with get_conn() as conn:
        for statement in statements:
            conn.execute(statement)


def record_search_history(user_id: int, query: str, matched_title: str | None = None) -> None:
    try:
        with get_conn() as conn:
            last = conn.execute(
                "SELECT query FROM search_history WHERE user_id=? ORDER BY history_id DESC LIMIT 1",
                (int(user_id),),
            ).fetchone()
            if not last or last["query"].strip().casefold() != query.strip().casefold():
                conn.execute(
                    "INSERT INTO search_history(user_id, query, matched_title) VALUES(?,?,?)",
                    (int(user_id), query.strip(), matched_title or ""),
                )
    except Exception as exc:
        print(f"record_search_history error: {exc}")


def get_search_history(user_id: int, limit: int = 15) -> list[dict]:
    try:
        with get_conn() as conn:
            rows = conn.execute(
                f"SELECT history_id, query, matched_title, searched_at FROM search_history WHERE user_id={int(user_id)} ORDER BY history_id DESC LIMIT {int(limit)}"
            ).fetchall()
        return [{"history_id": r["history_id"], "query": r["query"], "matched_title": r["matched_title"], "searched_at": r["searched_at"]} for r in rows]
    except Exception as exc:
        print(f"get_search_history error for user {user_id}: {exc}")
        return []


def clear_search_history(user_id: int) -> None:
    try:
        with get_conn() as conn:
            conn.execute("DELETE FROM search_history WHERE user_id=?", (user_id,))
    except Exception:
        pass


def normalize_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def first_existing(columns, options):
    lookup = {str(c).strip().casefold(): c for c in columns}
    for option in options:
        if option.casefold() in lookup:
            return lookup[option.casefold()]
    return None


def load_csv() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        st.error(f"Dataset not found: `{DATASET_PATH}`. Put your CSV in the `dataset` folder or set ANISTAT_DATASET.")
        st.stop()
    try:
        data = pd.read_csv(DATASET_PATH, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        data = pd.read_csv(DATASET_PATH, encoding="latin1", low_memory=False)
    data.columns = [str(c).strip() for c in data.columns]

    # Support the notebook's cleaned schema and common variations. A raw export
    # can still have an empty/corrupted "Title" (often the Japanese title) next
    # to a populated "English Title"; prefer whichever candidate actually has
    # usable text instead of just the first one present.
    title_candidates = ["English Title", "Title", "Anime Title", "Name"]
    title_col = None
    for option in title_candidates:
        col = first_existing(data.columns, [option])
        if col and data[col].astype(str).str.strip().replace({"nan": ""}).ne("").mean() > 0.5:
            title_col = col
            break
    if title_col is None:
        title_col = first_existing(data.columns, title_candidates)
    if title_col is None:
        st.error(f"Could not find a title column in {DATASET_PATH.name}. Columns: {list(data.columns)}")
        st.stop()
    if title_col != "Title":
        if "Title" in data.columns:
            # Drop the stale/empty original "Title" column so the rename below
            # doesn't produce two columns sharing the "Title" label.
            data = data.drop(columns=["Title"])
        data = data.rename(columns={title_col: "Title"})
    poster_candidates = ["Poster_URL", "poster_url", "image_url", "Image_URL"]
    poster_col = first_existing(data.columns, poster_candidates)
    if poster_col and poster_col != "Poster_URL":
        data = data.rename(columns={poster_col: "Poster_URL"})
    for col in ["Genres", "Themes", "Description", "Type", "Status", "Demographics", "URL", "Score", "Episodes", "Start Date", "End Date", "Rank", "Poster_URL"]:
        if col not in data.columns:
            data[col] = "" if col not in {"Score", "Episodes", "Rank"} else pd.NA
    for col in ["Title", "Genres", "Themes", "Description", "Type", "Status", "Demographics", "URL", "Start Date", "End Date", "Poster_URL"]:
        data[col] = data[col].map(normalize_value)
    for col in ["Score", "Episodes", "Rank"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data[data["Title"].astype(str).str.strip().ne("")].copy()
    data = data.drop_duplicates(subset=["URL"], keep="first") if data["URL"].astype(str).str.strip().ne("").any() else data.drop_duplicates(subset=["Title"], keep="first")
    data = data.reset_index(drop=True)
    # Stable key uses MAL ID where possible; otherwise a normalized title + row index.
    keys = []
    for idx, row in data.iterrows():
        mal_id = extract_mal_id(row.get("URL", ""))
        keys.append(f"mal:{mal_id}" if mal_id else f"title:{slugify(row['Title'])}:{idx}")
    data["_anime_key"] = keys
    return data


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).casefold()).strip("-")[:100] or "unknown"


def extract_mal_id(url) -> int | None:
    match = re.search(r"myanimelist\.net/anime/(\d+)", str(url or ""), flags=re.I)
    return int(match.group(1)) if match else None


def sync_catalog_to_db(data: pd.DataFrame) -> None:
    """Bulk-sync the catalog, preserving poster cache and user data."""
    catalog_rows = []
    poster_rows = []
    for _, row in data.iterrows():
        key = str(row["_anime_key"])
        metadata = {
            str(k): (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
            for k, v in row.drop(labels=["_anime_key"]).to_dict().items()
        }
        catalog_rows.append((key, str(row["Title"]), json.dumps(metadata, ensure_ascii=False, default=str)))
        p_url = normalize_value(row.get("Poster_URL")) or None
        status = "cached" if p_url else "pending"
        poster_rows.append((key, extract_mal_id(row.get("URL", "")), p_url, status))

    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO anime_catalog(anime_key,title,metadata_json) VALUES(?,?,?) "
            "ON CONFLICT(anime_key) DO UPDATE SET title=excluded.title, metadata_json=excluded.metadata_json",
            catalog_rows,
        )
        conn.executemany(
            "INSERT INTO poster_cache(anime_key,mal_id,poster_url,status) VALUES(?,?,?,?) "
            "ON CONFLICT(anime_key) DO UPDATE SET mal_id=excluded.mal_id, "
            "poster_url=COALESCE(excluded.poster_url, poster_cache.poster_url), "
            "status=CASE WHEN excluded.poster_url IS NOT NULL AND TRIM(excluded.poster_url)<>'' THEN 'cached' ELSE poster_cache.status END",
            poster_rows,
        )


def read_poster_cache() -> dict[str, str]:
    try:
        with get_conn() as conn:
            rows = conn.execute("SELECT anime_key, poster_url FROM poster_cache WHERE poster_url IS NOT NULL AND TRIM(poster_url) <> ''").fetchall()
        return {str(row["anime_key"]): str(row["poster_url"]) for row in rows}
    except Exception:
        return {}


def get_poster_url(anime_key: str) -> str | None:
    try:
        with get_conn() as conn:
            row = conn.execute("SELECT poster_url FROM poster_cache WHERE anime_key=?", (anime_key,)).fetchone()
        return row["poster_url"] if row and row["poster_url"] else None
    except Exception:
        return None


def set_poster(anime_key: str, url: str | None, status: str) -> None:
    try:
        with get_conn() as conn:
            conn.execute("UPDATE poster_cache SET poster_url=?, status=?, fetched_at=CURRENT_TIMESTAMP WHERE anime_key=?", (url, status, anime_key))
    except Exception:
        pass


def poster_cache_counts() -> tuple[int, int]:
    try:
        with get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS total, SUM(CASE WHEN poster_url IS NOT NULL AND TRIM(poster_url)<>'' THEN 1 ELSE 0 END) AS cached FROM poster_cache").fetchone()
        return int(row["total"] or 0), int(row["cached"] or 0)
    except Exception:
        return 0, 0


def ensure_posters_cached(data: pd.DataFrame, posters: dict[str, str]) -> dict[str, str]:
    """Ensure poster URLs from the dataset are populated in the posters mapping instantly."""
    if "Poster_URL" in data.columns:
        for _, row in data.iterrows():
            key = str(row.get("_anime_key", ""))
            p_url = row.get("Poster_URL")
            if key and key not in posters and pd.notna(p_url) and str(p_url).strip():
                posters[key] = str(p_url).strip()
    return posters

# ------------------------------ Recommendation -------------------------------
def text_field(data: pd.DataFrame, name: str) -> pd.Series:
    if name not in data.columns:
        return pd.Series([""] * len(data), index=data.index, dtype=str)
    return data[name].fillna("").astype(str).str.strip()


def build_recommender(data: pd.DataFrame):
    """Build a content index, tolerating missing/empty metadata columns."""
    genres = text_field(data, "Genres")
    themes = text_field(data, "Themes")
    descriptions = text_field(data, "Description")

    def fit_or_empty(corpus: pd.Series, vectorizer: TfidfVectorizer):
        try:
            return vectorizer.fit_transform(corpus)
        except ValueError as exc:
            # TF-IDF raises "empty vocabulary" when a column is blank or has
            # no usable terms. Represent that feature family as zero columns;
            # do not let one sparse field stop the whole app.
            if "empty vocabulary" in str(exc).lower() or "no terms remain" in str(exc).lower() or "max_df corresponds" in str(exc).lower():
                return csr_matrix((len(corpus), 0), dtype=float)
            raise

    # MAL metadata can be missing or inconsistently formatted. Try the intended
    # comma-separated genre/theme tokens first, then use generic tokenization.
    genre_matrix = fit_or_empty(
        genres,
        TfidfVectorizer(
            tokenizer=lambda x: [g.strip().casefold() for g in x.split(",") if g.strip()],
            token_pattern=None, lowercase=False
        )
    )
    if genre_matrix.shape[1] == 0:
        genre_matrix = fit_or_empty(genres, TfidfVectorizer(token_pattern=r"(?u)\b\w+\b"))
    genre_matrix = genre_matrix * 2.5

    theme_matrix = fit_or_empty(
        themes,
        TfidfVectorizer(
            tokenizer=lambda x: [g.strip().casefold() for g in x.split(",") if g.strip()],
            token_pattern=None, lowercase=False
        )
    )
    if theme_matrix.shape[1] == 0:
        theme_matrix = fit_or_empty(themes, TfidfVectorizer(token_pattern=r"(?u)\b\w+\b"))
    theme_matrix = theme_matrix * 3.0

    desc_matrix = fit_or_empty(
        descriptions,
        TfidfVectorizer(stop_words="english", max_features=10000, min_df=2, ngram_range=(1, 2), max_df=0.85)
    )
    if desc_matrix.shape[1] == 0:
        desc_matrix = fit_or_empty(descriptions, TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=10000))
    desc_matrix = desc_matrix * 0.5

    feature_parts = [part for part in (genre_matrix, theme_matrix, desc_matrix) if part.shape[1] > 0]
    if feature_parts:
        matrix = normalize(hstack(feature_parts).tocsr())
    else:
        # Last-resort fallback: use anime titles to create a usable index when
        # all metadata fields are blank or absent from the source CSV.
        titles = text_field(data, "Title")
        title_matrix = fit_or_empty(titles, TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2)))
        if title_matrix.shape[1] == 0:
            # A one-column zero matrix still lets the UI load; recommendations
            # will be weak until the dataset's text columns are corrected.
            title_matrix = csr_matrix((len(data), 1), dtype=float)
        matrix = normalize(title_matrix.tocsr())

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(matrix)
    return matrix, nn


def best_title_match(query: str, titles: list[str]) -> tuple[int | None, float]:
    query = query.strip()
    if not query or not titles:
        return None, 0.0
    exact_map = {t.casefold(): i for i, t in enumerate(titles)}
    if query.casefold() in exact_map:
        return exact_map[query.casefold()], 100.0
    if rapidfuzz_process is not None:
        match = rapidfuzz_process.extractOne(query, titles, scorer=rapidfuzz_fuzz.WRatio)
        if match:
            return int(match[2]), float(match[1])
    choices = get_close_matches(query.casefold(), [t.casefold() for t in titles], n=1, cutoff=0.0)
    if not choices:
        return None, 0.0
    matched = choices[0]
    idx = next(i for i, t in enumerate(titles) if t.casefold() == matched)
    return idx, SequenceMatcher(None, query.casefold(), titles[idx].casefold()).ratio() * 100


def recommend_from_index(data: pd.DataFrame, matrix, nn_model, source_idx: int) -> pd.DataFrame:
    """Full ranked candidate list (excluding the source itself), nearest first."""
    distances, indices = nn_model.kneighbors(matrix[source_idx], n_neighbors=len(data))
    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if int(idx) == source_idx:
            continue
        row = data.iloc[int(idx)]
        title = str(row.get("Title", "")).strip()
        if not title:
            continue
        similarity = max(0.0, 1.0 - float(distance))
        item = row.copy()
        item["Similarity (%)"] = round(similarity * 100, 1)
        results.append(item)
    return pd.DataFrame(results)


_SEQUEL_MARKERS = re.compile(
    r"\b(season\s*\d*|part\s*\d*|cour\s*\d*|movie|ova|ona|specials?|recap|"
    r"final season|\d+(?:st|nd|rd|th)\s*season|sequel|zoku|kou-?hen|zenpen|kouhen)\b",
    re.IGNORECASE,
)


def franchise_stem(title: str) -> str:
    """Collapse a title to its base-franchise name so sequels/seasons/movies match."""
    # ponytail: splitting on any hyphen (not just a subtitle separator) over-collapses
    # titles like "Mushi-shi" or "86 Eighty-Six"; harmless since it's applied the same
    # way to sequels and the base title, but tighten further if that ever merges two
    # unrelated franchises. Colon only counts as a subtitle separator when followed by
    # a space, so a bare-colon title like "Re:ZERO" isn't truncated to "Re".
    t = re.split(r": |[\-–—]", str(title), maxsplit=1)[0]  # drop subtitle after ": " or a dash
    t = _SEQUEL_MARKERS.sub("", t)
    t = re.sub(r"\b[IVX]{1,4}\b", "", t)  # roman numerals as standalone words (II, III, IV...)
    t = re.sub(r"\d+", "", t)  # stray numbers like "2"
    t = re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    return t

# ---------------------------------- Auth -------------------------------------
def password_digest(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return salt.hex() + "$" + derived.hex()


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 310_000)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def create_token(user_id: int, username: str) -> str:
    if jwt is None:
        raise RuntimeError("JWT auth is enabled but PyJWT is not installed. Install PyJWT.")
    if not JWT_SECRET or JWT_SECRET == "change-me":
        raise RuntimeError("Set ANISTAT_JWT_SECRET to a long random secret before enabling auth.")
    return jwt.encode({"sub": str(user_id), "username": username, "iat": int(time.time()), "exp": int(time.time()) + 60 * 60 * 24 * 7}, JWT_SECRET, algorithm="HS256")


def verify_token(token: str):
    if jwt is None or not JWT_SECRET:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return {"user_id": int(payload["sub"]), "username": payload["username"]}
    except Exception:
        return None


def render_auth_and_header() -> None:
    token = st.session_state.get("jwt_token") or st.query_params.get("token")
    user = st.session_state.get("user")
    if token and not user:
        user = verify_token(token)
        if user:
            st.session_state.user = user
            st.session_state.jwt_token = token
            st.query_params["token"] = token
        else:
            st.session_state.pop("jwt_token", None)
            if "token" in st.query_params:
                del st.query_params["token"]

    prof_label = f"![Profile]({_profile_icon})" if _profile_icon else "Profile"
    hist_label = f"![History]({_history_icon})" if _history_icon else "History"

    if user:
        col_prof, col_hist, col_brand = st.columns([0.12, 0.12, 1.6], vertical_alignment="center", gap="small")
        with col_prof:
            with st.popover(prof_label):
                st.markdown(f"### hello, {user['username']}")
                st.markdown("<span style='color:#00e676;font-size:0.8rem;font-weight:700;'>● Active Session</span>", unsafe_allow_html=True)
                if st.button("Log out", key="logout_btn", type="secondary", use_container_width=True):
                    st.session_state.user = None
                    st.session_state.pop("jwt_token", None)
                    if "token" in st.query_params:
                        del st.query_params["token"]
                    st.rerun()
        with col_hist:
            with st.popover(hist_label):
                st.markdown("###Search History")
                history = get_search_history(user["user_id"])
                if not history:
                    st.info("No past searches yet. Search for any anime to see it here!")
                else:
                    for item in history:
                        q_text = item["query"]
                        m_text = item.get("matched_title")
                        label = f"{q_text}"
                        if m_text and m_text.casefold() != q_text.casefold():
                            label += f" ({m_text})"
                        if st.button(label, key=f"hist_btn_{item['history_id']}", use_container_width=True):
                            st.session_state["anime_query"] = q_text
                            st.session_state["last_search_query"] = q_text
                            st.rerun()
                    st.markdown("---")
                    if st.button("Clear history", key="clear_hist_btn", use_container_width=True):
                        clear_search_history(user["user_id"])
                        st.rerun()
        with col_brand:
            st.markdown('<div class="brand">ANISTAT <span style="color:#f5f5f5;font-weight:800;letter-spacing:0">/ YOUR NEXT FAVORITE ANIME</span></div>', unsafe_allow_html=True)
    else:
        col_prof, col_brand = st.columns([0.12, 1.6], vertical_alignment="center", gap="small")
        with col_prof:
            with st.popover(prof_label):
                st.markdown("### Account Access")
                st.caption("Sign in with your SQLite credentials or create an account.")
                mode = st.radio("Mode", ["Login", "Sign up"], horizontal=True, label_visibility="collapsed", key="auth_mode")
                username = st.text_input("Username", key="auth_username_pop").strip()
                password = st.text_input("Password", type="password", key="auth_password_pop")
                if st.button(f"{mode}", type="primary", use_container_width=True, key="submit_auth_btn"):
                    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
                        st.error("Username must be 3–32 characters: letters, numbers, dot, underscore or hyphen.")
                    elif len(password) < 6:
                        st.error("Use a password with at least 6 characters.")
                    else:
                        with get_conn() as conn:
                            existing = conn.execute("SELECT * FROM users WHERE username=? COLLATE NOCASE", (username,)).fetchone()
                            if mode == "Sign up":
                                if existing:
                                    st.error("That username is already registered.")
                                else:
                                    cur = conn.execute("INSERT INTO users(username,password_hash) VALUES(?,?)", (username, password_digest(password)))
                                    uid = int(cur.lastrowid)
                                    token = create_token(uid, username)
                                    st.session_state.user = {"user_id": uid, "username": username}
                                    st.session_state.jwt_token = token
                                    st.query_params["token"] = token
                                    st.rerun()
                            else:
                                if not existing or not verify_password(password, existing["password_hash"]):
                                    st.error("Incorrect username or password.")
                                else:
                                    uid = int(existing["user_id"])
                                    token = create_token(uid, existing["username"])
                                    st.session_state.user = {"user_id": uid, "username": existing["username"]}
                                    st.session_state.jwt_token = token
                                    st.query_params["token"] = token
                                    st.rerun()
        with col_brand:
            st.markdown(
             '<div class="brand" style="font-size: 32px;">ANISTAT <span style="color:#f5f5f5;font-weight:800;letter-spacing:0;">/ YOUR NEXT FAVORITE ANIME</span></div>', 
                unsafe_allow_html=True
            )
    
# ---------------------------------- UI utils ---------------------------------
def display_title(row) -> str:
    return normalize_value(row.get("Title", "Untitled")) or "Untitled"


@st.cache_resource
def load_score_predictor():
    if joblib is None:
        raise RuntimeError("The joblib package is not installed.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Score model not found at {MODEL_PATH}.")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Model feature metadata not found at {FEATURES_PATH}.")
    # The artifact was trained with scikit-learn 1.6.1. Streamlit Cloud now
    # provisions Python 3.14, whose available wheel is scikit-learn 1.9.x.
    # Restore the renamed private pickle class before loading the pipeline.
    import sklearn.compose._column_transformer as column_transformer

    if not hasattr(column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        _RemainderColsList.__module__ = column_transformer.__name__
        column_transformer._RemainderColsList = _RemainderColsList

    model = joblib.load(MODEL_PATH)
    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    for _, transformer, _ in getattr(preprocessor, "transformers_", []):
        if hasattr(transformer, "_fit_dtype") and not hasattr(transformer, "_fill_dtype"):
            transformer._fill_dtype = transformer._fit_dtype
    feature_columns = joblib.load(FEATURES_PATH)
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("The model feature metadata must be a non-empty list.")
    return model, feature_columns


def render_score_predictor() -> None:
    st.markdown('<div class="section-title">Anime score predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Estimate a title\'s score from its metadata using the AniStat model.</div>',
        unsafe_allow_html=True,
    )
    try:
        model, feature_columns = load_score_predictor()
    except (FileNotFoundError, ImportError, AttributeError, ValueError, RuntimeError) as exc:
        st.warning(f"Score prediction is unavailable: {exc}")
        return

    with st.container(border=True):
        with st.form("anime_score_predictor_form"):
            left, right = st.columns(2, gap="large")
            with left:
                genres = st.text_input("Genres", placeholder="Action, Adventure, Fantasy")
                themes = st.text_input("Themes", placeholder="School, Military, Time Travel")
                description = st.text_area(
                    "Synopsis",
                    placeholder="Describe the anime's story and premise.",
                    height=140,
                )
                demographics = st.text_input("Demographics", placeholder="Shounen, Seinen")
            with right:
                anime_type = st.selectbox("Anime type", ["TV", "Movie", "OVA", "ONA", "Special", "Music"])
                status = st.selectbox(
                    "Airing status",
                    ["Finished Airing", "Currently Airing", "Not yet aired"],
                )
                episodes = st.number_input("Episodes", min_value=1, max_value=2000, value=12, step=1)
                submitted = st.form_submit_button("Predict score", type="primary", use_container_width=True)

            if submitted:
                values = {
                    "Genres": genres.strip(),
                    "Themes": themes.strip(),
                    "Description": description.strip(),
                    "Type": anime_type,
                    "Status": status,
                    "Demographics": demographics.strip(),
                    "Episodes": episodes,
                }
                missing_columns = [column for column in feature_columns if column not in values]
                if missing_columns:
                    st.error(f"The model expects unsupported fields: {', '.join(missing_columns)}.")
                else:
                    input_data = pd.DataFrame([{column: values[column] for column in feature_columns}])
                    try:
                        prediction = float(model.predict(input_data)[0])
                    except (AttributeError, KeyError, TypeError, ValueError) as exc:
                        st.error(f"Could not predict a score with this model: {exc}")
                    else:
                        prediction = max(0.0, min(10.0, prediction))
                        st.markdown(
                            f'<div class="predictor-result"><span>Estimated score</span><strong>{prediction:.2f} / 10</strong></div>',
                            unsafe_allow_html=True,
                        )


def display_date(value) -> str:
    value = normalize_value(value)
    if not value or value.lower() in {"nan", "nat", "none"}:
        return "Unknown"
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%b %Y")
    except Exception:
        pass
    return value[:30]


def safe_poster_url(url) -> str | None:
    """Only allow https MyAnimeList/Jikan and AniList CDN image URLs into raw HTML."""
    if not url:
        return None
    try:
        parsed = urlparse(str(url))
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        return None
    allowed_domains = ("myanimelist.net", "anilist.co")
    if not any(host == d or host.endswith("." + d) for d in allowed_domains):
        return None
    return str(url)


def card_grid(data: pd.DataFrame, posters: dict[str, str], key_prefix: str, show_similarity: bool = False, user=None):
    if data.empty:
        st.info("No anime found for this query.")
        return
    columns = st.columns(5, gap="medium")
    for i, (_, row) in enumerate(data.iterrows()):
        col = columns[i % len(columns)]
        key = str(row.get("_anime_key", ""))
        title = display_title(row)
        safe_title = html.escape(title, quote=True)
        raw_poster = posters.get(key) or (str(row.get("Poster_URL", "")).strip() if pd.notna(row.get("Poster_URL")) else None)
        poster = safe_poster_url(raw_poster)
        with col:
            if poster:
                st.markdown(f'<div class="anime-card"><img src="{html.escape(poster, quote=True)}" alt="{safe_title}" loading="lazy" decoding="async"><div class="card-body"><div class="card-title">{safe_title}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="anime-card"><div class="poster-placeholder">◉</div><div class="card-body"><div class="card-title">{safe_title}</div>', unsafe_allow_html=True)
            score = row.get("Score")
            score_text = f"{float(score):.2f}" if pd.notna(score) and str(score).strip() else "N/A"
            typ = normalize_value(row.get("Type", "")) or "Anime"
            genres = normalize_value(row.get("Genres", ""))
            if len(genres) > 72:
                genres = genres[:69] + "…"
            start = display_date(row.get("Start Date", ""))
            end = display_date(row.get("End Date", ""))
            similarity = row.get("Similarity (%)")
            safe_typ = html.escape(typ, quote=True)
            safe_genres = html.escape(genres, quote=True) if genres else ""
            safe_start = html.escape(start, quote=True)
            safe_end = html.escape(end, quote=True)
            st.markdown(f'<div class="card-meta"><span class="score">★ {score_text}</span> {safe_typ}<br> {safe_genres or "Genres unavailable"}<br> {safe_start} – {safe_end}</div>', unsafe_allow_html=True)
            if show_similarity and pd.notna(similarity):
                st.caption(f"Similarity · {float(similarity):.1f}%")
            st.markdown("</div></div>", unsafe_allow_html=True)
           


def is_favorite(user_id: int, anime_key: str) -> bool:
    with get_conn() as conn:
        return conn.execute("SELECT 1 FROM favorites WHERE user_id=? AND anime_key=?", (user_id, anime_key)).fetchone() is not None


def toggle_favorite(user_id: int, anime_key: str) -> None:
    with get_conn() as conn:
        existing = conn.execute("SELECT 1 FROM favorites WHERE user_id=? AND anime_key=?", (user_id, anime_key)).fetchone()
        if existing:
            conn.execute("DELETE FROM favorites WHERE user_id=? AND anime_key=?", (user_id, anime_key))
        else:
            conn.execute("INSERT OR IGNORE INTO favorites(user_id, anime_key) VALUES(?,?)", (user_id, anime_key))

# ---------------------------------- Main app ---------------------------------
if not SQLITECLOUD_CONNECTION_STRING:
    st.error(
        "SQLite Cloud is not configured. Add `SQLITECLOUD_CONNECTION_STRING` to "
        "`.streamlit/secrets.toml` or your environment variables. Keep the API key secret."
    )
    st.code(
        'SQLITECLOUD_CONNECTION_STRING = "sqlitecloud://cuw6zxarvk.g4.sqlite.cloud:8860/auth.sqlitecloud?apikey=YOUR_API_KEY"',
        language="toml",
    )
    st.stop()

try:
    init_db()
except Exception as exc:
    st.error(
        "Could not connect to SQLite Cloud or initialize AniStat tables. "
        "Verify the connection string, API key, database name, and network access."
    )
    st.exception(exc)
    st.stop()

df = load_csv()
if not any(df[col].fillna("").astype(str).str.strip().ne("").any() for col in ["Genres", "Themes", "Description"]):
    st.warning("The CSV has no usable text in Genres, Themes, or Description. Similarity search will fall back to anime titles. Check that the correct dataset is in the dataset folder and that these column names match your CSV.")
# Avoid rewriting the entire SQLite catalog on every Streamlit widget rerun.
try:
    _dataset_signature = f"{DATASET_PATH.stat().st_mtime_ns}:{len(df)}"
except OSError:
    _dataset_signature = str(len(df))
if st.session_state.get("catalog_sync_signature") != _dataset_signature:
    sync_catalog_to_db(df)
    st.session_state["catalog_sync_signature"] = _dataset_signature

@st.cache_data(show_spinner="Preparing anime similarity index…")
def cached_recommender(data_for_model: pd.DataFrame):
    return build_recommender(data_for_model)

# Use only fields that affect similarity. Stable row order is retained within this app run.
model_data = df.copy().reset_index(drop=True)
try:
    feature_matrix, nn_model = cached_recommender(model_data[["Title", "Genres", "Themes", "Description"]].copy())
except Exception as exc:
    st.error(f"Could not build the anime similarity index: {exc}")
    st.stop()

# Restore the original rows to the same order used by the feature matrix.
titles = model_data["Title"].astype(str).tolist()
posters = {
    str(row["_anime_key"]): str(row["Poster_URL"])
    for _, row in model_data.iterrows()
    if pd.notna(row.get("Poster_URL")) and str(row.get("Poster_URL")).strip()
}
try:
    for k, v in read_poster_cache().items():
        if v and k not in posters:
            posters[k] = v
except Exception:
    pass

render_auth_and_header()

hero_bg_tag = f'<img class="hero-bg" src="{_hero_image}" alt="AniStat Banner">' if _hero_image else ''
st.markdown(f"""
<div class="hero">
  {hero_bg_tag}
  <div class="hero-overlay"></div>
  <div class="hero-content">
    <div class="eyebrow">Discover something new</div>
    <h1>Find your next obsession.</h1>
    <p>Search for an anime and get recommendations based on its genres, themes and synopsis. Typos are okay — try a title even if you don't remember the exact spelling.</p>
  </div>
</div>
""", unsafe_allow_html=True)

query = st.text_input(
    "Anime search",
    placeholder="An anime similar to..",
    label_visibility="collapsed",
    key="anime_query",
)
col_btn, col_orig = st.columns([0.24, 0.76], vertical_alignment="center")
with col_btn:
    search_clicked = st.button("Find similar anime", type="primary", use_container_width=True)
with col_orig:
    originals_only = st.checkbox(
        "Originals only",
        value=True,
        key="originals_only",
        help="Hide sequels, seasons, movies and spin-offs of the same series.",
    )

if query.strip() and (search_clicked or st.session_state.get("last_search_query") == query.strip()):
    match_idx, match_score = best_title_match(query, titles)
    if match_idx is None or match_score < 45:
        st.warning("I couldn't find a confident title match. Try a shorter or more recognizable part of the anime title.")
        if titles:
            st.caption("Closest matches: " + " · ".join(titles[i] for i in range(min(5, len(titles)))))
    else:
        st.session_state.last_search_query = query.strip()
        source = model_data.iloc[match_idx]
        if st.session_state.get("user"):
            record_search_history(st.session_state["user"]["user_id"], query.strip(), str(source["Title"]))
        if match_score < 80:
            st.info(f"Matching **{query}** to **{source['Title']}** ({match_score:.0f}% title match).")
        else:
            st.markdown(f"### Because you searched for **{source['Title']}**")
        source_key = str(source["_anime_key"])
        posters = ensure_posters_cached(model_data.iloc[[match_idx]], posters)
        raw_source_poster = posters.get(source_key) or (str(source.get("Poster_URL", "")).strip() if pd.notna(source.get("Poster_URL")) else None)
        source_poster = safe_poster_url(raw_source_poster)
        left, right = st.columns([1, 4], gap="large")
        with left:
            if source_poster:
                st.markdown(
                    f'<div class="anime-card" style="margin-bottom:0;"><img src="{html.escape(source_poster, quote=True)}" alt="{html.escape(str(source["Title"]), quote=True)}" loading="eager" decoding="async"></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="poster-placeholder">◉</div>', unsafe_allow_html=True)
        with right:
            st.subheader(str(source["Title"]))
            score = source.get("Score")
            st.markdown(f"**Score:** {float(score):.2f} / 10" if pd.notna(score) else "**Score:** Not available")
            st.write(f"**Type:** {normalize_value(source.get('Type')) or 'Unknown'}  ·  **Episodes:** {normalize_value(source.get('Episodes')) or 'Unknown'}")
            st.write(f"**Genres:** {normalize_value(source.get('Genres')) or 'Unknown'}")
            st.write(f"**Themes:** {normalize_value(source.get('Themes')) or 'Unknown'}")
            st.write(f"**Aired:** {display_date(source.get('Start Date'))} – {display_date(source.get('End Date'))}")
            desc = normalize_value(source.get("Description"))
            if desc:
                st.write(desc[:900] + ("…" if len(desc) > 900 else ""))
        recs_all = recommend_from_index(model_data, feature_matrix, nn_model, match_idx)
        if originals_only:
            source_stem = franchise_stem(source["Title"])
            recs_all = recs_all[recs_all["Title"].map(lambda t: franchise_stem(t) != source_stem)]

        # Reset to the first page whenever the search or the filter toggle changes.
        page_key = (source_key, originals_only)
        if st.session_state.get("_recs_page_key") != page_key:
            st.session_state["_recs_page_key"] = page_key
            st.session_state["recs_shown"] = 10
        shown = st.session_state["recs_shown"]
        recs = recs_all.head(shown)

        posters = ensure_posters_cached(recs, posters)
        st.markdown('<div class="section-title">Similar anime</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Ranked by content similarity, not by predicted personal preference.</div>', unsafe_allow_html=True)
        card_grid(recs, posters, "recommendations", show_similarity=True, user=st.session_state.get("user"))
        if len(recs_all) > shown:
            if st.button("Load 10 more"):
                st.session_state["recs_shown"] += 10
                st.rerun()
else:
    # Default home: highest-scored anime, with stable title tie-breaker.
    st.markdown('<div class="section-title">Top-rated anime</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">The highest-scored titles in your local dataset.</div>', unsafe_allow_html=True)
    home = model_data.copy()
    home["Score"] = pd.to_numeric(home["Score"], errors="coerce")
    home = home.dropna(subset=["Score"]).sort_values(["Score", "Rank"], ascending=[False, True], na_position="last").head(25)
    posters = ensure_posters_cached(home, posters)
    card_grid(home, posters, "home", user=st.session_state.get("user"))
    render_score_predictor()
    st.markdown("---")
    st.markdown('<div class="small-muted">Tip: search for a title to see similar anime. Typos are okay — try a title even if you don\'t remember the exact spelling.</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("AniStat · Local CSV catalog + SQLite poster cache · Similarity uses TF-IDF cosine similarity")
