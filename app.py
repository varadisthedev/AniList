"""AniStat: anime discovery and content-based recommendations.

Expected project layout:
AniList/
  app.py
  dataset/myanimelist_with_all_metadata_complete.csv
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
import hashlib
import sqlite3
from pathlib import Path
from urllib.parse import urlparse
from difflib import SequenceMatcher, get_close_matches

import pandas as pd
import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

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
DATASET_PATH = Path(os.getenv("ANISTAT_DATASET", str(ROOT / "dataset" / "myanimelist_with_all_metadata_complete.csv")))
DB_DIR = ROOT / "db"
DB_PATH = Path(os.getenv("ANISTAT_DB_PATH", str(DB_DIR / "anistat.db")))
MODEL_PATH = ROOT / "models" / "anistat_score_regressor.joblib"
FEATURES_PATH = ROOT / "models" / "anistat_feature_columns.joblib"
AUTH_ENABLED = os.getenv("ANISTAT_AUTH_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
JWT_SECRET = os.getenv("ANISTAT_JWT_SECRET", "")
JIKAN_BASE = "https://api.jikan.moe/v4"
POSTER_BATCH_SIZE = 25

st.set_page_config(page_title="AniStat · Discover Anime", page_icon="◉", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #141414; color: #f5f5f5; }
[data-testid="stHeader"] { background: rgba(20,20,20,.96); }
[data-testid="stSidebar"] { background: #1f1f1f; border-right: 1px solid #303030; }
.block-container { padding-top: 1.6rem; max-width: 1600px; }
.brand { font-size: 1.15rem; font-weight: 900; letter-spacing: .13em; color: #ff5e00; }
.hero { background: linear-gradient(90deg, rgba(20,20,20,.98), rgba(20,20,20,.80) 55%, rgba(20,20,20,.20)), linear-gradient(0deg, #141414 0%, rgba(20,20,20,0) 65%); border-radius: 18px; padding: 2rem 2rem 1.5rem; border: 1px solid #303030; margin-bottom: 1.2rem; }
.hero h1 { font-size: clamp(2rem, 4vw, 3.4rem); margin: 0; line-height: 1.1; }
.hero p { color: #d0d0d0; max-width: 720px; }
.eyebrow { color: #ff5e00; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; font-size: .78rem; }
.section-title { font-size: 1.5rem; font-weight: 800; margin: 1.5rem 0 .2rem; }
.section-sub { color: #aaa; margin-bottom: 1rem; }
.anime-card { background: #1f1f1f; border: 1px solid #303030; border-radius: 12px; overflow: hidden; height: 100%; margin-bottom: .8rem; }
.anime-card img { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; background: #292929; }
.poster-placeholder { aspect-ratio: 2/3; background: linear-gradient(135deg,#282828,#101010); display:flex; align-items:center; justify-content:center; color:#777; font-size:3rem; }
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

# ---------------------------------- SQLite -----------------------------------
def get_conn() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS anime_catalog (
            anime_key TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS poster_cache (
            anime_key TEXT PRIMARY KEY,
            mal_id INTEGER,
            poster_url TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY(anime_key) REFERENCES anime_catalog(anime_key) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS favorites (
            user_id INTEGER NOT NULL,
            anime_key TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(user_id, anime_key),
            FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY(anime_key) REFERENCES anime_catalog(anime_key) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS user_ratings (
            user_id INTEGER NOT NULL,
            anime_key TEXT NOT NULL,
            rating REAL NOT NULL CHECK(rating >= 1 AND rating <= 10),
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(user_id, anime_key),
            FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY(anime_key) REFERENCES anime_catalog(anime_key) ON DELETE CASCADE
        );
        """)


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

    # Support the notebook's cleaned schema and common variations.
    title_col = first_existing(data.columns, ["Title", "English Title", "Anime Title", "Name"])
    if title_col is None:
        st.error(f"Could not find a title column in {DATASET_PATH.name}. Columns: {list(data.columns)}")
        st.stop()
    if title_col != "Title":
        data = data.rename(columns={title_col: "Title"})
    for col in ["Genres", "Themes", "Description", "Type", "Status", "Demographics", "URL", "Score", "Episodes", "Start Date", "End Date", "Rank"]:
        if col not in data.columns:
            data[col] = "" if col not in {"Score", "Episodes", "Rank"} else pd.NA
    for col in ["Title", "Genres", "Themes", "Description", "Type", "Status", "Demographics", "URL", "Start Date", "End Date"]:
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
    # Refresh metadata while preserving cached posters and user data.
    with get_conn() as conn:
        for _, row in data.iterrows():
            key = str(row["_anime_key"])
            metadata = {str(k): (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.drop(labels=["_anime_key"]).to_dict().items()}
            metadata_json = json.dumps(metadata, ensure_ascii=False, default=str)
            conn.execute("INSERT INTO anime_catalog(anime_key,title,metadata_json) VALUES(?,?,?) ON CONFLICT(anime_key) DO UPDATE SET title=excluded.title, metadata_json=excluded.metadata_json", (key, str(row["Title"]), metadata_json))
            conn.execute("INSERT OR IGNORE INTO poster_cache(anime_key,mal_id,poster_url,status) VALUES(?,?,NULL,?)", (key, extract_mal_id(row.get("URL", "")), "pending"))


def read_poster_cache() -> dict[str, str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT anime_key, poster_url FROM poster_cache WHERE poster_url IS NOT NULL AND TRIM(poster_url) <> ''").fetchall()
    return {row["anime_key"]: row["poster_url"] for row in rows}


def get_poster_url(anime_key: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute("SELECT poster_url FROM poster_cache WHERE anime_key=?", (anime_key,)).fetchone()
    return row["poster_url"] if row and row["poster_url"] else None


def set_poster(anime_key: str, url: str | None, status: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE poster_cache SET poster_url=?, status=?, fetched_at=CURRENT_TIMESTAMP WHERE anime_key=?", (url, status, anime_key))


def poster_cache_counts() -> tuple[int, int]:
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS total, SUM(CASE WHEN poster_url IS NOT NULL AND TRIM(poster_url)<>'' THEN 1 ELSE 0 END) AS cached FROM poster_cache").fetchone()
    return int(row["total"] or 0), int(row["cached"] or 0)


def fetch_poster_for_mal_id(mal_id: int) -> str | None:
    # Use the API, not the MAL ranking page. API requests are rate-limited and cached in SQLite.
    response = requests.get(f"{JIKAN_BASE}/anime/{mal_id}", timeout=12, headers={"User-Agent": "AniStat/1.0 (anime poster metadata lookup)"})
    if response.status_code == 429:
        time.sleep(2.0)
        response = requests.get(f"{JIKAN_BASE}/anime/{mal_id}", timeout=12, headers={"User-Agent": "AniStat/1.0"})
    response.raise_for_status()
    payload = response.json().get("data", {})
    images = payload.get("images", {}).get("jpg", {})
    return images.get("large_image_url") or images.get("image_url") or None


def populate_poster_cache(data: pd.DataFrame, limit: int = POSTER_BATCH_SIZE) -> tuple[int, int]:
    with get_conn() as conn:
        rows = conn.execute("""SELECT p.anime_key, p.mal_id FROM poster_cache p
            WHERE (p.poster_url IS NULL OR TRIM(p.poster_url)='') AND p.status != 'not_found'
            ORDER BY CASE WHEN p.status='pending' THEN 0 ELSE 1 END, p.anime_key LIMIT ?""", (limit,)).fetchall()
    updated = 0
    progress = st.progress(0, text="Fetching poster URLs from Jikan…")
    status_box = st.empty()
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        key, mal_id = row["anime_key"], row["mal_id"]
        if not mal_id:
            set_poster(key, None, "not_found")
        else:
            try:
                poster = fetch_poster_for_mal_id(int(mal_id))
                set_poster(key, poster, "cached" if poster else "not_found")
                updated += 1 if poster else 0
            except requests.RequestException as exc:
                # Leave as pending so the user can retry later; avoid hammering API.
                with get_conn() as conn:
                    conn.execute("UPDATE poster_cache SET status='pending' WHERE anime_key=?", (key,))
                status_box.caption(f"Temporary API issue at item {i}: {exc}. Stopping this batch so the API isn't hammered.")
                break
            except (ValueError, KeyError, TypeError):
                set_poster(key, None, "not_found")
        progress.progress(i / max(total, 1), text=f"Checking poster {i}/{total}…")
        time.sleep(0.38)  # conservative pacing for public API
    progress.empty()
    return updated, total

# ------------------------------ Recommendation -------------------------------
def text_field(data: pd.DataFrame, name: str) -> pd.Series:
    if name not in data.columns:
        return pd.Series([""] * len(data), index=data.index, dtype=str)
    return data[name].fillna("").astype(str).str.strip()


def build_recommender(data: pd.DataFrame):
    genres = text_field(data, "Genres")
    themes = text_field(data, "Themes")
    descriptions = text_field(data, "Description")
    genre_vec = TfidfVectorizer(
        tokenizer=lambda x: [g.strip().casefold() for g in x.split(",") if g.strip()],
        token_pattern=None, lowercase=False
    )
    theme_vec = TfidfVectorizer(
        tokenizer=lambda x: [g.strip().casefold() for g in x.split(",") if g.strip()],
        token_pattern=None, lowercase=False
    )
    # Ensure empty / near-empty descriptions do not crash vectorization.
    try:
        genre_matrix = genre_vec.fit_transform(genres) * 2.5
    except ValueError:
        genre_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        genre_matrix = genre_vec.fit_transform(genres) * 2.5
    try:
        theme_matrix = theme_vec.fit_transform(themes) * 3.0
    except ValueError:
        theme_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        theme_matrix = theme_vec.fit_transform(themes) * 3.0
    try:
        desc_vec = TfidfVectorizer(stop_words="english", max_features=10000, min_df=2, ngram_range=(1, 2), max_df=0.85)
        desc_matrix = desc_vec.fit_transform(descriptions) * 0.5
    except ValueError:
        desc_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        desc_matrix = desc_vec.fit_transform(descriptions) * 0.5
    matrix = normalize(hstack([genre_matrix, theme_matrix, desc_matrix]).tocsr())
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


def recommend_from_index(data: pd.DataFrame, matrix, nn_model, source_idx: int, n: int = 12) -> pd.DataFrame:
    distances, indices = nn_model.kneighbors(matrix[source_idx], n_neighbors=len(data))
    results = []
    source_title = str(data.iloc[source_idx]["Title"]).casefold()
    for distance, idx in zip(distances[0], indices[0]):
        if int(idx) == source_idx:
            continue
        row = data.iloc[int(idx)]
        title = str(row.get("Title", "")).strip()
        if not title:
            continue
        # Avoid obvious sequel/recap matches with the exact full title prefix; still allow thematic matches.
        similarity = max(0.0, 1.0 - float(distance))
        item = row.copy()
        item["Similarity (%)"] = round(similarity * 100, 1)
        results.append(item)
        if len(results) >= n:
            break
    return pd.DataFrame(results)

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


def auth_panel() -> None:
    if not AUTH_ENABLED:
        st.sidebar.caption("Guest mode · login is optional")
        st.session_state.setdefault("user", None)
        return
    if jwt is None:
        st.sidebar.error("JWT auth enabled, but PyJWT is missing. Install requirements.txt.")
        st.stop()
    if not JWT_SECRET:
        st.sidebar.error("Set ANISTAT_JWT_SECRET before enabling authentication.")
        st.stop()
    if "user" not in st.session_state:
        st.session_state.user = None
    if st.session_state.user:
        st.sidebar.success(f"Signed in as {st.session_state.user['username']}")
        if st.sidebar.button("Log out", use_container_width=True):
            st.session_state.user = None
            st.session_state.pop("jwt_token", None)
            st.rerun()
        return
    with st.sidebar.expander("Login / Sign up", expanded=True):
        mode = st.radio("Account", ["Login", "Sign up"], horizontal=True)
        username = st.text_input("Username", key="auth_username").strip()
        password = st.text_input("Password", type="password", key="auth_password")
        if st.button(mode, use_container_width=True):
            if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
                st.error("Username must be 3–32 characters: letters, numbers, dot, underscore or hyphen.")
            elif len(password) < 8:
                st.error("Use a password with at least 8 characters.")
            else:
                with get_conn() as conn:
                    user = conn.execute("SELECT * FROM users WHERE username=? COLLATE NOCASE", (username,)).fetchone()
                    if mode == "Sign up":
                        if user:
                            st.error("That username is already registered.")
                        else:
                            cur = conn.execute("INSERT INTO users(username,password_hash) VALUES(?,?)", (username, password_digest(password)))
                            uid = int(cur.lastrowid)
                            st.session_state.user = {"user_id": uid, "username": username}
                            st.session_state.jwt_token = create_token(uid, username)
                            st.rerun()
                    elif not user or not verify_password(password, user["password_hash"]):
                        st.error("Incorrect username or password.")
                    else:
                        st.session_state.user = {"user_id": int(user["user_id"]), "username": user["username"]}
                        st.session_state.jwt_token = create_token(int(user["user_id"]), user["username"])
                        st.rerun()

# ---------------------------------- UI utils ---------------------------------
def display_title(row) -> str:
    return normalize_value(row.get("Title", "Untitled")) or "Untitled"


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


def card_grid(data: pd.DataFrame, posters: dict[str, str], key_prefix: str, show_similarity: bool = False, user=None):
    if data.empty:
        st.info("No anime found for this query.")
        return
    columns = st.columns(5, gap="medium")
    for i, (_, row) in enumerate(data.iterrows()):
        col = columns[i % len(columns)]
        key = str(row.get("_anime_key", ""))
        title = display_title(row)
        poster = posters.get(key)
        with col:
            if poster:
                st.markdown(f'<div class="anime-card"><img src="{poster}" alt="{title.replace(chr(34), "")}" loading="lazy"><div class="card-body"><div class="card-title">{title}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="anime-card"><div class="poster-placeholder">◉</div><div class="card-body"><div class="card-title">{title}</div>', unsafe_allow_html=True)
            score = row.get("Score")
            score_text = f"{float(score):.2f}" if pd.notna(score) and str(score).strip() else "N/A"
            typ = normalize_value(row.get("Type", "")) or "Anime"
            genres = normalize_value(row.get("Genres", ""))
            if len(genres) > 72:
                genres = genres[:69] + "…"
            start = display_date(row.get("Start Date", ""))
            end = display_date(row.get("End Date", ""))
            similarity = row.get("Similarity (%)")
            st.markdown(f'<div class="card-meta"><span class="score">★ {score_text}</span> {typ}<br> {genres or "Genres unavailable"}<br> {start} – {end}</div>', unsafe_allow_html=True)
            if show_similarity and pd.notna(similarity):
                st.caption(f"Similarity · {float(similarity):.1f}%")
            st.markdown("</div></div>", unsafe_allow_html=True)
            if user:
                is_fav = is_favorite(user["user_id"], key)
                if st.button("♥ Saved" if is_fav else "♡ Add to list", key=f"{key_prefix}_fav_{key}", use_container_width=True):
                    toggle_favorite(user["user_id"], key)
                    st.rerun()
            elif AUTH_ENABLED:
                st.caption("Sign in to save")


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
init_db()
df = load_csv()
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
    feature_matrix, nn_model = cached_recommender(model_data[["Genres", "Themes", "Description"]].copy())
except Exception as exc:
    st.error(f"Could not build the anime similarity index: {exc}")
    st.stop()

# Restore the original rows to the same order used by the feature matrix.
titles = model_data["Title"].astype(str).tolist()
posters = read_poster_cache()

st.sidebar.markdown('<div class="brand">ANISTAT</div>', unsafe_allow_html=True)
st.sidebar.caption("Anime discovery · Similarity search · Your watchlist")
auth_panel()

with st.sidebar.expander("Poster image cache", expanded=False):
    total_posters, cached_posters = poster_cache_counts()
    st.write(f"Cached posters: **{cached_posters} / {total_posters}**")
    st.caption("Poster URLs are stored in SQLite and reused between app sessions. No ranking-page scraping is performed.")
    if st.button(f"Fetch next {POSTER_BATCH_SIZE} missing posters", use_container_width=True, disabled=cached_posters >= total_posters):
        with st.spinner("Caching poster URLs…"):
            added, checked = populate_poster_cache(df, POSTER_BATCH_SIZE)
        st.success(f"Cached {added} new poster URL(s). Checked {checked} entries.")
        st.rerun()
    if st.button("Refresh cached poster list", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown('<div class="brand">ANISTAT <span style="color:#f5f5f5;font-weight:500;letter-spacing:0">/ YOUR NEXT FAVORITE ANIME</span></div>', unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <div class="eyebrow">Discover something new</div>
  <h1>Find your next obsession.</h1>
  <p>Search for an anime and get recommendations based on its genres, themes and synopsis. Typos are okay — try a title even if you don't remember the exact spelling.</p>
</div>
""", unsafe_allow_html=True)

query = st.text_input("", placeholder="recommend anime similar to", label_visibility="collapsed", key="anime_query")
search_clicked = st.button("Find similar anime", type="primary", use_container_width=False)

if query.strip() and (search_clicked or st.session_state.get("last_search_query") == query.strip()):
    match_idx, match_score = best_title_match(query, titles)
    if match_idx is None or match_score < 45:
        st.warning("I couldn't find a confident title match. Try a shorter or more recognizable part of the anime title.")
        if titles:
            st.caption("Closest matches: " + " · ".join(titles[i] for i in range(min(5, len(titles)))))
    else:
        st.session_state.last_search_query = query.strip()
        source = model_data.iloc[match_idx]
        if match_score < 80:
            st.info(f"Matching **{query}** to **{source['Title']}** ({match_score:.0f}% title match).")
        else:
            st.markdown(f"### Because you searched for **{source['Title']}**")
        source_key = str(source["_anime_key"])
        source_poster = posters.get(source_key)
        left, right = st.columns([1, 4], gap="large")
        with left:
            if source_poster:
                st.image(source_poster, use_container_width=True)
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
        recs = recommend_from_index(model_data, feature_matrix, nn_model, match_idx, n=12)
        st.markdown('<div class="section-title">Similar anime</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Ranked by content similarity, not by predicted personal preference.</div>', unsafe_allow_html=True)
        card_grid(recs, posters, "recommendations", show_similarity=True, user=st.session_state.get("user"))
else:
    # Default home: highest-scored anime, with stable title tie-breaker.
    st.markdown('<div class="section-title">Top-rated anime</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">The highest-scored titles in your local dataset.</div>', unsafe_allow_html=True)
    home = model_data.copy()
    home["Score"] = pd.to_numeric(home["Score"], errors="coerce")
    home = home.dropna(subset=["Score"]).sort_values(["Score", "Rank"], ascending=[False, True], na_position="last").head(25)
    card_grid(home, posters, "home", user=st.session_state.get("user"))
    st.markdown("---")
    st.markdown('<div class="small-muted">Tip: search for a title to see similar anime. For posters that have not been cached yet, use “Fetch next missing posters” in the sidebar.</div>', unsafe_allow_html=True)

# Optional score prediction: only show when the model artifact exists and is loadable.
if joblib is not None and MODEL_PATH.exists():
    with st.expander("Experimental score estimate", expanded=False):
        st.caption("Model output is an estimate based on metadata, not an official score or guarantee.")
        selected_title = st.selectbox("Choose an anime from the catalog", model_data["Title"].tolist(), key="score_estimate_title")
        if st.button("Estimate score"):
            try:
                predictor = joblib.load(MODEL_PATH)
                features = joblib.load(FEATURES_PATH) if FEATURES_PATH.exists() else ["Genres", "Themes", "Description", "Type", "Status", "Demographics", "Episodes"]
                row = model_data.loc[model_data["Title"] == selected_title].iloc[0]
                prediction_input = pd.DataFrame([{col: row.get(col, "") for col in features}])[features]
                predicted = float(predictor.predict(prediction_input)[0])
                st.metric("Estimated score", f"{predicted:.2f} / 10")
            except Exception as exc:
                st.error(f"Could not run score prediction: {exc}")

st.markdown("---")
st.caption("AniStat · Local CSV catalog + SQLite poster cache · Similarity uses TF-IDF cosine similarity")
