"""One-off backfill: copy poster URLs from a Kaggle MyAnimeList export into poster_cache.

Usage: .venv\\Scripts\\python.exe scripts\\backfill_posters_from_kaggle.py [path\\to\\AnimeList.csv]
Source: https://www.kaggle.com/datasets/azathoth42/myanimelist (anime_id, image_url columns).
Re-run after swapping datasets or refreshing the Kaggle export; rows no longer 'pending' are untouched.
"""
import os
import sys
import tomllib
from pathlib import Path

import pandas as pd
import sqlitecloud

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "kaggle" / "AnimeList.csv"
# The Kaggle export's CDN ("myanimelist.cdn-dena.com") was decommissioned years ago;
# MAL kept the same image paths when it moved to this host.
OLD_CDN_HOST = "myanimelist.cdn-dena.com"
NEW_CDN_HOST = "cdn.myanimelist.net"


def connection_string() -> str:
    value = os.getenv("SQLITECLOUD_CONNECTION_STRING", "").strip()
    if value:
        return value
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        value = str(secrets.get("SQLITECLOUD_CONNECTION_STRING", "")).strip()
    if not value:
        raise SystemExit("Set SQLITECLOUD_CONNECTION_STRING or add it to .streamlit/secrets.toml")
    return value


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        raise SystemExit(f"Kaggle CSV not found: {csv_path}")

    kaggle = pd.read_csv(csv_path, low_memory=False, usecols=["anime_id", "image_url"]).dropna(subset=["image_url"])
    kaggle["image_url"] = kaggle["image_url"].str.replace(OLD_CDN_HOST, NEW_CDN_HOST, regex=False)
    url_by_mal_id = dict(zip(kaggle["anime_id"].astype(int), kaggle["image_url"]))

    conn = sqlitecloud.connect(connection_string())
    conn.row_factory = sqlitecloud.Row
    try:
        pending = conn.execute(
            "SELECT anime_key, mal_id FROM poster_cache WHERE status='pending' AND mal_id IS NOT NULL"
        ).fetchall()
        updates = [
            (url_by_mal_id[int(row["mal_id"])], str(row["anime_key"]))
            for row in pending
            if int(row["mal_id"]) in url_by_mal_id
        ]
        if updates:
            conn.executemany(
                "UPDATE poster_cache SET poster_url=?, status='cached', fetched_at=CURRENT_TIMESTAMP WHERE anime_key=?",
                updates,
            )
        print(
            f"Backfilled {len(updates)} poster(s) from {csv_path.name}; "
            f"{len(pending) - len(updates)} still pending (not in this export)."
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
