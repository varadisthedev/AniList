"""Enrich dataset/myanimelist_cleaned.csv with direct poster URLs from Kaggle and AniList API.
Also updates SQLite Cloud poster_cache table so both CSV and DB have complete image coverage.
"""
import re
import shutil
import time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "dataset" / "myanimelist_cleaned.csv"
KAGGLE_PATH = ROOT / "kaggle" / "AnimeList.csv"
BACKUP_PATH = ROOT / "dataset" / "myanimelist_cleaned.csv.bak"

OLD_CDN_HOST = "myanimelist.cdn-dena.com"
NEW_CDN_HOST = "cdn.myanimelist.net"


def extract_mal_id(url):
    m = re.search(r"myanimelist\.net/anime/(\d+)", str(url or ""), flags=re.I)
    return int(m.group(1)) if m else None


def main():
    if not BACKUP_PATH.exists():
        shutil.copy2(CSV_PATH, BACKUP_PATH)
        print(f"Backed up original dataset to {BACKUP_PATH.name}")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH.name}")

    df["mal_id"] = [extract_mal_id(u) for u in df["URL"]]

    # 1. Kaggle matches
    kaggle_map = {}
    if KAGGLE_PATH.exists():
        kaggle = pd.read_csv(KAGGLE_PATH, usecols=["anime_id", "image_url"]).dropna()
        kaggle["image_url"] = kaggle["image_url"].str.replace(OLD_CDN_HOST, NEW_CDN_HOST, regex=False)
        kaggle_map = dict(zip(kaggle["anime_id"].astype(int), kaggle["image_url"]))
        print(f"Loaded {len(kaggle_map)} anime images from Kaggle dataset")

    poster_urls = {}
    for mid in df["mal_id"]:
        if mid and mid in kaggle_map:
            poster_urls[mid] = kaggle_map[mid]

    print(f"Matched {len(poster_urls)} posters from Kaggle")

    # 2. AniList query for missing
    missing_ids = [mid for mid in df["mal_id"] if mid and mid not in poster_urls]
    print(f"Querying AniList for {len(missing_ids)} missing posters in batches...")

    query = """
    query($ids: [Int]) {
      Page(page: 1, perPage: 50) {
        media(idMal_in: $ids, type: ANIME) {
          idMal
          coverImage {
            large
          }
        }
      }
    }
    """

    chunk_size = 50
    for i in range(0, len(missing_ids), chunk_size):
        chunk = missing_ids[i:i + chunk_size]
        try:
            resp = requests.post(
                "https://graphql.anilist.co",
                json={"query": query, "variables": {"ids": chunk}},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {}).get("Page", {}).get("media", [])
                for item in data:
                    mid = item.get("idMal")
                    c_url = item.get("coverImage", {}).get("large")
                    if mid and c_url:
                        poster_urls[mid] = c_url
            else:
                print(f"AniList returned status {resp.status_code} for batch {i//chunk_size}")
        except Exception as e:
            print(f"Error fetching batch {i//chunk_size}: {e}")
        time.sleep(0.5)

    print(f"Total posters resolved: {len(poster_urls)} / {len(df)}")

    # Add Poster_URL column to df
    df["Poster_URL"] = [poster_urls.get(mid, "") for mid in df["mal_id"]]
    df = df.drop(columns=["mal_id"])

    df.to_csv(CSV_PATH, index=False)
    print(f"Saved updated dataset with Poster_URL column to {CSV_PATH.name}")

    # Also sync into SQLite Cloud if configured
    try:
        import tomllib
        import sqlitecloud
        sec_path = ROOT / ".streamlit" / "secrets.toml"
        if sec_path.exists():
            with open(sec_path, "rb") as f:
                sec = tomllib.load(f)
            conn_str = sec.get("SQLITECLOUD_CONNECTION_STRING")
            if conn_str:
                conn = sqlitecloud.connect(conn_str)
                conn.row_factory = sqlitecloud.Row
                updates = [
                    (poster_urls[mid], mid)
                    for mid in poster_urls
                ]
                conn.executemany(
                    "UPDATE poster_cache SET poster_url=?, status='cached', fetched_at=CURRENT_TIMESTAMP WHERE mal_id=?",
                    updates,
                )
                conn.close()
                print("Successfully synced updated poster URLs to SQLite Cloud poster_cache")
    except Exception as e:
        print(f"Note: Could not sync to SQLite Cloud (optional): {e}")


if __name__ == "__main__":
    main()
