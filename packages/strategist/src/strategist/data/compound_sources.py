"""Compile Pirelli compound nominations (C0–C6) per race from Wikipedia race articles.

Usage (from the repository root):

    python -m strategist.data.compound_sources --config configs/data.yaml

For every ingested race, fetches the English Wikipedia article "<year> <event name>" (plain
text via the MediaWiki API, cached under ``<fastf1_cache>/wikipedia``) and extracts the
sentence in its "Tyre choices" section, e.g. "Pirelli brought the C1, C2, and C3 tyre
compounds (designated hard, medium, and soft, respectively)". Writes a review file
``<processed>/compound_candidates.csv`` with the parsed compounds and the source sentence;
the curated table is ``configs/compounds.csv`` (see ``strategist.data.compounds``).
"""

import argparse
import json
import logging
import re
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

from strategist.config import load_config

log = logging.getLogger(__name__)

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "f1-neural-strategist/0.1 (research project; compound table)"}
COMPOUND = re.compile(r"\bC([0-6])\b")


def fetch_extract(title: str, cache_dir: Path) -> tuple[str, str]:
    path = cache_dir / f"{re.sub(r'[^A-Za-z0-9_-]', '_', title)}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        params = {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                  "format": "json", "titles": title}
        for attempt in range(6):  # full-page extracts are one page per request: be gentle
            resp = requests.get(API, params=params, headers=HEADERS, timeout=30)
            if resp.status_code != 429:
                break
            wait = int(resp.headers.get("Retry-After", 0)) or 20 * (attempt + 1)
            log.warning("Wikipedia rate limit; waiting %d s", wait)
            time.sleep(wait)
        resp.raise_for_status()
        data = resp.json()
        path.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(3.0)
    page = next(iter(data["query"]["pages"].values()))
    return page.get("title", title), page.get("extract", "")


def tyre_sentence(text: str) -> str | None:
    """The sentence naming the nominated compounds (prefers the 'Tyre choices' section)."""
    section = text.split("=== Tyre choices ===", 1)
    scope = section[1] if len(section) > 1 else text
    for sentence in re.split(r"(?<=[.!?])\s+", scope):
        if len(set(COMPOUND.findall(sentence))) >= 3 and "Pirelli" in sentence + scope[:200]:
            return " ".join(sentence.split())
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    cache = Path(cfg["paths"]["fastf1_cache"]) / "wikipedia"
    cache.mkdir(parents=True, exist_ok=True)
    races = pd.read_parquet(Path(cfg["paths"]["processed"]) / "races.parquet")

    rows = []
    for r in races.sort_values(["year", "round"]).itertuples():
        title = f"{r.year} {r.event_name}"
        page_title, text = fetch_extract(title, cache)
        sentence = tyre_sentence(text)
        found = sorted({int(c) for c in COMPOUND.findall(sentence or "")})
        rows.append({
            "race_id": r.race_id, "year": r.year, "round": r.round, "event_name": r.event_name,
            "wiki_title": page_title, "found": " ".join(f"C{c}" for c in found),
            "hard": f"C{found[0]}" if len(found) == 3 else "",
            "medium": f"C{found[1]}" if len(found) == 3 else "",
            "soft": f"C{found[2]}" if len(found) == 3 else "",
            "source": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(page_title.replace(" ", "_")),
            "sentence": sentence or "",
        })
        log.info("%s %-28s %s", r.race_id, r.event_name, rows[-1]["found"] or "NOT FOUND")
    out = pd.DataFrame(rows)
    path = Path(cfg["paths"]["processed"]) / "compound_candidates.csv"
    out.to_csv(path, index=False)
    log.info("parsed %d/%d races; wrote %s", (out["hard"] != "").sum(), len(out), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
