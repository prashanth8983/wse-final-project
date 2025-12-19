import csv
import requests
from pathlib import Path
from typing import List, Dict, Any, Iterator, Tuple
from dataclasses import dataclass
import logging

from .expander import Result
from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    doc_id: str
    passage_id: str
    score: float
    text: str


class Bridge:

    def __init__(self, data_dir: Path = None, output_dir: Path = None, api_url: str = None):
        self.data_dir = Path(data_dir or config.data_dir)
        self.output_dir = Path(output_dir or config.output_dir)
        self.api_url = api_url or config.indexer_api_url
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def read(self, filename: str = None, limit: int = None) -> Iterator[Tuple[str, str]]:
        path = self.data_dir / (filename or config.input_tsv)
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 2:
                    yield row[0], row[1]
                    count += 1
                    if limit and count >= limit:
                        break

    def write(self, results: Iterator[Result], filename: str = None) -> Tuple[int, Path]:
        path = self.output_dir / (filename or config.output_tsv)
        count = 0
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            for r in results:
                w.writerow([r.doc_id, r.expanded])
                count += 1
        return count, path

    def write_comparison(self, results: List[Result], prefix: str = "cmp") -> Dict[str, Path]:
        paths = {}
        for name, get_text in [("original", lambda r: r.original), ("d2q", lambda r: f"{r.original} {' '.join(r.queries)}"), ("hqfde", lambda r: r.expanded)]:
            p = self.output_dir / f"{prefix}_{name}.tsv"
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                for r in results:
                    w.writerow([r.doc_id, get_text(r)])
            paths[name] = p
        return paths

    def search(self, query: str, mode: str = "or", limit: int = 10) -> List[SearchResult]:
        try:
            resp = requests.get(f"{self.api_url}/search", params={"q": query, "mode": mode, "limit": limit}, timeout=30)
            resp.raise_for_status()
            return [SearchResult(doc_id=str(r.get("doc_id", "")), passage_id=str(r.get("passage_id", "")), score=float(r.get("score", 0)), text=r.get("text", "")) for r in resp.json().get("results", [])]
        except:
            return []

    def health(self) -> bool:
        try:
            return requests.get(f"{self.api_url}/health", timeout=5).status_code == 200
        except:
            return False
