import csv
import requests
from pathlib import Path
from dataclasses import dataclass
from ..config import config


@dataclass
class SearchResult:
    doc_id: str
    passage_id: str
    score: float
    text: str


class Bridge:
    def __init__(self, data_dir=None, output_dir=None, api_url=None):
        self.data_dir = Path(data_dir or config.data_dir)
        self.output_dir = Path(output_dir or config.output_dir)
        self.api_url = api_url or getattr(config, 'indexer_api_url', 'http://localhost:8080')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def read(self, filename=None, limit=None):
        path = self.data_dir / (filename or getattr(config, 'input_tsv', 'collection.tsv'))
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

    def write(self, results, filename=None):
        path = self.output_dir / (filename or getattr(config, 'output_tsv', 'expanded.tsv'))
        count = 0
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            for r in results:
                w.writerow([r.get('doc_id', r.doc_id if hasattr(r, 'doc_id') else ''), r.get('expanded', r.expanded if hasattr(r, 'expanded') else '')])
                count += 1
        return count, path

    def search(self, query, mode="or", limit=10):
        try:
            resp = requests.get(f"{self.api_url}/search", params={"q": query, "mode": mode, "limit": limit}, timeout=30)
            resp.raise_for_status()
            return [SearchResult(doc_id=str(r.get("doc_id", "")), passage_id=str(r.get("passage_id", "")), score=float(r.get("score", 0)), text=r.get("text", "")) for r in resp.json().get("results", [])]
        except:
            return []

    def health(self):
        try:
            return requests.get(f"{self.api_url}/health", timeout=5).status_code == 200
        except:
            return False
