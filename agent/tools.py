"""
Real-time tool definitions for LangGraph's tool node.
These use free APIs — no key needed for Wikipedia summary.
"""

import requests


def wikipedia_summary(topic: str) -> str:
    """Fetch a short Wikipedia summary for a topic."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        return data.get("extract", "No summary found.")
    except Exception as e:
        return f"Error fetching Wikipedia: {e}"


def arxiv_search(query: str) -> str:
    """Search arXiv for recent papers on a topic."""
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "max_results": 3, "sortBy": "submittedDate"}
    try:
        r = requests.get(url, params=params, timeout=8)
        import re
        titles = re.findall(r"<title>(.*?)</title>", r.text)[1:]  # skip feed title
        summaries = re.findall(r"<summary>(.*?)</summary>", r.text, re.DOTALL)
        results = []
        for t, s in zip(titles[:3], summaries[:3]):
            results.append(f"Title: {t.strip()}\nAbstract: {s.strip()[:300]}")
        return "\n\n".join(results) if results else "No papers found."
    except Exception as e:
        return f"Error fetching arXiv: {e}"


TOOLS = {
    "wikipedia_summary": wikipedia_summary,
    "arxiv_search": arxiv_search,
}
