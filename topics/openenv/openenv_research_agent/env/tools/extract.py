"""
Tavily content extraction tool — action: tavily_extract

Wraps the Tavily extract API as a discrete action in the research
RL environment. Registered on the MCPEnvironment via @mcp.tool in
research_env.py.

Use this action when the agent already has specific URLs from a
tavily_search step and needs full page content rather than snippets.
"""

import time
from tavily import TavilyClient


def _is_rate_limit(error: Exception) -> bool:
    msg = str(error).lower()
    return "usage limit" in msg or "rate limit" in msg or "429" in msg


def run_extract(
    client: TavilyClient,
    urls: list[str],
    extract_depth: str = "basic",
) -> dict:
    """Extract structured full-page content from specific URLs.

    Use this tool when you already have URLs (e.g. from tavily_search) and
    need the full readable content of those pages rather than just snippets.
    More precise than crawling when you have a known list of pages to read.

    Args:
        urls: List of URLs to extract content from (1-20 URLs).
        extract_depth: "basic" for main content, "advanced" for deeper
                       extraction including tables and structured data.
    """
    for attempt in range(3):
        try:
            response = client.extract(
                urls=urls,
                extract_depth=extract_depth,
            )
            return {
                "results": [
                    {
                        "url": r.get("url", ""),
                        "raw_content": r.get("raw_content", ""),
                    }
                    for r in response.get("results", [])
                ],
                "failed_results": response.get("failed_results", []),
            }
        except Exception as e:
            if _is_rate_limit(e) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return {"error": str(e), "results": []}
