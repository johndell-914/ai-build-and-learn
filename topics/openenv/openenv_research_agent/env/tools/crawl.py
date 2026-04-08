"""
Tavily web crawl tool — action: tavily_crawl

Wraps the Tavily crawl API as a discrete action in the research
RL environment. Registered on the MCPEnvironment via @mcp.tool in
research_env.py.

Use this action when the agent needs to explore an entire site or
documentation section. More expensive than extract — prefer extract
when specific URLs are already known.
"""

import time
from typing import Optional
from tavily import TavilyClient


def _is_rate_limit(error: Exception) -> bool:
    msg = str(error).lower()
    return "usage limit" in msg or "rate limit" in msg or "429" in msg


def run_crawl(
    client: TavilyClient,
    url: str,
    max_depth: int = 1,
    max_breadth: int = 10,
    limit: int = 10,
    instructions: Optional[str] = None,
) -> dict:
    """Crawl a website starting from a root URL to gather site-wide content.

    Use this tool when you need comprehensive information from an entire site
    or documentation section, not just a single page. Follows internal links
    up to max_depth levels deep. More expensive than extract — prefer extract
    when you have specific URLs.

    Args:
        url: The root URL to begin crawling from.
        max_depth: How many link-hops deep to follow (default: 1).
        max_breadth: Max number of links to follow per page (default: 10).
        limit: Total maximum pages to return (default: 10).
        instructions: Optional natural-language guidance to focus the crawl.
    """
    for attempt in range(3):
        try:
            kwargs = dict(
                url=url,
                max_depth=max_depth,
                max_breadth=max_breadth,
                limit=limit,
            )
            if instructions:
                kwargs["instructions"] = instructions

            response = client.crawl(**kwargs)
            return {
                "root_url": url,
                "results": [
                    {
                        "url": r.get("url", ""),
                        "raw_content": r.get("raw_content", ""),
                    }
                    for r in response.get("results", [])
                ],
            }
        except Exception as e:
            if _is_rate_limit(e) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return {"root_url": url, "error": str(e), "results": []}
