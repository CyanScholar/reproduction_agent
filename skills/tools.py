"""
Skills Tools - Unified tool layer for Skills
Merges tools/base.py, tools/arxiv_search.py, tools/semantic_scholar.py,
tools/github_search.py, tools/papers_with_code.py into a single module.

Provides: ToolResult, Tool, ToolRegistry, ArxivSearchTool, SemanticScholarTool,
GitHubSearchTool, PapersWithCodeTool, and create_default_registry().
"""

import os
import time
import logging
import tempfile
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------
ARXIV_API_URL = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1"
GITHUB_API_BASE = "https://api.github.com"
PAPERS_WITH_CODE_API_BASE = "https://paperswithcode.com/api/v1"


# ===========================================================================
# Base classes
# ===========================================================================

@dataclass
class ToolResult:
    """Tool invocation result."""
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None

    def to_context_string(self) -> str:
        """Format as a string suitable for LLM context injection."""
        if not self.success:
            return f"[{self.tool_name}] Failed: {self.error}"

        if isinstance(self.data, list):
            items = []
            for item in self.data[:10]:  # show at most 10 items
                if isinstance(item, dict):
                    parts = []
                    for k, v in item.items():
                        if v is not None:
                            parts.append(f"  {k}: {v}")
                    items.append("\n".join(parts))
                else:
                    items.append(str(item))
            return f"[{self.tool_name}] Found {len(self.data)} results:\n\n" + "\n---\n".join(items)

        if isinstance(self.data, dict):
            parts = [f"[{self.tool_name}] Result:"]
            for k, v in self.data.items():
                if v is not None:
                    parts.append(f"  {k}: {v}")
            return "\n".join(parts)

        return f"[{self.tool_name}] {self.data}"


class Tool(ABC):
    """Abstract base class for all tools."""
    name: str = "base_tool"
    description: str = ""

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass

    @abstractmethod
    def to_schema(self) -> dict:
        """Return API-compatible function schema for tool_use / function_calling.

        The returned dict uses the Anthropic ``input_schema`` key.  Callers
        targeting OpenAI can trivially rename the key to ``parameters``.
        """
        ...

    def _request_with_retry(
        self,
        func,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> Any:
        """HTTP request with exponential-backoff retry."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"[{self.name}] Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        raise last_error


class ToolRegistry:
    """Registry that holds and dispatches tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a registered tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(tool_name=name, success=False, error=f"Tool not found: {name}")
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"[{name}] Execution failed: {e}")
            return ToolResult(tool_name=name, success=False, error=str(e))

    def get_all_schemas(self) -> List[dict]:
        """Return schemas for every registered tool."""
        return [tool.to_schema() for tool in self._tools.values()]


# ===========================================================================
# ArxivSearchTool
# ===========================================================================

class ArxivSearchTool(Tool):
    """
    arXiv paper search and download tool.

    Actions:
        - search: search papers via the Atom API
        - download: download a PDF by arXiv ID
    """
    name = "arxiv_search"
    description = "Search and download papers from arXiv"

    def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "search")
        query = kwargs.get("query")

        if action == "search":
            if not query:
                return ToolResult(tool_name=self.name, success=False, error="Missing 'query' parameter")
            return self._search(query, limit=kwargs.get("limit", 5))
        elif action == "download":
            arxiv_id = kwargs.get("arxiv_id") or query
            if not arxiv_id:
                return ToolResult(tool_name=self.name, success=False, error="Missing 'arxiv_id' parameter")
            return self._download(arxiv_id, output_dir=kwargs.get("output_dir"))
        else:
            return ToolResult(tool_name=self.name, success=False, error=f"Unknown action: {action}")

    def to_schema(self) -> dict:
        return {
            "name": "arxiv_search",
            "description": "Search and download papers from arXiv",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or arXiv ID",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["search", "download"],
                        "default": "search",
                        "description": "Action to perform: search papers or download a PDF",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of search results to return",
                    },
                },
                "required": ["query"],
            },
        }

    # -- private helpers -----------------------------------------------------

    def _search(self, query: str, limit: int = 5) -> ToolResult:
        try:
            import requests
        except ImportError:
            return ToolResult(tool_name=self.name, success=False, error="requests library not installed")

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            resp = self._request_with_retry(
                lambda: requests.get(ARXIV_API_URL, params=params, timeout=30)
            )
            resp.raise_for_status()
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"arXiv API error: {e}")

        results = self._parse_atom_feed(resp.text)
        return ToolResult(tool_name=self.name, success=True, data=results)

    def _download(self, arxiv_id: str, output_dir: Optional[str] = None) -> ToolResult:
        try:
            import requests
        except ImportError:
            return ToolResult(tool_name=self.name, success=False, error="requests library not installed")

        # Normalise arXiv ID
        arxiv_id = arxiv_id.replace("https://arxiv.org/abs/", "").replace("https://arxiv.org/pdf/", "")
        arxiv_id = arxiv_id.rstrip("/").rstrip(".pdf")

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        try:
            resp = self._request_with_retry(
                lambda: requests.get(pdf_url, timeout=60, stream=True)
            )
            resp.raise_for_status()
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"Download failed: {e}")

        # Save the file
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            file_path = out_path / f"{arxiv_id.replace('/', '_')}.pdf"
            file_path.write_bytes(resp.content)
        else:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".pdf", prefix=f"arxiv_{arxiv_id.replace('/', '_')}_", delete=False
            )
            tmp.write(resp.content)
            tmp.close()
            file_path = Path(tmp.name)

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "arxiv_id": arxiv_id,
                "file_path": str(file_path),
                "size_bytes": file_path.stat().st_size,
                "pdf_url": pdf_url,
            },
        )

    def _parse_atom_feed(self, xml_text: str) -> List[Dict]:
        """Parse an arXiv Atom feed into a list of dicts."""
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        results = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return results

        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)

            # Extract arXiv ID
            id_el = entry.find("atom:id", ns)
            arxiv_id = ""
            if id_el is not None and id_el.text:
                arxiv_id = id_el.text.split("/abs/")[-1] if "/abs/" in id_el.text else id_el.text

            # Extract authors
            authors = []
            for author_el in entry.findall("atom:author", ns):
                name_el = author_el.find("atom:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text)

            # Extract PDF link
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            results.append({
                "arxiv_id": arxiv_id,
                "title": title_el.text.strip() if title_el is not None and title_el.text else "",
                "authors": authors[:5],  # at most 5 authors
                "abstract": summary_el.text.strip()[:500] if summary_el is not None and summary_el.text else "",
                "published": published_el.text[:10] if published_el is not None and published_el.text else "",
                "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            })

        return results


# ===========================================================================
# SemanticScholarTool
# ===========================================================================

class SemanticScholarTool(Tool):
    """Search academic papers, citations, and related work via Semantic Scholar API."""
    name = "semantic_scholar"
    description = "Search academic papers, citations, and related work via Semantic Scholar API"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    def execute(
        self,
        query: Optional[str] = None,
        paper_id: Optional[str] = None,
        action: str = "search",
        limit: int = 5,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a Semantic Scholar action.

        Args:
            query: Search keywords.
            paper_id: Paper ID (for citations/references/details).
            action: "search" | "citations" | "references" | "details"
            limit: Number of results to return.
        """
        try:
            import requests
        except ImportError:
            return ToolResult(tool_name=self.name, success=False, error="requests library not installed")

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            if action == "search" and query:
                return self._search(requests, headers, query, limit)
            elif action == "citations" and paper_id:
                return self._citations(requests, headers, paper_id, limit)
            elif action == "references" and paper_id:
                return self._references(requests, headers, paper_id, limit)
            elif action == "details" and paper_id:
                return self._details(requests, headers, paper_id)
            else:
                return ToolResult(
                    tool_name=self.name, success=False,
                    error=f"Invalid action '{action}' or missing required parameter",
                )
        except Exception as e:
            logger.warning(f"[{self.name}] API call failed: {e}")
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    def to_schema(self) -> dict:
        return {
            "name": "semantic_scholar",
            "description": "Search academic papers, citations, and related work via Semantic Scholar API",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords for finding papers",
                    },
                    "paper_id": {
                        "type": "string",
                        "description": "Semantic Scholar paper ID (for citations, references, or details)",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["search", "citations", "references", "details"],
                        "default": "search",
                        "description": "Action to perform",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of results to return",
                    },
                },
                "required": ["query"],
            },
        }

    # -- private helpers -----------------------------------------------------

    def _search(self, requests, headers, query: str, limit: int) -> ToolResult:
        fields = "title,authors,year,citationCount,url,externalIds,abstract"
        resp = self._request_with_retry(
            lambda: requests.get(
                f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search",
                params={"query": query, "limit": limit, "fields": fields},
                headers=headers,
                timeout=15,
            )
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        results = []
        for paper in data:
            authors = [a.get("name", "") for a in paper.get("authors", [])[:3]]
            results.append({
                "title": paper.get("title"),
                "authors": ", ".join(authors),
                "year": paper.get("year"),
                "citations": paper.get("citationCount"),
                "url": paper.get("url"),
                "paper_id": paper.get("paperId"),
            })
        return ToolResult(tool_name=self.name, success=True, data=results)

    def _citations(self, requests, headers, paper_id: str, limit: int) -> ToolResult:
        fields = "title,authors,year,citationCount"
        resp = self._request_with_retry(
            lambda: requests.get(
                f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}/citations",
                params={"limit": limit, "fields": fields},
                headers=headers,
                timeout=15,
            )
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        results = []
        for item in data:
            citing = item.get("citingPaper", {})
            results.append({
                "title": citing.get("title"),
                "year": citing.get("year"),
                "citations": citing.get("citationCount"),
            })
        return ToolResult(tool_name=self.name, success=True, data=results)

    def _references(self, requests, headers, paper_id: str, limit: int) -> ToolResult:
        fields = "title,authors,year,citationCount"
        resp = self._request_with_retry(
            lambda: requests.get(
                f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}/references",
                params={"limit": limit, "fields": fields},
                headers=headers,
                timeout=15,
            )
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        results = []
        for item in data:
            cited = item.get("citedPaper", {})
            results.append({
                "title": cited.get("title"),
                "year": cited.get("year"),
                "citations": cited.get("citationCount"),
            })
        return ToolResult(tool_name=self.name, success=True, data=results)

    def _details(self, requests, headers, paper_id: str) -> ToolResult:
        fields = "title,authors,year,abstract,citationCount,referenceCount,url,externalIds,fieldsOfStudy,venue"
        resp = self._request_with_retry(
            lambda: requests.get(
                f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}",
                params={"fields": fields},
                headers=headers,
                timeout=15,
            )
        )
        resp.raise_for_status()
        paper = resp.json()
        authors = [a.get("name", "") for a in paper.get("authors", [])[:5]]
        return ToolResult(tool_name=self.name, success=True, data={
            "title": paper.get("title"),
            "authors": ", ".join(authors),
            "year": paper.get("year"),
            "abstract": paper.get("abstract", "")[:500],
            "citations": paper.get("citationCount"),
            "references": paper.get("referenceCount"),
            "venue": paper.get("venue"),
            "fields": paper.get("fieldsOfStudy"),
            "url": paper.get("url"),
        })


# ===========================================================================
# GitHubSearchTool
# ===========================================================================

class GitHubSearchTool(Tool):
    """Search GitHub repositories for paper implementations."""
    name = "github_search"
    description = "Search GitHub repositories for paper implementations"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GITHUB_TOKEN")

    def execute(
        self,
        query: Optional[str] = None,
        language: str = "python",
        action: str = "search_repos",
        limit: int = 5,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a GitHub search.

        Args:
            query: Search keywords (paper title or method name).
            language: Programming language filter.
            action: "search_repos" | "search_code"
            limit: Number of results to return.
        """
        try:
            import requests
        except ImportError:
            return ToolResult(tool_name=self.name, success=False, error="requests library not installed")

        if not query:
            return ToolResult(tool_name=self.name, success=False, error="query is required")

        headers = {"Accept": "application/vnd.github+json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            if action == "search_repos":
                return self._search_repos(requests, headers, query, language, limit)
            elif action == "search_code":
                return self._search_code(requests, headers, query, language, limit)
            else:
                return ToolResult(
                    tool_name=self.name, success=False,
                    error=f"Invalid action: {action}",
                )
        except Exception as e:
            logger.warning(f"[{self.name}] API call failed: {e}")
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    def to_schema(self) -> dict:
        return {
            "name": "github_search",
            "description": "Search GitHub repositories and code for paper implementations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (paper title or method name)",
                    },
                    "language": {
                        "type": "string",
                        "default": "python",
                        "description": "Programming language filter",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["search_repos", "search_code"],
                        "default": "search_repos",
                        "description": "Action to perform: search repositories or search code",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of results to return",
                    },
                },
                "required": ["query"],
            },
        }

    # -- private helpers -----------------------------------------------------

    def _search_repos(self, requests, headers, query: str, language: str, limit: int) -> ToolResult:
        search_query = f"{query} language:{language}" if language else query
        resp = self._request_with_retry(
            lambda: requests.get(
                f"{GITHUB_API_BASE}/search/repositories",
                params={"q": search_query, "sort": "stars", "per_page": limit},
                headers=headers,
                timeout=15,
            )
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        results = []
        for repo in items:
            results.append({
                "name": repo.get("full_name"),
                "description": (repo.get("description") or "")[:200],
                "stars": repo.get("stargazers_count"),
                "language": repo.get("language"),
                "url": repo.get("html_url"),
                "updated": repo.get("updated_at", "")[:10],
            })
        return ToolResult(tool_name=self.name, success=True, data=results)

    def _search_code(self, requests, headers, query: str, language: str, limit: int) -> ToolResult:
        search_query = f"{query} language:{language}" if language else query
        resp = self._request_with_retry(
            lambda: requests.get(
                f"{GITHUB_API_BASE}/search/code",
                params={"q": search_query, "per_page": limit},
                headers=headers,
                timeout=15,
            )
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        results = []
        for item in items:
            repo = item.get("repository", {})
            results.append({
                "file": item.get("name"),
                "path": item.get("path"),
                "repo": repo.get("full_name"),
                "url": item.get("html_url"),
            })
        return ToolResult(tool_name=self.name, success=True, data=results)


# ===========================================================================
# PapersWithCodeTool
# ===========================================================================

class PapersWithCodeTool(Tool):
    """Search Papers With Code for implementations and benchmarks."""
    name = "papers_with_code"
    description = "Search Papers With Code for implementations and benchmarks"

    def execute(
        self,
        query: Optional[str] = None,
        paper_url: Optional[str] = None,
        action: str = "search",
        limit: int = 5,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a Papers With Code search.

        Args:
            query: Search keywords.
            paper_url: Paper URL (arXiv link).
            action: "search" | "implementations" | "benchmarks"
            limit: Number of results to return.
        """
        try:
            import requests
        except ImportError:
            return ToolResult(tool_name=self.name, success=False, error="requests library not installed")

        headers = {"Accept": "application/json"}

        try:
            if action == "search" and query:
                return self._search(requests, headers, query, limit)
            elif action == "implementations" and query:
                return self._implementations(requests, headers, query, limit)
            elif action == "benchmarks" and query:
                return self._benchmarks(requests, headers, query, limit)
            else:
                return ToolResult(
                    tool_name=self.name, success=False,
                    error=f"Invalid action '{action}' or missing required parameter",
                )
        except Exception as e:
            logger.warning(f"[{self.name}] API call failed: {e}")
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    def to_schema(self) -> dict:
        return {
            "name": "papers_with_code",
            "description": "Search Papers With Code for implementations and benchmarks",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (paper title, method, or task name)",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["search", "implementations", "benchmarks"],
                        "default": "search",
                        "description": "Action to perform: search papers, find implementations, or list benchmarks",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of results to return",
                    },
                },
                "required": ["query"],
            },
        }

    # -- private helpers -----------------------------------------------------

    def _search(self, requests, headers, query: str, limit: int) -> ToolResult:
        resp = self._request_with_retry(
            lambda: requests.get(
                f"{PAPERS_WITH_CODE_API_BASE}/papers/",
                params={"q": query, "items_per_page": limit},
                headers=headers,
                timeout=15,
            )
        )
        resp.raise_for_status()
        data = resp.json()
        results_list = data.get("results", [])
        results = []
        for paper in results_list:
            results.append({
                "title": paper.get("title"),
                "abstract": (paper.get("abstract") or "")[:300],
                "url_abs": paper.get("url_abs"),
                "url_pdf": paper.get("url_pdf"),
                "proceeding": paper.get("proceeding"),
            })
        return ToolResult(tool_name=self.name, success=True, data=results)

    def _implementations(self, requests, headers, query: str, limit: int) -> ToolResult:
        # Search for the paper first
        search_result = self._search(requests, headers, query, 1)
        if not search_result.success or not search_result.data:
            return ToolResult(tool_name=self.name, success=True, data=[])

        paper_url = search_result.data[0].get("url_abs", "")
        if not paper_url:
            return ToolResult(tool_name=self.name, success=True, data=[])

        # Extract paper ID and look up implementations
        resp = self._request_with_retry(
            lambda: requests.get(
                f"{PAPERS_WITH_CODE_API_BASE}/papers/",
                params={"q": query, "items_per_page": 1},
                headers=headers,
                timeout=15,
            )
        )
        resp.raise_for_status()
        papers = resp.json().get("results", [])
        if not papers:
            return ToolResult(tool_name=self.name, success=True, data=[])

        paper_id = papers[0].get("id")
        if not paper_id:
            return ToolResult(tool_name=self.name, success=True, data=[])

        # Fetch repositories for this paper
        repo_resp = self._request_with_retry(
            lambda: requests.get(
                f"{PAPERS_WITH_CODE_API_BASE}/papers/{paper_id}/repositories/",
                headers=headers,
                timeout=15,
            )
        )
        repo_resp.raise_for_status()
        repos = repo_resp.json().get("results", [])[:limit]
        results = []
        for repo in repos:
            results.append({
                "url": repo.get("url"),
                "framework": repo.get("framework"),
                "stars": repo.get("stars"),
                "is_official": repo.get("is_official", False),
            })
        return ToolResult(tool_name=self.name, success=True, data=results)

    def _benchmarks(self, requests, headers, query: str, limit: int) -> ToolResult:
        # Search for benchmark tasks related to the query
        resp = self._request_with_retry(
            lambda: requests.get(
                f"{PAPERS_WITH_CODE_API_BASE}/tasks/",
                params={"q": query, "items_per_page": limit},
                headers=headers,
                timeout=15,
            )
        )
        resp.raise_for_status()
        data = resp.json()
        tasks = data.get("results", [])
        results = []
        for task in tasks:
            results.append({
                "name": task.get("name"),
                "description": (task.get("description") or "")[:200],
                "datasets_count": task.get("datasets_count", 0),
                "papers_count": task.get("papers_count", 0),
            })
        return ToolResult(tool_name=self.name, success=True, data=results)


# ===========================================================================
# Factory
# ===========================================================================

def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-loaded with all built-in tools."""
    registry = ToolRegistry()
    registry.register(ArxivSearchTool())
    registry.register(SemanticScholarTool())
    registry.register(GitHubSearchTool())
    registry.register(PapersWithCodeTool())
    return registry
