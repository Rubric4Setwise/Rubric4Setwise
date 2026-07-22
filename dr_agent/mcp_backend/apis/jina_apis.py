import os
from typing import Dict, Optional

import dotenv
import requests
from typing_extensions import TypedDict

from ..cache import cached

dotenv.load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
TIMEOUT = int(os.getenv("API_TIMEOUT", 30))

# Internal proxy configuration
USE_INTERNAL_PROXY = os.getenv("USE_INTERNAL_PROXY", "true").lower() == "true"
INTERNAL_PROXY_HOST = os.getenv("INTERNAL_PROXY_HOST", "trpc-gpt-eval.production.polaris")
INTERNAL_PROXY_PORT = int(os.getenv("INTERNAL_PROXY_PORT", "8080"))
INTERNAL_PROXY_BASE_URL = f"http://{INTERNAL_PROXY_HOST}:{INTERNAL_PROXY_PORT}"
INTERNAL_PROXY_APP_ID = os.getenv("INTERNAL_API_APP_ID", "")
INTERNAL_PROXY_APP_KEY = os.getenv("INTERNAL_API_APP_KEY", "")
INTERNAL_PROXY_TIMEOUT = int(os.getenv("INTERNAL_PROXY_TIMEOUT", "60"))


class JinaMetadata(TypedDict, total=False):
    lang: str
    viewport: str


class JinaWebpageResponse(TypedDict, total=False):
    url: str
    title: str
    content: str
    description: str
    publishedTime: str
    metadata: JinaMetadata
    success: bool
    error: str


def _jina_via_internal_proxy(url: str, timeout: int = None) -> JinaWebpageResponse:
    """Fetch webpage content via internal Jina proxy."""
    proxy_url = f"{INTERNAL_PROXY_BASE_URL}/"
    _timeout = timeout or INTERNAL_PROXY_TIMEOUT
    auth = f"Bearer {INTERNAL_PROXY_APP_ID}:{INTERNAL_PROXY_APP_KEY}?provider=aws_jina&timeout={_timeout}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": auth,
    }

    response = requests.post(
        proxy_url, headers=headers, json={"url": url}, timeout=_timeout + 10
    )

    if response.status_code != 200:
        raise Exception(
            f"Internal proxy Jina request failed with status {response.status_code}: {response.text}"
        )

    json_response = response.json()
    data = json_response.get("data", {})

    return {
        "url": data.get("url", url),
        "title": data.get("title", ""),
        "content": data.get("content", ""),
        "description": data.get("description", ""),
        "publishedTime": data.get("publishedTime", ""),
        "metadata": data.get("metadata", {}),
        "success": True,
    }


@cached()
def fetch_webpage_content_jina(
    url: str,
    api_key: str = None,
    timeout: int = TIMEOUT,
) -> JinaWebpageResponse:
    """
    Fetch webpage content using Jina Reader API with JSON format.

    Args:
        url: The URL of the webpage to fetch
        api_key: Jina API key (if not provided, will use JINA_API_KEY env var)
        timeout: Request timeout in seconds (if not provided, will use TIMEOUT env var or default 30)

    Returns:
        JinaWebpageResponse containing:
        - url: The original URL that was fetched
        - title: The webpage title
        - content: The webpage content as clean text/markdown
        - description: The webpage description (if available)
        - publishedTime: Publication timestamp (if available)
        - metadata: Additional metadata (lang, viewport, etc.)
        - success: Boolean indicating if the fetch was successful
        - error: Error message if fetch failed
    """
    if timeout is None:
        timeout = TIMEOUT

    # Use internal proxy if configured
    if USE_INTERNAL_PROXY and INTERNAL_PROXY_APP_ID and INTERNAL_PROXY_APP_KEY:
        return _jina_via_internal_proxy(url, timeout=timeout)

    # Fallback: direct Jina API
    if not api_key:
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError(
                "JINA_API_KEY environment variable is not set or api_key parameter not provided"
            )

    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    response = requests.get(jina_url, headers=headers, timeout=timeout)

    if response.status_code != 200:
        raise Exception(
            f"API request failed with status {response.status_code}: {response.text}"
        )

    json_response = response.json()

    # Extract data from JSON response
    data = json_response.get("data", {})

    return {
        "url": data.get("url", url),
        "title": data.get("title", ""),
        "content": data.get("content", ""),
        "description": data.get("description", ""),
        "publishedTime": data.get("publishedTime", ""),
        "metadata": data.get("metadata", {}),
        "success": True,
    }
