"""Wikipedia/Wikidata image sourcing service.

Fetches landmark photos from Wikimedia Commons using the Wikipedia and
Wikidata APIs, with a fallback search chain.
"""

import hashlib
import re
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image
from io import BytesIO


# Wikimedia API policy requires a descriptive User-Agent
USER_AGENT = "MapGen/1.0 (illustrated map generator)"


class WikipediaImageService:
    """Service for fetching landmark images from Wikipedia/Wikidata."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the service.

        Args:
            cache_dir: Directory for caching downloaded images.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_image_for_landmark(
        self,
        name: str,
        wikipedia_url: Optional[str] = None,
        wikidata_id: Optional[str] = None,
        target_size: int = 800,
    ) -> Optional[Image.Image]:
        """Fetch an image for a landmark using a fallback chain.

        Tries sources in order:
        1. Wikipedia page image (if URL provided)
        2. Wikidata P18 image (if QID provided)
        3. Wikipedia search by name

        Args:
            name: Landmark name (used for search fallback).
            wikipedia_url: Wikipedia article URL.
            wikidata_id: Wikidata entity ID (e.g., "Q12345").
            target_size: Desired image width in pixels.

        Returns:
            PIL Image or None if no image found.
        """
        # Check disk cache first
        cached = self._load_from_cache(name)
        if cached is not None:
            return cached

        image = None

        # 1. Try Wikipedia URL
        if wikipedia_url and image is None:
            title = self._extract_title_from_url(wikipedia_url)
            lang = self._extract_lang_from_url(wikipedia_url)
            if title:
                image = self._fetch_from_wikipedia(title, target_size, lang=lang)

        # 2. Try Wikidata
        if wikidata_id and image is None:
            image = self._fetch_from_wikidata(wikidata_id, target_size)

        # 3. Fallback: search Wikipedia by name
        if image is None:
            image = self._fetch_from_wikipedia_search(name, target_size)

        # Cache on success
        if image is not None:
            self._save_to_cache(name, image)

        return image

    def _fetch_from_wikipedia(
        self, title: str, target_size: int, lang: str = "en"
    ) -> Optional[Image.Image]:
        """Fetch the main image from a Wikipedia article.

        Args:
            title: Wikipedia article title.
            target_size: Desired width in pixels.
            lang: Wikipedia language code.

        Returns:
            PIL Image or None.
        """
        try:
            url = f"https://{lang}.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "titles": title,
                "prop": "pageimages",
                "format": "json",
                "pithumbsize": target_size,
            }
            with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    thumb_url = page.get("thumbnail", {}).get("source")
                    if thumb_url:
                        return self._download_image(client, thumb_url)

        except Exception:
            pass
        return None

    def _fetch_from_wikidata(
        self, qid: str, target_size: int
    ) -> Optional[Image.Image]:
        """Fetch the P18 (image) property from Wikidata.

        Args:
            qid: Wikidata entity ID (e.g., "Q12345").
            target_size: Desired width in pixels.

        Returns:
            PIL Image or None.
        """
        try:
            url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbgetclaims",
                "entity": qid,
                "property": "P18",
                "format": "json",
            }
            with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

                claims = data.get("claims", {}).get("P18", [])
                if claims:
                    filename = claims[0]["mainsnak"]["datavalue"]["value"]
                    return self._download_commons_image(client, filename, target_size)

        except Exception:
            pass
        return None

    def _fetch_from_wikipedia_search(
        self, name: str, target_size: int
    ) -> Optional[Image.Image]:
        """Search Wikipedia for the landmark and get its page image.

        Args:
            name: Landmark name to search for.
            target_size: Desired width in pixels.

        Returns:
            PIL Image or None.
        """
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": name,
                "srlimit": 1,
                "format": "json",
            }
            with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

                results = data.get("query", {}).get("search", [])
                if results:
                    title = results[0]["title"]
                    return self._fetch_from_wikipedia(title, target_size)

        except Exception:
            pass
        return None

    def _download_commons_image(
        self, client: httpx.Client, filename: str, target_size: int
    ) -> Optional[Image.Image]:
        """Download an image from Wikimedia Commons.

        Args:
            client: httpx Client instance.
            filename: Commons filename.
            target_size: Desired width.

        Returns:
            PIL Image or None.
        """
        try:
            # URL-encode the filename
            encoded = filename.replace(" ", "_")
            url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded}?width={target_size}"
            return self._download_image(client, url)
        except Exception:
            return None

    def _download_image(
        self, client: httpx.Client, url: str
    ) -> Optional[Image.Image]:
        """Download an image from a URL.

        Args:
            client: httpx Client instance.
            url: Image URL.

        Returns:
            PIL Image or None.
        """
        try:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            return None

    def _extract_title_from_url(self, url: str) -> Optional[str]:
        """Extract the article title from a Wikipedia URL."""
        match = re.search(r"wikipedia\.org/wiki/(.+?)(?:#.*)?$", url)
        if match:
            from urllib.parse import unquote
            return unquote(match.group(1)).replace("_", " ")
        return None

    def _extract_lang_from_url(self, url: str) -> str:
        """Extract language code from a Wikipedia URL."""
        match = re.search(r"https?://(\w+)\.wikipedia\.org", url)
        if match:
            return match.group(1)
        return "en"

    def _cache_key(self, name: str) -> str:
        """Generate a cache filename for a landmark name."""
        safe = hashlib.md5(name.encode()).hexdigest()[:12]
        return f"wiki_{safe}.jpg"

    def _load_from_cache(self, name: str) -> Optional[Image.Image]:
        """Load a cached image if it exists."""
        if not self.cache_dir:
            return None
        path = self.cache_dir / self._cache_key(name)
        if path.exists():
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                return None
        return None

    def _save_to_cache(self, name: str, image: Image.Image) -> None:
        """Save an image to the disk cache."""
        if not self.cache_dir:
            return
        path = self.cache_dir / self._cache_key(name)
        try:
            image.save(path, "JPEG", quality=90)
        except Exception:
            pass
