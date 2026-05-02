import io
import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import ALLOWED_MIME, MAX_FILE_SIZE
from app.utils.errors import AppError


_HTTP_TIMEOUT_SECONDS = 8.0


async def load_from_upload(file: UploadFile) -> Image.Image:
    """Validate and decode an uploaded file into a PIL RGB image."""
    if file.content_type not in ALLOWED_MIME:
        raise AppError(415, f"Unsupported media type: {file.content_type}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise AppError(413, f"File too large: max {MAX_FILE_SIZE // (1024 * 1024)} MB")
    if len(content) == 0:
        raise AppError(400, "Empty file")

    return _decode_image(content)


async def load_from_url(url: str) -> Image.Image:
    """Fetch an image by URL with SSRF protection. Returns PIL RGB image."""
    _assert_url_is_safe(url)

    try:
        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT_SECONDS,
            follow_redirects=False,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AppError(400, f"Failed to fetch URL: {exc}") from exc

    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_MIME:
        raise AppError(415, f"Unsupported media type: {content_type or 'unknown'}")

    if len(response.content) > MAX_FILE_SIZE:
        raise AppError(413, f"File too large: max {MAX_FILE_SIZE // (1024 * 1024)} MB")

    return _decode_image(response.content)


def _decode_image(content: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise AppError(400, f"Invalid image: {exc}") from exc

    return image.convert("RGB")


def _assert_url_is_safe(url: str) -> None:
    """Reject non-http(s) schemes and any host that resolves to a private IP."""
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise AppError(400, f"Unsupported URL scheme: {parsed.scheme or 'none'}")

    if not parsed.hostname:
        raise AppError(400, "URL has no hostname")

    try:
        addr_infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise AppError(400, f"Cannot resolve hostname: {parsed.hostname}") from exc

    for info in addr_infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise AppError(400, f"Invalid resolved address: {ip_str}") from exc

        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise AppError(400, "URL points to a private or local address")
