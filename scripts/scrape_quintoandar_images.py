#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests


DEFAULT_SEARCH_URL = "https://www.quintoandar.com.br/alugar/imovel/sao-paulo-sp-brasil/mobiliado"
DEFAULT_OUTPUT_DIR = Path("data/external/quintoandar")
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)
LISTING_URL_RE = re.compile(r"(https://www\.quintoandar\.com\.br/imovel/(\d+)[^\"<\\\s]*)")
VALID_IMAGE_SIZES = {"sml", "med", "xxl"}


ENVIRONMENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sala", ("sala", "living")),
    ("quarto", ("quarto", "suite", "dormitorio", "dormitório")),
    ("cozinha", ("cozinha", "kitchen")),
    ("banheiro", ("banheiro", "lavabo", "bathroom")),
    ("varanda", ("varanda", "sacada", "terraco", "terraço", "balcony")),
    ("area_servico", ("area de servico", "área de serviço", "lavanderia", "laundry")),
    ("escritorio", ("escritorio", "escritório", "home office", "home-office")),
    ("corredor", ("corredor", "hall")),
    ("closet", ("closet",)),
    ("garagem", ("garagem", "vaga")),
    ("jardim", ("jardim", "quintal")),
    ("area_externa", ("area externa", "área externa", "fachada", "externa")),
    ("piscina", ("piscina",)),
    ("academia", ("academia",)),
    ("churrasqueira", ("churrasqueira", "espaco gourmet", "espaço gourmet")),
)


@dataclass(frozen=True)
class ListingSummary:
    listing_id: str
    search_url: str
    detail_url: str
    area: int | None
    is_furnished: bool
    bedrooms: int | None
    address: str
    region: str


@dataclass(frozen=True)
class PhotoRecord:
    listing: ListingSummary
    photo_id: str
    subtitle: str
    label: str
    image_url: str
    image_path: Path


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str, fallback: str = "sem_label") -> str:
    slug = normalize_text(value)
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug or fallback


def label_from_subtitle(subtitle: str) -> str | None:
    normalized = normalize_text(subtitle)
    if not normalized:
        return None

    labels = [
        label
        for label, patterns in ENVIRONMENT_PATTERNS
        if any(normalize_text(pattern) in normalized for pattern in patterns)
    ]
    if not labels:
        return None

    unique_labels = list(dict.fromkeys(labels))
    return "_".join(unique_labels[:2])


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.7,en;q=0.6",
        }
    )
    return session


def fetch_text(session: requests.Session, url: str, timeout: float) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def extract_next_data(page_html: str, url: str) -> dict[str, Any]:
    match = NEXT_DATA_RE.search(page_html)
    if not match:
        raise ValueError(f"Could not find __NEXT_DATA__ in {url}")
    return json.loads(html.unescape(match.group(1)))


def iter_dict_values(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        for item in value.values():
            if isinstance(item, dict):
                yield item


def extract_listing_urls(page_html: str) -> dict[str, str]:
    urls: dict[str, str] = {}
    for url, listing_id in LISTING_URL_RE.findall(page_html):
        urls.setdefault(listing_id, url)
    return urls


def address_to_text(address: Any, region: str) -> str:
    if isinstance(address, dict):
        parts = [address.get("address"), address.get("street"), region, address.get("city")]
        return ", ".join(str(part) for part in parts if part)
    if address:
        return str(address)
    return region


def extract_search_listings(session: requests.Session, search_url: str, timeout: float) -> list[ListingSummary]:
    page_html = fetch_text(session, search_url, timeout)
    next_data = extract_next_data(page_html, search_url)
    initial_state = next_data["props"]["pageProps"]["initialState"]
    houses = initial_state.get("houses", {})
    listing_urls = extract_listing_urls(page_html)

    listings: list[ListingSummary] = []
    for house in iter_dict_values(houses):
        listing_id = str(house.get("id") or "")
        if not listing_id.isdigit():
            continue

        region = str(house.get("regionName") or house.get("neighbourhood") or "")
        detail_url = listing_urls.get(listing_id) or urljoin(search_url, f"/imovel/{listing_id}")
        listings.append(
            ListingSummary(
                listing_id=listing_id,
                search_url=search_url,
                detail_url=detail_url,
                area=safe_int(house.get("area")),
                is_furnished=bool(house.get("isFurnished")),
                bedrooms=safe_int(house.get("bedrooms")),
                address=address_to_text(house.get("address"), region),
                region=region,
            )
        )

    return listings


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_detail_house(session: requests.Session, detail_url: str, timeout: float) -> dict[str, Any]:
    page_html = fetch_text(session, detail_url, timeout)
    next_data = extract_next_data(page_html, detail_url)
    initial_state = next_data["props"]["pageProps"]["initialState"]
    return initial_state["house"]["houseInfo"]


def build_image_url(photo_url: str, image_size: str) -> str:
    if photo_url.startswith("http://") or photo_url.startswith("https://"):
        return photo_url
    return f"https://www.quintoandar.com.br/img/{image_size}/{photo_url.lstrip('/')}"


def output_extension(image_url: str) -> str:
    suffix = Path(urlparse(image_url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix
    return ".jpg"


def iter_labeled_photos(
    listing: ListingSummary,
    detail_house: dict[str, Any],
    output_dir: Path,
    image_size: str,
) -> Iterable[PhotoRecord]:
    photos = detail_house.get("photos") or []
    for index, photo in enumerate(photos, start=1):
        if not isinstance(photo, dict):
            continue
        subtitle = str(photo.get("subtitle") or "").strip()
        label = label_from_subtitle(subtitle)
        photo_url = str(photo.get("url") or "")
        if not label or not photo_url:
            continue

        image_url = build_image_url(photo_url, image_size)
        photo_id = str(photo.get("id") or index)
        filename = (
            f"{listing.listing_id}_{index:03d}_{photo_id}_{slugify(subtitle)}"
            f"{output_extension(image_url)}"
        )
        yield PhotoRecord(
            listing=listing,
            photo_id=photo_id,
            subtitle=subtitle,
            label=label,
            image_url=image_url,
            image_path=output_dir / "images" / label / filename,
        )


def download_image(
    session: requests.Session,
    record: PhotoRecord,
    timeout: float,
    overwrite: bool,
) -> int:
    if record.image_path.exists() and not overwrite:
        return record.image_path.stat().st_size

    record.image_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(record.image_url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"Expected image content, got {content_type}: {record.image_url}")

        tmp_path = record.image_path.with_suffix(record.image_path.suffix + ".tmp")
        bytes_written = 0
        with tmp_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    bytes_written += len(chunk)
                    file.write(chunk)
        tmp_path.replace(record.image_path)
    return bytes_written


def write_manifest_csv(records: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path = output_dir / "manifest.csv"
    if not records:
        csv_path.write_text("", encoding="utf-8")
        return

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_manifest_jsonl(records: list[dict[str, Any]], output_dir: Path) -> None:
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def record_to_manifest(record: PhotoRecord, bytes_written: int | None, downloaded: bool) -> dict[str, Any]:
    listing = record.listing
    return {
        "source": "quintoandar",
        "listing_id": listing.listing_id,
        "detail_url": listing.detail_url,
        "search_url": listing.search_url,
        "area_m2": listing.area,
        "is_furnished": listing.is_furnished,
        "bedrooms": listing.bedrooms,
        "region": listing.region,
        "address": listing.address,
        "photo_id": record.photo_id,
        "subtitle": record.subtitle,
        "label": record.label,
        "image_url": record.image_url,
        "image_path": str(record.image_path),
        "downloaded": downloaded,
        "bytes": bytes_written,
    }


def scrape(args: argparse.Namespace) -> dict[str, int]:
    if args.image_size not in VALID_IMAGE_SIZES:
        raise ValueError(f"--image-size must be one of: {', '.join(sorted(VALID_IMAGE_SIZES))}")

    session = build_session()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, Any]] = []
    stats = {
        "search_listings": 0,
        "eligible_listings": 0,
        "duplicate_listings": 0,
        "visited_details": 0,
        "photos_seen": 0,
        "photos_saved": 0,
        "photos_skipped_download_error": 0,
    }
    seen_listing_ids: set[str] = set()

    for search_url in args.search_url:
        listings = extract_search_listings(session, search_url, args.timeout)
        stats["search_listings"] += len(listings)

        for listing in listings:
            if args.max_listings is not None and stats["eligible_listings"] >= args.max_listings:
                break
            if listing.listing_id in seen_listing_ids:
                stats["duplicate_listings"] += 1
                continue
            seen_listing_ids.add(listing.listing_id)
            if listing.area is None or listing.area <= args.min_area:
                continue
            if not listing.is_furnished:
                continue

            stats["eligible_listings"] += 1
            time.sleep(args.delay)
            detail_house = extract_detail_house(session, listing.detail_url, args.timeout)
            stats["visited_details"] += 1

            detail_area = safe_int(detail_house.get("area"))
            detail_furnished = bool(detail_house.get("hasFurniture", listing.is_furnished))
            if detail_area is not None and detail_area <= args.min_area:
                continue
            if not detail_furnished:
                continue

            for photo in iter_labeled_photos(listing, detail_house, output_dir, args.image_size):
                if stats["photos_saved"] >= args.max_photos:
                    break
                stats["photos_seen"] += 1

                bytes_written: int | None = None
                downloaded = False
                if not args.dry_run:
                    try:
                        time.sleep(args.delay)
                        bytes_written = download_image(session, photo, args.timeout, args.overwrite)
                        downloaded = True
                    except Exception as error:  # noqa: BLE001
                        stats["photos_skipped_download_error"] += 1
                        print(f"[WARN] failed to download {photo.image_url}: {error}")
                        continue

                manifest_records.append(record_to_manifest(photo, bytes_written, downloaded))
                stats["photos_saved"] += 1

            if stats["photos_saved"] >= args.max_photos:
                break

    write_manifest_jsonl(manifest_records, output_dir)
    write_manifest_csv(manifest_records, output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape public QuintoAndar rental listings and download labeled room photos "
            "for external validation datasets."
        )
    )
    parser.add_argument(
        "--search-url",
        action="append",
        default=None,
        help=(
            "Search/listing URL to scrape. Can be repeated. "
            f"Default: {DEFAULT_SEARCH_URL}"
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--max-photos", type=int, default=1000)
    parser.add_argument("--max-listings", type=int, default=None)
    parser.add_argument("--image-size", default="med", choices=sorted(VALID_IMAGE_SIZES))
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build manifests without downloading images.")
    args = parser.parse_args()
    if args.search_url is None:
        args.search_url = [DEFAULT_SEARCH_URL]
    if args.max_photos < 1 or args.max_photos > 1000:
        parser.error("--max-photos must be between 1 and 1000")
    return args


def main() -> None:
    stats = scrape(parse_args())
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
