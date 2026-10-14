"""Download the Olist dataset from Kaggle into data/raw/.

Safe to re-run: if all expected CSVs are already in data/raw/, nothing is
downloaded or overwritten.

Usage:
    uv run python src/download_data.py
"""

import shutil
from pathlib import Path

# import kagglehub

DATASET = "olistbr/brazilian-ecommerce"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

EXPECTED_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def missing_files() -> list[str]:
    return [name for name in EXPECTED_FILES if not (RAW_DIR / name).exists()]


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    to_fetch = missing_files()
    if not to_fetch:
        print(f"All {len(EXPECTED_FILES)} files already in {RAW_DIR}, skipping download.")
        return

    # kagglehub downloads into its own cache (~/.cache/kagglehub) and returns
    # that folder; we copy from there so the project has a fixed data path.
    import kagglehub

    cache_dir = Path(kagglehub.dataset_download(DATASET))
    print(f"Downloaded to cache: {cache_dir}")

    for name in to_fetch:
        source = cache_dir / name
        if not source.exists():
            raise FileNotFoundError(f"{name} not found in the Kaggle download at {cache_dir}")
        shutil.copy2(source, RAW_DIR / name)


def report() -> None:
    still_missing = missing_files()
    if still_missing:
        raise SystemExit(f"Missing after download: {still_missing}")

    print(f"\n{len(EXPECTED_FILES)} files in {RAW_DIR}:")
    for name in EXPECTED_FILES:
        size_mb = (RAW_DIR / name).stat().st_size / 1_000_000
        print(f"  {name:<45} {size_mb:7.1f} MB")


if __name__ == "__main__":
    download()
    report()
