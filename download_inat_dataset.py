from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from pyinaturalist import get_observations, get_taxa


@dataclass(frozen=True)
class SpeciesConfig:
    species: str
    common_name_de: str
    target: int = 500
    minimum: int = 200


def slugify_species(name: str) -> str:
    return name.strip().replace(" ", "_").replace("/", "_")


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_species_list(csv_path: Path, target: int, minimum: int) -> List[SpeciesConfig]:
    """
    Robust CSV read:
      - prefers UTF-8
      - falls back to Windows cp1252 for typical Excel exports
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp1252")

    missing = [c for c in ["species", "common_name_de"] if c not in df.columns]
    if missing:
        raise ValueError(f"species_list.csv missing columns: {missing}")

    configs: List[SpeciesConfig] = []
    for _, row in df.iterrows():
        configs.append(
            SpeciesConfig(
                species=str(row["species"]).strip(),
                common_name_de=str(row["common_name_de"]).strip(),
                target=target,
                minimum=minimum,
            )
        )
    return configs


def lookup_taxon_id(scientific_name: str) -> Optional[int]:
    """
    Resolve a scientific name to iNaturalist taxon_id.
    Prefer exact match when possible.
    """
    res = get_taxa(q=scientific_name, rank="species")
    taxa = res.get("results", [])
    if not taxa:
        # fallback without rank restriction
        res = get_taxa(q=scientific_name)
        taxa = res.get("results", [])
        if not taxa:
            return None

    exact = [t for t in taxa if str(t.get("name", "")).lower() == scientific_name.lower()]
    chosen = exact[0] if exact else taxa[0]
    return chosen.get("id")


def pick_one_photo_from_observation(obs: dict) -> Optional[dict]:
    photos = obs.get("photos") or []
    if not photos:
        return None
    # deterministic: first photo (keeps runs repeatable)
    return photos[0]


def best_photo_url(photo: dict) -> Optional[str]:
    url = photo.get("url")
    if not url:
        return None
    # Try to fetch a larger variant when possible
    if "square." in url:
        return url.replace("square.", "large.")
    return url


def download_image(url: str, out_path: Path, session: requests.Session, timeout_s: int = 30) -> bool:
    try:
        r = session.get(url, timeout=timeout_s)
        if r.status_code != 200 or not r.content:
            return False
        out_path.write_bytes(r.content)
        return True
    except Exception:
        return False


def iter_observations(
    taxon_id: int,
    per_page: int = 200,
    max_pages: int = 120,
    quality_grade: str = "research",
    photos: bool = True,
) -> Iterable[dict]:
    """
    Generator over observations with pagination.
    """
    page = 1
    while page <= max_pages:
        res = get_observations(
            taxon_id=taxon_id,
            quality_grade=quality_grade,
            photos=photos,
            per_page=per_page,
            page=page,
            order="desc",
            order_by="created_at",
        )
        results = res.get("results", [])
        if not results:
            break
        for obs in results:
            yield obs
        page += 1


def load_existing_ids(meta_csv: Path, species: str) -> Tuple[set, set]:
    """
    Load already-downloaded photo_ids and observation_ids for a given species.
    This enables resume/top-up without duplicates.
    """
    if not meta_csv.exists():
        return set(), set()

    try:
        old = pd.read_csv(meta_csv, encoding="utf-8")
    except UnicodeDecodeError:
        old = pd.read_csv(meta_csv, encoding="cp1252")

    if old.empty:
        return set(), set()

    old = old[old["species"].astype(str).str.strip() == species].copy()
    if old.empty:
        return set(), set()

    photo_ids = set(old["photo_id"].dropna().astype(int).tolist()) if "photo_id" in old.columns else set()
    obs_ids = set(old["observation_id"].dropna().astype(int).tolist()) if "observation_id" in old.columns else set()
    return photo_ids, obs_ids


def build_dataset_for_species(
    cfg: SpeciesConfig,
    base_dir: Path,
    session: requests.Session,
    sleep_s: float,
    seed: int,
    quality_grade: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Downloads images for a given species until cfg.target is reached.
    Resumes from existing observations.csv (skips known photo_id and observation_id).
    Returns (new_rows_df, stats).
    """
    random.seed(seed)

    taxon_id = lookup_taxon_id(cfg.species)
    if taxon_id is None:
        return pd.DataFrame(), {"status": 0, "error_taxon_not_found": 1}

    out_dir = base_dir / "raw" / slugify_species(cfg.species)
    safe_mkdir(out_dir)

    # Existing IDs for resume/top-up
    meta_csv = base_dir / "meta" / "observations.csv"
    existing_photo_ids, existing_obs_ids = load_existing_ids(meta_csv, cfg.species)

    # Count already downloaded files for this species (trusted by metadata)
    already = len(existing_photo_ids)

    # If already at/above target, nothing to do
    if already >= cfg.target:
        stats = {
            "taxon_id": int(taxon_id),
            "downloaded_before": int(already),
            "downloaded_now": 0,
            "downloaded_total": int(already),
            "target": int(cfg.target),
            "minimum": int(cfg.minimum),
            "meets_minimum": int(already >= cfg.minimum),
        }
        return pd.DataFrame(), stats

    rows: List[dict] = []
    seen_photo_ids = set(existing_photo_ids)
    seen_obs_ids = set(existing_obs_ids)

    needed = cfg.target - already

    obs_stream = iter_observations(
        taxon_id=taxon_id,
        per_page=200,
        max_pages=120,
        quality_grade=quality_grade,
        photos=True,
    )

    for obs in obs_stream:
        if len(rows) >= needed:
            break

        obs_id = obs.get("id")
        if obs_id is None or obs_id in seen_obs_ids:
            continue

        photo = pick_one_photo_from_observation(obs)
        if not photo:
            continue

        photo_id = photo.get("id")
        if photo_id is None or photo_id in seen_photo_ids:
            continue

        url = best_photo_url(photo)
        if not url:
            continue

        out_path = out_dir / f"{photo_id}.jpg"
        if out_path.exists():
            ok = True
        else:
            ok = download_image(url, out_path, session=session)
            time.sleep(sleep_s)

        if not ok:
            continue

        user = obs.get("user") or {}
        user_id = user.get("id")

        seen_obs_ids.add(obs_id)
        seen_photo_ids.add(photo_id)

        rows.append(
            {
                "species": cfg.species,
                "common_name_de": cfg.common_name_de,
                "taxon_id": taxon_id,
                "observation_id": obs_id,
                "observer_id": user_id,
                "photo_id": photo_id,
                "photo_url": url,
                "license": obs.get("license_code") or None,
                "quality_grade": obs.get("quality_grade") or None,
                "observed_on": obs.get("observed_on") or None,
                "local_path": str(out_path.as_posix()),
            }
        )

    df_new = pd.DataFrame(rows)
    total = already + len(df_new)

    stats = {
        "taxon_id": int(taxon_id),
        "downloaded_before": int(already),
        "downloaded_now": int(len(df_new)),
        "downloaded_total": int(total),
        "target": int(cfg.target),
        "minimum": int(cfg.minimum),
        "meets_minimum": int(total >= cfg.minimum),
        "still_missing_for_target": int(max(0, cfg.target - total)),
    }
    return df_new, stats


def merge_and_save_observations(obs_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new observations into existing observations.csv without duplicates by photo_id.
    """
    if obs_path.exists():
        try:
            old = pd.read_csv(obs_path, encoding="utf-8")
        except UnicodeDecodeError:
            old = pd.read_csv(obs_path, encoding="cp1252")

        merged = pd.concat([old, new_df], ignore_index=True) if not new_df.empty else old
    else:
        merged = new_df.copy()

    if not merged.empty and "photo_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["photo_id"], keep="first")

    merged.to_csv(obs_path, index=False, encoding="utf-8")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species_csv", type=str, required=True, help="Path to species_list.csv")
    parser.add_argument("--out_dir", type=str, default="data", help="Base output directory")
    parser.add_argument("--target", type=int, default=500, help="Target images per species")
    parser.add_argument("--minimum", type=int, default=200, help="Minimum images per species")
    parser.add_argument("--sleep", type=float, default=0.8, help="Sleep seconds between downloads (rate limiting)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop_below_minimum", action="store_true", help="Drop species with < minimum images")
    parser.add_argument("--quality_grade", type=str, default="research", choices=["research", "needs_id", "casual", "any"])
    args = parser.parse_args()

    base_dir = Path(args.out_dir)
    safe_mkdir(base_dir / "raw")
    safe_mkdir(base_dir / "meta")

    configs = read_species_list(Path(args.species_csv), target=args.target, minimum=args.minimum)

    all_new: List[pd.DataFrame] = []
    stats_rows: List[dict] = []

    with requests.Session() as session:
        for cfg in configs:
            print(f"\n=== {cfg.species} ({cfg.common_name_de}) ===")
            qg = args.quality_grade if args.quality_grade != "any" else "research"

            df_new, stats = build_dataset_for_species(
                cfg=cfg,
                base_dir=base_dir,
                session=session,
                sleep_s=args.sleep,
                seed=args.seed,
                quality_grade=qg,
            )

            stats_rows.append({"species": cfg.species, "common_name_de": cfg.common_name_de, **stats})
            if not df_new.empty:
                all_new.append(df_new)

    new_df = pd.concat(all_new, ignore_index=True) if all_new else pd.DataFrame()

    obs_path = base_dir / "meta" / "observations.csv"
    stats_path = base_dir / "meta" / "download_stats.csv"

    merged_obs = merge_and_save_observations(obs_path, new_df)

    stats_df = pd.DataFrame(stats_rows)

    # Recompute counts from merged observations (more reliable)
    if not merged_obs.empty:
        counts = merged_obs["species"].value_counts()
        stats_df["downloaded_total"] = stats_df["species"].map(counts).fillna(0).astype(int)
        stats_df["meets_minimum"] = (stats_df["downloaded_total"] >= args.minimum).astype(int)
        stats_df["still_missing_for_target"] = (args.target - stats_df["downloaded_total"]).clip(lower=0).astype(int)

    # Optionally drop below minimum by rewriting observations.csv
    if args.drop_below_minimum and not merged_obs.empty:
        ok_species = set(stats_df.loc[stats_df["meets_minimum"] == 1, "species"].tolist())
        merged_obs = merged_obs[merged_obs["species"].isin(ok_species)].copy()
        merged_obs.to_csv(obs_path, index=False, encoding="utf-8")

    stats_df.to_csv(stats_path, index=False, encoding="utf-8")

    print("\n=== SUMMARY (downloaded_total) ===")
    if merged_obs.empty:
        print("No images in observations.csv.")
        return

    print(merged_obs["species"].value_counts().sort_values(ascending=False).to_string())
    print(f"\nSaved observations: {obs_path}")
    print(f"Saved stats:        {stats_path}")


if __name__ == "__main__":
    main()