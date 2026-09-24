#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_ENDPOINTS = {
    "direct": "https://huggingface.co",
    "mirror": "https://hf-mirror.com",
}


@dataclass(frozen=True, slots=True)
class Artifact:
    name: str
    repo_id: str
    revision: str
    filename: str
    output: Path


DATASETS: dict[str, tuple[Artifact, ...]] = {
    "lpff": (
        Artifact(
            name="LPFF",
            repo_id="onethousand/LPFF",
            revision="f598e6ea5c871f9b557d11b07d8d4248f127ea08",
            filename="LPFF-dataset/crop/stylegan.zip",
            output=Path("lpff/stylegan.zip"),
        ),
    ),
    "ffhq": (
        Artifact(
            name="FFHQ-1",
            repo_id="Iceclear/FFHQ-HQ1024",
            revision="93e4ac2b99e06b492efcee5d2182bdca6b6f4671",
            filename="FFHQ-1024-1.zip",
            output=Path("ffhq/FFHQ-1024-1.zip"),
        ),
        Artifact(
            name="FFHQ-2",
            repo_id="Iceclear/FFHQ-HQ1024",
            revision="93e4ac2b99e06b492efcee5d2182bdca6b6f4671",
            filename="FFHQ-1024-2.zip",
            output=Path("ffhq/FFHQ-1024-2.zip"),
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download project datasets with huggingface_hub.")
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=(*DATASETS, "all"),
        default="all",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/downloads"),
        help="Symlink root. Default: datasets/downloads.",
    )
    parser.add_argument(
        "--source",
        choices=tuple(HF_ENDPOINTS),
        help="Override the Hugging Face endpoint. Omit to respect HF_ENDPOINT or the official default.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force huggingface_hub to download the file again.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print selected artifacts without downloading.",
    )
    return parser.parse_args()


def selected_artifacts(dataset: str) -> list[Artifact]:
    if dataset == "all":
        return [artifact for artifacts in DATASETS.values() for artifact in artifacts]
    return list(DATASETS[dataset])


def selected_endpoint(source: str | None) -> str | None:
    return None if source is None else HF_ENDPOINTS[source]


def replace_symlink(target: Path, destination: Path) -> None:
    if destination.exists() and not destination.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink destination: {destination}")

    fd, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(fd)
    os.unlink(temporary)
    try:
        os.symlink(target, temporary)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def download_artifact(
    artifact: Artifact,
    root: Path,
    endpoint: str | None,
    force_download: bool,
) -> Path:
    destination = root / artifact.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink destination: {destination}")

    downloaded = Path(
        hf_hub_download(
            repo_id=artifact.repo_id,
            filename=artifact.filename,
            repo_type="dataset",
            revision=artifact.revision,
            endpoint=endpoint,
            force_download=force_download,
        )
    )
    replace_symlink(downloaded, destination)

    print(f"[{artifact.name}] COMPLETE: {destination} -> {downloaded}")
    return destination


def main() -> None:
    args = parse_args()
    artifacts = selected_artifacts(args.dataset)
    endpoint = selected_endpoint(args.source)

    if args.list:
        for artifact in artifacts:
            print(f"{artifact.name}\t{artifact.repo_id}\t{artifact.revision}\t{artifact.filename}\t{artifact.output}")
        return

    for artifact in artifacts:
        download_artifact(
            artifact=artifact,
            root=args.root,
            endpoint=endpoint,
            force_download=args.force,
        )


if __name__ == "__main__":
    main()
