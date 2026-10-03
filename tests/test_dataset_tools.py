from __future__ import annotations

import argparse
import hashlib
import io
import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image

from tools import download_datasets as downloader
from tools import postprocess_dataset as postprocess


class DownloaderTests(unittest.TestCase):
    @staticmethod
    def artifact(payload: bytes = b"data") -> downloader.Artifact:
        return downloader.Artifact(
            name="TEST",
            repo_id="owner/repo",
            revision="deadbeef",
            filename="data.bin",
            output=Path("test/data.bin"),
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def test_endpoint_selection(self) -> None:
        self.assertEqual(downloader.selected_endpoint("direct"), "https://huggingface.co")
        self.assertEqual(downloader.selected_endpoint("mirror"), "https://hf-mirror.com")
        with patch.dict(os.environ, {"HF_ENDPOINT": "https://example.invalid/"}, clear=False):
            self.assertEqual(downloader.selected_endpoint(None), "https://example.invalid")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(downloader.selected_endpoint(None), "https://hf-mirror.com")

    def test_artifact_url_is_pinned_to_revision(self) -> None:
        artifact = downloader.Artifact(
            name="TEST",
            repo_id="owner/repo",
            revision="deadbeef",
            filename="nested/data file.bin",
            output=Path("test/data.bin"),
            size=10,
            sha256="0" * 64,
        )
        self.assertEqual(
            downloader.artifact_url(artifact, "https://hf-mirror.com"),
            "https://hf-mirror.com/datasets/owner/repo/resolve/deadbeef/nested/data%20file.bin",
        )

    def test_download_lock_rejects_second_instance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / "data.bin"
            with downloader.download_lock(destination):
                with self.assertRaisesRegex(RuntimeError, "already being downloaded"):
                    with downloader.download_lock(destination):
                        self.fail("second lock unexpectedly acquired")

    def test_parse_final_headers_uses_last_redirect_response(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            headers = Path(td) / "headers"
            headers.write_bytes(
                b"HTTP/1.1 200 Connection established\r\n\r\n"
                b"HTTP/2 302\r\nlocation: https://example.invalid/blob\r\n\r\n"
                b"HTTP/2 206\r\ncontent-range: bytes 4-7/10\r\ncontent-length: 4\r\n\r\n"
            )
            status, parsed = downloader._parse_final_headers(headers)
            self.assertEqual(status, 206)
            self.assertEqual(parsed["content-range"], "bytes 4-7/10")

    def test_curl_inherits_download_lock_fd(self) -> None:
        class FakeProcess:
            returncode = 0

            def communicate(self) -> tuple[None, bytes]:
                return None, b""

            def poll(self) -> int:
                return 0

        with tempfile.TemporaryDirectory() as td:
            part = Path(td) / "00000.part"
            process = FakeProcess()
            with patch("tools.download_datasets.subprocess.Popen", return_value=process) as popen:
                downloader._curl_range(
                    "https://example.invalid/data",
                    0,
                    3,
                    part,
                    threading.Event(),
                    set(),
                    threading.Lock(),
                    123,
                )

            self.assertEqual(popen.call_args.kwargs["pass_fds"], (123,))

    def test_invalid_range_response_is_rolled_back_and_retried(self) -> None:
        payload = b"abcd"
        with tempfile.TemporaryDirectory() as td:
            part = Path(td) / "00000.part"
            calls = 0

            def fake_curl_range(
                url: str,
                start: int,
                end: int,
                part_path: Path,
                stop_event: threading.Event,
                active_processes: set[subprocess.Popen[bytes]],
                active_lock: threading.Lock,
                lock_fd: int,
            ) -> tuple[int, bytes, int | None, str | None]:
                del url, stop_event, active_processes, active_lock, lock_fd
                nonlocal calls
                calls += 1
                with part_path.open("ab") as output:
                    output.write(payload[start : end + 1])
                if calls == 1:
                    return 0, b"", 200, None
                return 0, b"", 206, f"bytes {start}-{end}/{len(payload)}"

            with patch("tools.download_datasets._curl_range", side_effect=fake_curl_range):
                downloader._download_chunk(
                    url="https://example.invalid/data",
                    total_size=len(payload),
                    part_path=part,
                    start=0,
                    end=3,
                    retries=2,
                    stop_event=threading.Event(),
                    active_processes=set(),
                    active_lock=threading.Lock(),
                    lock_fd=0,
                )

            self.assertEqual(calls, 2)
            self.assertEqual(part.read_bytes(), payload)

    def test_interrupted_invalid_range_does_not_keep_unverified_bytes(self) -> None:
        payload = b"abcd"
        stop_event = threading.Event()
        with tempfile.TemporaryDirectory() as td:
            part = Path(td) / "00000.part"

            def fake_curl_range(
                url: str,
                start: int,
                end: int,
                part_path: Path,
                stop: threading.Event,
                active_processes: set[subprocess.Popen[bytes]],
                active_lock: threading.Lock,
                lock_fd: int,
            ) -> tuple[int, bytes, int | None, str | None]:
                del url, active_processes, active_lock, lock_fd
                with part_path.open("ab") as output:
                    output.write(payload[start : end + 1])
                stop.set()
                return 0, b"", 200, None

            with patch("tools.download_datasets._curl_range", side_effect=fake_curl_range):
                downloader._download_chunk(
                    url="https://example.invalid/data",
                    total_size=len(payload),
                    part_path=part,
                    start=0,
                    end=3,
                    retries=0,
                    stop_event=stop_event,
                    active_processes=set(),
                    active_lock=threading.Lock(),
                    lock_fd=0,
                )

            self.assertEqual(part.read_bytes(), b"")

    def test_unverified_tail_is_truncated_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            part = Path(td) / "00000.part"
            part.write_bytes(b"abcdef")
            downloader._write_verified_size(part, 3)
            part.write_bytes(b"abcdefXYZ")

            verified = downloader._normalize_part(part, 16)

            self.assertEqual(verified, 3)
            self.assertEqual(part.read_bytes(), b"abc")

    def test_full_sized_unverified_chunk_is_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            part = Path(td) / "00000.part"
            part.write_bytes(b"abcd")

            verified = downloader._normalize_part(part, 4)

            self.assertEqual(verified, 0)
            self.assertEqual(part.read_bytes(), b"")

    def test_assemble_hashes_while_copying(self) -> None:
        payload = b"abcdefgh"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            destination = root / "data.bin"
            part_dir = root / "data.bin.parts"
            part_dir.mkdir()
            for index, data in enumerate((payload[:4], payload[4:])):
                part = part_dir / f"{index:05d}.part"
                part.write_bytes(data)
                downloader._write_verified_size(part, len(data))

            with patch("tools.download_datasets._sha256", side_effect=AssertionError("second pass SHA should not run")):
                downloader._assemble(
                    destination,
                    part_dir,
                    [(0, 0, 3), (1, 4, 7)],
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                )

            self.assertEqual(destination.read_bytes(), payload)

    def test_download_resumes_persistent_parts_and_assembles_file(self) -> None:
        payload = b"abcdefghij"
        artifact = self.artifact(payload)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            destination = root / artifact.output
            part_dir = destination.with_name(destination.name + ".parts")
            url = downloader.artifact_url(artifact, "https://hf-mirror.com")
            downloader._write_manifest(
                part_dir,
                url=url,
                size=len(payload),
                sha256=artifact.sha256,
                chunk_size=4,
                force=False,
            )
            part0 = part_dir / "00000.part"
            part1 = part_dir / "00001.part"
            part0.write_bytes(payload[:4])
            part1.write_bytes(payload[4:6])
            downloader._write_verified_size(part0, 4)
            downloader._write_verified_size(part1, 2)
            calls: list[tuple[int, int]] = []

            def fake_curl_range(
                url_arg: str,
                start: int,
                end: int,
                part_path: Path,
                stop_event: threading.Event,
                active_processes: set[subprocess.Popen[bytes]],
                active_lock: threading.Lock,
                lock_fd: int,
            ) -> tuple[int, bytes, int | None, str | None]:
                del stop_event, active_processes, active_lock, lock_fd
                self.assertEqual(url_arg, url)
                calls.append((start, end))
                with part_path.open("ab") as output:
                    output.write(payload[start : end + 1])
                return 0, b"", 206, f"bytes {start}-{end}/{len(payload)}"

            with patch("tools.download_datasets._curl_range", side_effect=fake_curl_range):
                result = downloader.download_artifact(
                    artifact,
                    root,
                    "https://hf-mirror.com",
                    workers=2,
                    chunk_size=4,
                    retries=1,
                    force_download=False,
                )

            self.assertEqual(result, destination)
            self.assertEqual(result.read_bytes(), payload)
            self.assertFalse(part_dir.exists())
            self.assertNotIn((0, 3), calls)
            self.assertIn((6, 7), calls)
            self.assertIn((8, 9), calls)

    def test_complete_destination_requires_matching_sha256(self) -> None:
        artifact = self.artifact(b"data")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            destination = root / artifact.output
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"DATA")
            with self.assertRaisesRegex(RuntimeError, "failed integrity check"):
                downloader.download_artifact(
                    artifact,
                    root,
                    "https://hf-mirror.com",
                    workers=2,
                    chunk_size=2,
                    retries=1,
                    force_download=False,
                )

    def test_complete_destination_is_reused_without_network(self) -> None:
        artifact = self.artifact(b"data")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            destination = root / artifact.output
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"data")
            with patch("tools.download_datasets._curl_range") as download:
                result = downloader.download_artifact(
                    artifact,
                    root,
                    "https://hf-mirror.com",
                    workers=2,
                    chunk_size=2,
                    retries=1,
                    force_download=False,
                )
            self.assertEqual(result, destination)
            download.assert_not_called()

    def test_terminate_processes_stops_active_child(self) -> None:
        process = subprocess.Popen(["sleep", "30"])
        active = {process}
        lock = threading.Lock()
        downloader._terminate_processes(active, lock)
        process.wait(timeout=2)
        self.assertIsNotNone(process.returncode)

    def test_fairface_artifacts_merge_train_and_validation_into_one_pool(self) -> None:
        artifacts = downloader.DATASETS["fairface"]
        self.assertEqual(len(artifacts), 3)
        self.assertTrue(all(artifact.repo_id == "HuggingFaceM4/FairFace" for artifact in artifacts))
        self.assertTrue(all(artifact.revision == "54d573cdb8b5af490ba8da9da2799628f6e5c496" for artifact in artifacts))
        self.assertTrue(all(artifact.filename.startswith("0.25/") and artifact.filename.endswith(".parquet") for artifact in artifacts))
        self.assertEqual(
            [artifact.output for artifact in artifacts],
            [
                Path("fairface/part-00000.parquet"),
                Path("fairface/part-00001.parquet"),
                Path("fairface/part-00002.parquet"),
            ],
        )
        self.assertEqual([artifact.size for artifact in artifacts], [250_030_031, 250_217_804, 63_189_799])
        self.assertTrue(all(len(artifact.sha256) == 64 for artifact in artifacts))
        self.assertEqual(sum("/train-" in artifact.filename for artifact in artifacts), 2)
        self.assertEqual(sum("/validation-" in artifact.filename for artifact in artifacts), 1)


class PostprocessTests(unittest.TestCase):
    def make_zip(
        self,
        path: Path,
        members: list[tuple[str, tuple[int, int], int]],
    ) -> None:
        with ZipFile(path, "w") as archive:
            for member, image_size, value in members:
                image_bytes = io.BytesIO()
                Image.new("RGB", image_size, (value, 20, 30)).save(
                    image_bytes,
                    "PNG",
                )
                archive.writestr(member, image_bytes.getvalue())

    def args(self, archives: list[Path], output: Path) -> argparse.Namespace:
        return argparse.Namespace(
            archives=archives,
            output=output,
            size=8,
            workers=2,
        )


    def test_fairface_race_grouping(self) -> None:
        self.assertEqual(postprocess.fairface_group(0), "asian")
        self.assertEqual(postprocess.fairface_group(1), "indian")
        self.assertEqual(postprocess.fairface_group(6), "asian")
        self.assertEqual(postprocess.fairface_group(2), "black")
        self.assertEqual(postprocess.fairface_group(3), "white")
        self.assertEqual(postprocess.fairface_group(4), "middle_eastern")
        self.assertEqual(postprocess.fairface_group(5), "latino")
        with self.assertRaisesRegex(ValueError, "unsupported FairFace race label"):
            postprocess.fairface_group(7)

    def test_fairface_item_key_is_stable_across_shards(self) -> None:
        self.assertEqual(
            postprocess.fairface_item_key(Path("part-00000.parquet"), 42),
            "part-00000_000042",
        )
        self.assertEqual(
            postprocess.fairface_item_key(Path("part-00002.parquet"), 3),
            "part-00002_000003",
        )

    def test_fairface_manifest_survives_json_round_trip(self) -> None:
        args = argparse.Namespace(mobilenet=False, confidence=0.9)
        manifest = postprocess.fairface_manifest_value(
            args,
            [postprocess.ArchiveIdentity(path="/tmp/a.parquet", size=10, mtime_ns=20)],
        )
        import json

        self.assertEqual(json.loads(json.dumps(manifest)), manifest)

    def test_output_lock_rejects_second_instance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "out"
            with postprocess.output_lock(output):
                with self.assertRaisesRegex(RuntimeError, "already being processed"):
                    with postprocess.output_lock(output):
                        self.fail("second lock unexpectedly acquired")

    def test_process_and_resume_missing_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "images.zip"
            self.make_zip(
                archive,
                [
                    ("nested/one.png", (16, 16), 10),
                    ("nested/two.png", (16, 16), 20),
                ],
            )
            args = self.args([archive], root / "out")
            items, identities = postprocess.scan_archives(args.archives)

            postprocess.run(args, items, identities)
            one = args.output / "one.png"
            two = args.output / "two.png"
            one_bytes = one.read_bytes()
            two.unlink()
            stale_part = args.output / "two.png.stale.part"
            stale_part.write_bytes(b"partial")

            args.workers = 1
            postprocess.run(args, items, identities)

            self.assertEqual(one.read_bytes(), one_bytes)
            self.assertFalse(stale_part.exists())
            self.assertTrue(two.is_file())
            with Image.open(two) as image:
                self.assertEqual(image.size, (8, 8))
                self.assertEqual(image.mode, "RGB")

    def test_changed_size_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "images.zip"
            self.make_zip(archive, [("one.png", (16, 16), 10)])
            args = self.args([archive], root / "out")
            items, identities = postprocess.scan_archives(args.archives)
            postprocess.run(args, items, identities)

            args.size = 16
            with self.assertRaisesRegex(RuntimeError, "different postprocess run"):
                postprocess.run(args, items, identities)

    def test_changed_archive_rejects_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "images.zip"
            self.make_zip(archive, [("one.png", (16, 16), 10)])
            args = self.args([archive], root / "out")
            items, identities = postprocess.scan_archives(args.archives)
            postprocess.run(args, items, identities)

            self.make_zip(
                archive,
                [
                    ("one.png", (16, 16), 10),
                    ("two.png", (16, 16), 20),
                ],
            )
            changed_items, changed_identities = postprocess.scan_archives(args.archives)
            with self.assertRaisesRegex(RuntimeError, "different postprocess run"):
                postprocess.run(args, changed_items, changed_identities)

    def test_stale_manifest_temp_is_ignored_on_first_start(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "images.zip"
            self.make_zip(archive, [("one.png", (16, 16), 10)])
            args = self.args([archive], root / "out")
            args.output.mkdir()
            stale = args.output / f"{postprocess.MANIFEST_NAME}.stale.tmp"
            stale.write_bytes(b"partial")

            items, identities = postprocess.scan_archives(args.archives)
            postprocess.run(args, items, identities)

            self.assertFalse(stale.exists())
            self.assertTrue((args.output / "one.png").is_file())

    def test_nonempty_output_without_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "images.zip"
            self.make_zip(archive, [("one.png", (16, 16), 10)])
            args = self.args([archive], root / "out")
            args.output.mkdir()
            (args.output / "old.png").write_bytes(b"old")

            items, identities = postprocess.scan_archives(args.archives)
            with self.assertRaisesRegex(RuntimeError, "not empty"):
                postprocess.run(args, items, identities)

    def test_flat_name_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "first.zip"
            second = root / "second.zip"
            self.make_zip(first, [("a/same.png", (16, 16), 10)])
            self.make_zip(second, [("b/same.jpg", (16, 16), 20)])

            with self.assertRaisesRegex(RuntimeError, "flat-name collision"):
                postprocess.scan_archives([first, second])

    def test_duplicate_zip_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "duplicate.zip"
            image_bytes = io.BytesIO()
            Image.new("RGB", (16, 16), (10, 20, 30)).save(
                image_bytes,
                "PNG",
            )
            with ZipFile(archive, "w") as zip_file:
                zip_file.writestr("same.png", image_bytes.getvalue())
                zip_file.writestr("same.png", image_bytes.getvalue())

            with self.assertRaisesRegex(RuntimeError, "duplicate ZIP member"):
                postprocess.scan_archives([archive])

    def test_non_square_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "images.zip"
            self.make_zip(archive, [("bad.png", (16, 12), 10)])
            args = self.args([archive], root / "out")
            items, identities = postprocess.scan_archives(args.archives)

            with self.assertRaisesRegex(
                RuntimeError,
                r"failed to process .*images\.zip::bad\.png",
            ):
                postprocess.run(args, items, identities)

            self.assertFalse((args.output / "bad.png").exists())


if __name__ == "__main__":
    unittest.main()
