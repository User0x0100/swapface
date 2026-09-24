from __future__ import annotations

import argparse
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image

from tools import download_datasets as downloader
from tools import postprocess_dataset as postprocess


class DownloaderTests(unittest.TestCase):
    def test_endpoint_selection(self) -> None:
        self.assertIsNone(downloader.selected_endpoint(None))
        self.assertEqual(
            downloader.selected_endpoint("direct"),
            "https://huggingface.co",
        )
        self.assertEqual(
            downloader.selected_endpoint("mirror"),
            "https://hf-mirror.com",
        )

    def test_download_uses_huggingface_hub_and_creates_symlink(self) -> None:
        artifact = downloader.Artifact(
            name="TEST",
            repo_id="owner/repo",
            revision="deadbeef",
            filename="nested/data.bin",
            output=Path("test/data.bin"),
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_file = root / "cache/data.bin"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_bytes(b"data")

            def fake_hf_hub_download(**kwargs: object) -> str:
                self.assertEqual(kwargs["repo_id"], "owner/repo")
                self.assertEqual(kwargs["filename"], "nested/data.bin")
                self.assertEqual(kwargs["repo_type"], "dataset")
                self.assertEqual(kwargs["revision"], "deadbeef")
                self.assertEqual(kwargs["endpoint"], "https://hf-mirror.com")
                self.assertFalse(kwargs["force_download"])
                self.assertNotIn("local_dir", kwargs)
                return str(cache_file)

            with patch(
                "tools.download_datasets.hf_hub_download",
                side_effect=fake_hf_hub_download,
            ):
                result = downloader.download_artifact(
                    artifact=artifact,
                    root=root,
                    endpoint="https://hf-mirror.com",
                    force_download=False,
                )

            self.assertEqual(result, root / "test/data.bin")
            self.assertTrue(result.is_symlink())
            self.assertEqual(result.resolve(), cache_file.resolve())
            self.assertEqual(result.read_bytes(), b"data")

    def test_existing_symlink_is_refreshed_after_hub_download(self) -> None:
        artifact = downloader.Artifact(
            name="TEST",
            repo_id="owner/repo",
            revision="deadbeef",
            filename="data.bin",
            output=Path("test/data.bin"),
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_cache = root / "cache/old.bin"
            new_cache = root / "cache/new.bin"
            old_cache.parent.mkdir(parents=True)
            old_cache.write_bytes(b"old")
            new_cache.write_bytes(b"new")
            destination = root / artifact.output
            destination.parent.mkdir(parents=True)
            destination.symlink_to(old_cache)

            with patch(
                "tools.download_datasets.hf_hub_download",
                return_value=str(new_cache),
            ) as download:
                result = downloader.download_artifact(
                    artifact=artifact,
                    root=root,
                    endpoint=None,
                    force_download=False,
                )

            download.assert_called_once()
            self.assertTrue(result.is_symlink())
            self.assertEqual(result.resolve(), new_cache.resolve())
            self.assertEqual(result.read_bytes(), b"new")

    def test_regular_destination_is_not_replaced(self) -> None:
        artifact = downloader.Artifact(
            name="TEST",
            repo_id="owner/repo",
            revision="deadbeef",
            filename="data.bin",
            output=Path("test/data.bin"),
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_file = root / "cache/data.bin"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_bytes(b"new")
            destination = root / artifact.output
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"old")

            with patch(
                "tools.download_datasets.hf_hub_download",
                return_value=str(cache_file),
            ) as download:
                with self.assertRaisesRegex(RuntimeError, "refusing to replace non-symlink"):
                    downloader.download_artifact(
                        artifact=artifact,
                        root=root,
                        endpoint=None,
                        force_download=False,
                    )

            download.assert_not_called()
            self.assertEqual(destination.read_bytes(), b"old")


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
