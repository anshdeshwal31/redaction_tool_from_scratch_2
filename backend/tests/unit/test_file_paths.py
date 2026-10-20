"""File paths and file names: detection and same-shape surrogates."""

from __future__ import annotations

from mlredact.config.loader import load_config
from mlredact.core.types import ActionKind, EntityType
from mlredact.detect.engine import detect_document
from mlredact.surrogate.drbg import Chooser
from mlredact.surrogate.generators import SurrogateFactory
from mlredact.text.normalize import match_key
from support import pages_from_lines

CFG = load_config("broad")
E = EntityType


class TestFilePaths:
    LINES = (
        "File: S:\\Clients\\Smith J\\IME report.docx Page 1 of 3",
        "Saved as Forbes_DPI_report.docx.",
        "Source /home/jsmith/reports/labrom_ime.pdf",
        "Scan C:\\temp\\scan0001 received",
        "See www.example.com/a.pdf or write to john.smith@example.com, and/or call 3 times/day.",
    )

    def test_paths_and_names_are_detected(self) -> None:
        result = detect_document(pages_from_lines(list(self.LINES)), CFG, "doc")
        found = {m.value for m in result.mentions if m.entity_type is E.FILE_PATH}
        assert found == {
            "S:\\Clients\\Smith J\\IME report.docx",
            "Forbes_DPI_report.docx.",
            "/home/jsmith/reports/labrom_ime.pdf",
            "C:\\temp\\scan0001",
        }

    def test_surrogate_keeps_the_shape_and_replaces_the_names(self) -> None:
        factory = SurrogateFactory(Chooser(b"k" * 32), ["smith", "jsmith", "forbes"])
        out = factory.file_path("C:\\Users\\jsmith\\Documents\\Smith_John_2024-03-01.pdf")
        assert out.startswith("C:\\Users\\") and "\\Documents\\" in out
        assert out.endswith("_2024-03-01.pdf")
        assert "smith" not in out.lower() and "John" not in out
        assert factory.file_path("C:\\Users\\jsmith\\Documents\\Smith_John_2024-03-01.pdf") == out

    def test_surrogate_never_equals_a_generic_original(self) -> None:
        factory = SurrogateFactory(Chooser(b"k" * 32), [match_key("C:\\Users\\Documents\\report.docx")])
        out = factory.file_path("C:\\Users\\Documents\\report.docx")
        assert out != "C:\\Users\\Documents\\report.docx" and out.endswith(".docx")

    def test_folder_names_with_spaces_are_detected_whole(self) -> None:
        lines = [
            "Path C:\\Users\\John Smith",
            "Saved C:\\Users\\jsmith\\Documents and sent",
            "From /Users/jsmith/Smith Family/report.pdf today",
            "Mac /Users/John Smith",
        ]
        result = detect_document(pages_from_lines(lines), CFG, "doc")
        found = {m.value for m in result.mentions if m.entity_type is E.FILE_PATH}
        assert found == {
            "C:\\Users\\John Smith",
            "C:\\Users\\jsmith\\Documents",
            "/Users/jsmith/Smith Family/report.pdf",
            "/Users/John Smith",
        }

    def test_a_single_changeable_character_becomes_a_word(self) -> None:
        for value in ("IME report 3.docx", "S:\\Scans\\scan 3.pdf", "scan J.pdf"):
            factory = SurrogateFactory(Chooser(b"k" * 32), [match_key(value)])
            out = factory.file_path(value)
            assert match_key(out) != match_key(value) and out.endswith(value[-4:])

    def test_every_profile_replaces_them(self) -> None:
        for profile in ("broad", "claimant_focused", "maximal"):
            assert load_config(profile).policy.entity_actions[E.FILE_PATH] is ActionKind.SURROGATE
