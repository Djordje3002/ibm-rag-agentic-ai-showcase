import zipfile
from pathlib import Path

import pytest

from ibm_rag_agentic_showcase.multimodal_augmentation import (
    augment_recipes,
    parse_image_urls,
    recipe_image_path,
    safe_extract_zip,
)


def test_parse_image_urls_accepts_dataset_string():
    value = "['https://example.com/one.png', 'https://example.com/two.png']"

    assert parse_image_urls(value) == [
        "https://example.com/one.png",
        "https://example.com/two.png",
    ]


def test_parse_image_urls_rejects_non_string_items():
    with pytest.raises(ValueError, match="list of URL strings"):
        parse_image_urls(["https://example.com/image.png", 42])


def test_recipe_image_path_uses_recipe_id(tmp_path: Path):
    expected = tmp_path / "recipe7.png"
    expected.write_bytes(b"png")

    assert recipe_image_path({"id": 7}, tmp_path) == expected


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("../escape.txt", "nope")

    with pytest.raises(ValueError, match="Unsafe ZIP path"):
        safe_extract_zip(archive, tmp_path / "output")


def test_augment_recipes_adds_caption(tmp_path: Path):
    image_path = tmp_path / "recipe1.png"
    image_path.write_bytes(b"fake-png")
    calls = []

    def fake_vision(system, prompt, image_bytes, media_type):
        calls.append((system, prompt, image_bytes, media_type))
        return "A bright bowl of seasonal food."

    result = augment_recipes(
        [{"id": 1, "name": "Seasonal Bowl"}],
        tmp_path,
        fake_vision,
    )

    assert result[0]["image_description"] == "A bright bowl of seasonal food."
    assert calls[0][2:] == (b"fake-png", "image/png")
