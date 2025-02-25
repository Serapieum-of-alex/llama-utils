import shutil
import unittest
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from docling.document_converter import DocumentConverter as docling_doc_converter
from llama_index.core.schema import ImageDocument

from llama_utils.retrieval.pdf_reader import DocumentConverter, PDFReader


class TestDocumentConverterE2E:
    """End-to-end tests for DocumentConverterWrapper."""

    def test_convert_valid_pdf(self, geoscience_pdf: Path):
        """Test converting a valid PDF file to Markdown successfully."""
        wrapper = DocumentConverter()
        md_file, images_dir = wrapper.convert(geoscience_pdf)
        assert md_file.exists()
        assert md_file.suffix == ".md"
        assert images_dir.exists() and images_dir.is_dir()
        im_list = list(images_dir.iterdir())
        assert len(im_list) == 1
        # clean up
        md_file.unlink()
        try:
            shutil.rmtree(images_dir)
        except PermissionError:
            pass

    def test_convert_invalid_pdf_path(self):
        """Test converting an invalid PDF path raises an error."""
        wrapper = DocumentConverter()
        pdf_path = Path("tests/nonexistent.pdf")
        with pytest.raises(FileNotFoundError):
            wrapper.convert(pdf_path)


class TestPDFReaderE2E:
    """End-to-end tests for PDFReader."""

    @classmethod
    def setup_class(cls):
        """Initialize PDFReader for end-to-end tests."""
        cls.reader = PDFReader()

    def test_extract_figures_data(self):
        """Test extracting figure data from realistic markdown content."""
        md_text = """Some introduction text ...

        Figure 2. Study area: The main campus ...
        ![Image](paper_artifacts\\image_000000_ccc2c343.png)

        Some other random text ...

        Figure 3. Another figure's caption.
        ![Image](paper_artifacts\\image_000001_abc123.png)
        """

        expected = [
            {
                "figure_number": "Figure 2.",
                "caption_text": "Study area: The main campus ...",
                "image_path": "paper_artifacts\\image_000000_ccc2c343.png",
            },
            {
                "figure_number": "Figure 3.",
                "caption_text": "Another figure's caption.",
                "image_path": "paper_artifacts\\image_000001_abc123.png",
            },
        ]
        result = self.reader.extract_figures_data(md_text)

        assert result == expected

    def test_create_image_document(self):
        """Test creating an ImageDocument from a real image file."""
        image_path = Path(
            "tests/data/docling-parsed-markdown_artifacts/image_000000_ccc2c343942b491ee2456fc1c02f25091363aa6075b1a6d115247ab0096c8d17.png"
        )
        caption = "any caption related to the fist figure."
        metadata = {"any-key": "any-value"}

        image_doc = self.reader.create_image_document(
            image_path, caption, metadata=metadata
        )
        assert isinstance(image_doc, ImageDocument)
        assert image_doc.doc_id == f"img-{image_path.name}"
        assert image_doc.metadata["filename"] == image_path.name
        assert image_doc.metadata["any-key"] == "any-value"
        assert image_doc.text == f"figure caption: {caption}"
        assert isinstance(image_doc.image, str)
        assert image_doc.id_ == f"img-{image_path.name}"

    def test_parse_pdf(
        self, geoscience_pdf: Path, geoscience_paper_artifacts: Dict[str, str]
    ):
        """Test parsing a real PDF file to generate markdown and image documents."""

        result = self.reader.parse_pdf(geoscience_pdf)

        image_docs = result["images"]
        assert len(image_docs) == 1
        assert isinstance(image_docs[0], ImageDocument)
        assert image_docs[0].doc_id.startswith("img-")
        assert (
            Path(geoscience_paper_artifacts["md_file"]).name == result["markdown"].name
        )
        # clean
        md_file = result["markdown"]
        md_file.unlink()
        try:
            images_dir = md_file.parent / f"{md_file.stem}_artifacts"
            shutil.rmtree(images_dir)
        except PermissionError:
            pass

    def test_parse_pdf_invalid_file(self):
        """Test parsing an invalid PDF file raises an error."""
        pdf_path = "tests/invalid.pdf"
        with pytest.raises(FileNotFoundError):
            self.reader.parse_pdf(pdf_path)


class TestDocumentConverterMock(unittest.TestCase):

    def test_constructor(self):
        """
        test the constructor of the DocumentConvert
        """
        converter = DocumentConverter()
        assert isinstance(converter.converter, docling_doc_converter)

    @patch("llama_utils.retrieval.pdf_reader.DocumentConverter")
    def test_convert(self, mock_converter):
        """
        mock the converter and test the convert method
        """
        mock_converter = mock_converter.return_value
        mock_document = MagicMock()
        mock_converter.convert.return_value = mock_document

        wrapper = DocumentConverter(mock_converter)
        pdf_path = Path("test.pdf")
        with patch("pathlib.Path.exists", return_value=True):
            md_file, images_dir = wrapper.convert(pdf_path)

        self.assertEqual(md_file, pdf_path.with_suffix(".md"))
        self.assertEqual(str(images_dir), f"{pdf_path.stem}_artifacts")
        mock_document.document.save_as_markdown.assert_called_once()


class TestPDFReader(unittest.TestCase):

    def setUp(self):
        self.reader = PDFReader()

    @patch(
        "builtins.open", new_callable=unittest.mock.mock_open, read_data=b"image data"
    )
    def test_create_image_document(self, mock_open):
        image_doc = self.reader.create_image_document("image.png", "Test Caption")
        self.assertIsInstance(image_doc, ImageDocument)
        self.assertEqual(image_doc.text, "figure caption: Test Caption")

    @patch.object(DocumentConverter, "convert")
    @patch(
        "builtins.open", new_callable=unittest.mock.mock_open, read_data=b"image data"
    )
    @patch(
        "pathlib.Path.read_text",
        return_value="Figure 1. Caption\n![Image](image.png)",
    )
    def test_parse_pdf(self, mock_read, mock_open, mock_convert):
        mock_convert.return_value = (Path("test.md"), "")
        result = self.reader.parse_pdf("test.pdf")
        self.assertIn("markdown", result)
        self.assertIn("images", result)
        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(result["images"][0].text, "figure caption: Caption")
