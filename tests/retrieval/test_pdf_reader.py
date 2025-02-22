import shutil
from pathlib import Path

from llama_index.core.schema import ImageDocument

from llama_utils.retrieval.pdf_reader import (
    create_image_document,
    extract_figures_data,
    parse_pdf_with_docling,
)


def test_extract_figures_data():
    pdf_text = """Some introduction text ...

    Figure 2. Study area: The main campus ...
    ![Image](paper_artifacts\\image_000000_ccc2c343.png)

    Some other random text ...

    Figure 3. Another figure's caption.
    ![Image](paper_artifacts\\image_000001_abc123.png)
    """

    results = extract_figures_data(pdf_text)
    assert len(results) == 2
    assert results[0]["figure_number"] == "Figure 2."
    assert results[0]["caption_text"] == "Study area: The main campus ..."
    assert results[0]["image_path"] == "paper_artifacts\\image_000000_ccc2c343.png"
    assert results[1]["figure_number"] == "Figure 3."
    assert results[1]["caption_text"] == "Another figure's caption."
    assert results[1]["image_path"] == "paper_artifacts\\image_000001_abc123.png"


def test_create_image_document():
    """Test create_image_document function."""
    image_path = Path(
        "tests/data/docling-parsed-markdown_artifacts/image_000000_ccc2c343942b491ee2456fc1c02f25091363aa6075b1a6d115247ab0096c8d17.png"
    )
    caption = "any caption related to the fist figure."
    metadata = {"any-key": "any-value"}
    im_document = create_image_document(
        image_path, **{"caption_text": caption, "metadata": metadata}
    )
    assert isinstance(im_document, ImageDocument)
    assert im_document.doc_id == f"img-{image_path.name}"
    assert im_document.metadata["filename"] == image_path.name
    assert im_document.metadata["any-key"] == "any-value"
    assert im_document.text == f"figure caption: {caption}"
    assert isinstance(im_document.image, str)
    assert im_document.id_ == f"img-{image_path.name}"


def test_parse_with_docling(geoscience_pdf: Path):
    md_file, ims_rdir = parse_pdf_with_docling(geoscience_pdf)
    assert md_file.exists()
    assert ims_rdir.exists() and ims_rdir.is_dir()
    im_list = list(ims_rdir.iterdir())
    assert len(im_list) == 1
    # clean up
    md_file.unlink()
    try:
        shutil.rmtree(ims_rdir)
    except PermissionError:
        pass
