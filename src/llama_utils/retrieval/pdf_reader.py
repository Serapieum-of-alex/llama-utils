"""PDF Text Extraction Utilities."""

import base64
import re
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode
from llama_index.core.schema import ImageDocument

IMAGE_RESOLUTION_SCALE = 2.0
IMAGE_DIR_SUFFIX = "_artifacts"


def extract_figures_data(pdf_text: str):
    r"""Extract Figure Data from PDF Text.

    Extract figure data (local path/ caption /figure number) from a PDF text dump,
    purely via regex. We assume each figure looks like:

        Figure 2. Study area: ...
        ![Image](paper_artifacts\\image_000000_xyz.png)

    where "Figure 2." or "Figure 12." etc. precedes the caption text,
    and the actual image reference is on a separate line starting with ![Image]()

    Parameters
    ----------
    pdf_text : str
        The entire PDF content as plain text.

    Returns
    -------
    list of dict
        Each dict contains:
            {
                "figure_number": "Figure 2.",
                "caption_text": "Study area: ...",
                "image_path": "paper_artifacts\\image_000000_xyz.png"
            }

    Examples
    --------
    ```python
    >>> pdf_text = '''Some introduction text ...\n
    ...             Figure 2. Study area: The main campus ...
    ...             "![Image](paper_artifacts\\image_000000_ccc2c343.png)
    ...
    ...            "Some other random text ...
    ...
    ...             "Figure 3. Another figure's caption.
    ...             "![Image](paper_artifacts\\image_000001_abc123.png)
    ...             '''
    >>> results = extract_figures_data(pdf_text)
    >>> print(results) # doctest: +SKIP
    [
        {
            'figure_number': 'Figure 2.',
            'caption_text': 'Study area: The main campus ...',
            'image_path': 'paper_artifacts\\image_000000_ccc2c343.png'
        },
        {
            'figure_number': 'Figure 3.',
            'caption_text': "Another figure's caption.",
            'image_path': 'paper_artifacts\\image_000001_abc123.png'
        }
    ]
    ```
    """
    # Regex Explanation:
    # 1) (Figure\s+\d+\.\s*) captures text like "Figure 2. " or "Figure 10. "
    # 2) (.*?) captures the figure caption until ...
    # 3) \n?\!\[Image\]\((.*?)\) looks for an optional newline, then "![Image](",
    #    then captures the path inside parentheses, then a closing ")"
    #
    # The DOTALL flag (re.DOTALL) makes the '.' match newlines, so we can capture
    # multi-line captions if they exist.
    pattern = re.compile(
        r"(Figure\s+\d+\.\s*)(.*?)\n?\!\[Image\]\((.*?)\)", flags=re.DOTALL
    )

    matches = pattern.findall(pdf_text)

    figures = []
    for match in matches:
        figure_number_raw = match[0]  # e.g. "Figure 2. "
        caption_raw = match[1]  # e.g. "Study area: ..."
        image_path_raw = match[2]  # e.g. "paper_artifacts\image_000000.png"
        image_path_raw = image_path_raw.replace("%5C", "/")

        figure_entry = {
            "figure_number": figure_number_raw.strip(),
            "caption_text": caption_raw.strip(),
            "image_path": image_path_raw.strip(),
        }
        figures.append(figure_entry)

    return figures


def create_image_document(image_path: str, **kwargs) -> ImageDocument:
    """Creates an ImageDocument from a local image file.

    Parameters
    ----------
    image_path : str
        The path to the image file.
    **kwargs
        Additional keyword arguments to pass to the ImageDocument.
        caption_text : str, optional
            The caption text for the image.
        metadata : dict, optional
            Any additional metadata to store.

    Returns
    -------
    ImageDocument
        The ImageDocument object.

    Examples
    --------
    ```python
    >>> from llama_utils.retrieval.pdf_reader import create_image_document
    >>> path = "examples/data/images/calibration.png"
    >>> caption = "Calibration framework of hydrological models."
    >>> doc = create_image_document(path, **{"caption_text": caption})
    >>> print(doc.doc_id)
    img-calibration.png
    >>> print(doc.metadata["filename"])
    calibration.png
    >>> print(doc.text)
    figure caption: Calibration framework of hydrological models.

    ```
    """
    image_path = Path(image_path)
    # base64 encoded image
    with open(image_path, "rb") as f:
        raw_image_data = f.read()
        im_base64 = base64.b64encode(raw_image_data).decode("utf-8")

    image_text = ""
    if "caption_text" in kwargs:
        caption_text = kwargs["caption_text"]
        image_text += f"figure caption: {caption_text}\n"

    # Create the ImageDocument.
    # 'text' is for any search-relevant text (OCR results, caption, etc.).
    # 'image' will hold the raw image data.
    doc = ImageDocument(
        id_=f"img-{image_path.name}",
        image=im_base64,
        text=image_text,
        image_path=str(image_path),
        metadata={"filename": image_path.name} | kwargs.get("metadata", {}),
    )
    return doc


def parse_pdf_with_docling(pdf_path: Path) -> tuple[Path, Path]:
    """Parse PDF with Docling and save as markdown with image references.

    Parameters
    ----------
    pdf_path : Path
        The path to the PDF file to parse.

    Examples
    --------
    ```python
    >>> from llama_utils.retrieval.pdf_reader import parse_pdf_with_docling
    >>> pdf_path = Path("examples/data/pdfs/geoscience-paper.pdf")
    >>> md_path, images_dir = parse_pdf_with_docling(pdf_path) # doctest: +SKIP
    Markdown file saved to examples/data/pdfs/geoscience-paper.md
    >>> print(images_dir) # doctest: +SKIP
    examples/data/pdfs/geoscience-paper_artifacts
    >>> print(list(images_dir.iterdir())) # doctest: +SKIP
    [PosixPath('examples/data/pdfs/geoscience-paper_artifacts/image_000000_xyz.png')]
    ```

    Note
    ----
    - The markdown file will be saved with the same name as the pdf file but with a `.md` extension.
    - The markdown file will contain image references to the local files.
    - The images are saved externally and referenced in the markdown file.
    - The images are saved in the same directory as the pdf in a subfolder named `<pdf-file-name>_artifacts`.
    - The images will have names like `image_000000_xyz.png`.
    """
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = doc_converter.convert(pdf_path)

    md_file = pdf_path.with_suffix(".md")
    result.document.save_as_markdown(md_file, image_mode=ImageRefMode.REFERENCED)
    print(f"Markdown file saved to {md_file}")

    ims_rdir = md_file.parent / f"{md_file.stem}{IMAGE_DIR_SUFFIX}"

    return md_file, ims_rdir


def get_image_docs_from_md(md_file: Path) -> list[ImageDocument]:
    r"""Get ImageDocuments from a Markdown file.

    Parameters
    ----------
    md_file : Path
        The path to the Markdown file.

    Returns
    -------
    list of ImageDocument
        The list of ImageDocument objects.

    Examples
    --------
    ```python
    >>> from llama_utils.retrieval.pdf_reader import get_image_docs_from_md
    >>> md_file = Path("examples/data/pdfs/geoscience-paper.md")
    >>> image_docs = get_image_docs_from_md(md_file)
    >>> print((image_docs)) # doctest: +SKIP
    [
        ImageDocument(
            id_='img-image_000000_0bb3fab8c73dc60d39d1aefd87fcffa8d95aa7ed8f67ac920355a00c50bb4456.png',
            embedding=None,
            metadata={
                'filename': 'image_000000_0bb3fab8c73dc60d39d1aefd87fcffa8d95aa7ed8f67ac920355a00c50bb4456.png'},
                excluded_embed_metadata_keys=[],
                excluded_llm_metadata_keys=[],
                relationships={},
                metadata_template='{key}: {value}',
                metadata_separator='\n',
                text_resource=MediaResource(
                    embeddings=None,
                    data=None,
                    text='figure caption: Two variants of raster based conceptual distributed models (of type 2): ...',
                    path=None,
                    url=None,
                    mimetype=None
                ),
                image_resource=MediaResource(
                    embeddings=None,
                    data=b'iVBORw0KGgoAAAANSUhEUgAAAtgAAAFSCAIAAABHcj9xAAEAAElEQVR4nOz9B5gcV3YmiF4TJl1571EACt47ggBJ0H...',
                    text=None,
                    path=None,
                    url=None,
                    mimetype='image/png'
                ),
                audio_resource=None,
                video_resource=None,
                text_template='{metadata_str}\n\n{content}'
        )
    ]

    ```
    """
    if not md_file.exists():
        raise FileNotFoundError(f"Markdown file not found at {md_file}")

    doc_text = md_file.read_text(encoding="utf-8")
    # read the images from the markdown file

    # extract the figure captions
    images_data = extract_figures_data(doc_text)

    image_docs = []
    for im_data in images_data:
        im_path = md_file.parent / im_data["image_path"]
        image_document = create_image_document(
            im_path, caption_text=im_data["caption_text"]
        )
        image_docs.append(image_document)
    return image_docs
