"""PDF Text Extraction Utilities."""

import base64
import re
from pathlib import Path

from llama_index.core.schema import ImageDocument


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
