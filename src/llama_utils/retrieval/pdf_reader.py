"""PDF Text Extraction Utilities."""

import base64
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter as Docling_DocConverter
from docling.document_converter import PdfFormatOption
from docling_core.types.doc import ImageRefMode
from llama_index.core.schema import ImageDocument
from pydantic import BaseModel, Field

IMAGE_RESOLUTION_SCALE = 2.0
IMAGE_DIR_SUFFIX = "_artifacts"


class FigureData(BaseModel):
    """Model for extracted figure data."""

    figure_number: str = Field(..., description="Figure number and label.")
    caption_text: str = Field(..., description="Caption describing the figure.")
    image_path: Path = Field(..., description="Path to the extracted image file.")
    metadata: Dict[str, str] = Field(
        {}, description="Additional metadata for the figure."
    )


class ImageDocConfig(BaseModel):
    """Configuration for image document processing."""

    resolution_scale: float = Field(
        2.0, description="Scale factor for image resolution."
    )
    image_dir_suffix: str = Field(
        "_artifacts", description="Suffix for image directory."
    )


class DocumentConversionConfig(BaseModel):
    """Configuration for document conversion."""

    image_resolution_scale: float = Field(
        2.0, description="Scale factor for image resolution."
    )
    enable_page_images: bool = Field(
        True, description="Enable extraction of full page images."
    )
    enable_picture_images: bool = Field(
        True, description="Enable extraction of embedded figures."
    )


class DocumentConverter:
    """Handle document conversion, defaults to using docling's DocumentConverter.

    Parameters
    ----------
    converter : Optional[DocumentConverter], optional
        Custom document converter instance, by default None, which initializes a default DocumentConverter.

    Methods
    -------
    convert(pdf_path)
        Converts a PDF file to a Markdown file with extracted images.

    Examples
    --------
    ```python
    >>> converter = DocumentConverter()

    ```
    """

    def __init__(
        self,
        converter: Optional[Docling_DocConverter] = None,
        config: DocumentConversionConfig = None,
    ):
        """Initialize the DocumentConverter instance."""
        self.config = config or DocumentConversionConfig()
        if converter is None:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.images_scale = self.config.image_resolution_scale
            pipeline_options.generate_page_images = self.config.enable_page_images
            pipeline_options.generate_picture_images = self.config.enable_picture_images
            self.converter = Docling_DocConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        else:
            self.converter = converter

    def convert(self, pdf_path: Path) -> Tuple[Path, Path]:
        """Convert a PDF file to a Markdown file with extracted images as external reference in the file.

        Parameters
        ----------
        pdf_path : [Path]
            Path to the PDF file to be converted.

        Returns
        -------
        pdf_path: [Path]
            Path to the generated Markdown file.
        images_dir: [Path]
            Path to the directory containing the extracted images.

        Examples
        --------
        ```python
        >>> from llama_utils.retrieval.pdf_reader import DocumentConverter
        >>> pdf_path = Path("examples/data/pdfs/geoscience-paper.pdf")
        >>> converter = DocumentConverter()
        >>> markdown_file, images_dir = converter.convert(pdf_path)  # doctest: +SKIP
        >>> print(images_dir) # doctest: +SKIP
        examples/data/pdfs/geoscience-paper_artifacts
        >>> print(list(images_dir.iterdir())) # doctest: +SKIP
        [
            PosixPath('examples/data/pdfs/geoscience-paper_artifacts/image_000000_xyz.png'),
            PosixPath('examples/data/pdfs/geoscience-paper_artifacts/image_000001_xyz.png')
        ]

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

        result = self.converter.convert(pdf_path)
        md_file = pdf_path.with_suffix(".md")
        result.document.save_as_markdown(md_file, image_mode=ImageRefMode.REFERENCED)
        images_rdir = md_file.parent / f"{md_file.stem}{IMAGE_DIR_SUFFIX}"
        return md_file, images_rdir


class PDFReader:
    """Main class to handle PDF text extraction and image processing.

    Parameters
    ----------
    document_converter : Optional[DocumentConverterWrapper], optional
        A document converter instance to use for parsing PDFs, by default None which uses DocumentConverterWrapper.

    Methods
    -------
    extract_figures_data(pdf_text)
        Extracts figure captions and image references from a PDF text dump.

    create_image_document(image_path, caption_text)
        Creates an ImageDocument from an image file.

    parse_pdf(pdf_path)
        Parses the PDF, extracting images and generating markdown output.
    """

    def __init__(
        self,
        document_converter: Optional[DocumentConverter] = None,
        image_config: ImageDocConfig = None,
    ):
        """Initialize the PDFReader instance."""
        self.document_converter = document_converter or DocumentConverter()
        self.image_config = image_config or ImageDocConfig()

    @staticmethod
    def extract_figures_data(pdf_text: str) -> List[FigureData]:
        r"""Extract figure captions and image references from a PDF text.

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
        List[Dict[str, str]]
            A list of dictionaries containing figure numbers, captions, and image paths.
            Each dict contains:
            {
                "figure_number": "Figure 2.",
                "caption_text": "Study area: ...",
                "image_path": "paper_artifacts\\image_000000_xyz.png"
            }

        Examples
        --------
        ```python
        >>> from llama_utils.retrieval.pdf_reader import PDFReader
        >>> reader = PDFReader()
        >>> pdf_text = '''Some introduction text ...\n
        ...             Figure 2. Study area: The main campus ...
        ...             "![Image](paper_artifacts\\image_000000_ccc2c343.png)
        ...
        ...            "Some other random text ...
        ...
        ...             "Figure 3. Another figure's caption.
        ...             "![Image](paper_artifacts\\image_000001_abc123.png)
        ...             '''
        >>> figures_data = reader.extract_figures_data(pdf_text)
        >>> print(figures_data) # doctest: +SKIP
        [
            FigureData(
                figure_number='Figure 2.',
                caption_text='Study area: The main campus ...',
                image_path='paper_artifacts\\image_000000_ccc2c343.png'
            ),
            FigureData(
                figure_number='Figure 3.',
                caption_text="Another figure's caption.",
                image_path='paper_artifacts\\image_000001_abc123.png'
            )
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
            r"(Figure\s+\d+\.\s*)(.*?)\n?!\[Image\]\((.*?)\)", re.DOTALL
        )
        matches = pattern.findall(pdf_text)

        return [
            FigureData(
                figure_number=match[0].strip(),
                caption_text=match[1].strip(),
                image_path=match[2].replace("%5C", "/").strip(),
            )
            for match in matches
        ]

    @staticmethod
    def create_image_document(
        image_path: str, caption_text: Optional[str] = None, **kwargs
    ) -> ImageDocument:
        """Create an ImageDocument from an image file.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        caption_text : Optional[str], optional, default is None.
            The caption text associated with the image.
        **kwargs:
            Additional keyword arguments to pass to the ImageDocument.
            metadata : dict, optional
                Any additional metadata to store.

        Returns
        -------
        ImageDocument
            The ImageDocument object containing the image and metadata.

        Examples
        --------
        ```python
        >>> from llama_utils.retrieval.pdf_reader import PDFReader
        >>> image_path = "examples/data/images/calibration.png"
        >>> caption = "Calibration framework of hydrological models."
        >>> reader = PDFReader()
        >>> image_doc = reader.create_image_document(image_path, caption)
        >>> print(image_doc.doc_id)
        img-calibration.png
        >>> print(image_doc.metadata["filename"])
        calibration.png
        >>> print(image_doc.text)
        figure caption: Calibration framework of hydrological models.

        ```
        """
        image_path = Path(image_path)
        with open(image_path, "rb") as f:
            im_base64 = base64.b64encode(f.read()).decode("utf-8")

        return ImageDocument(
            id_=f"img-{image_path.name}",
            image=im_base64,
            text=f"figure caption: {caption_text}\n" if caption_text else "",
            image_path=str(image_path),
            metadata={"filename": image_path.name} | kwargs.get("metadata", {}),
        )

    def parse_pdf(
        self, pdf_path: Union[str, Path]
    ) -> Dict[str, Union[Path, List[ImageDocument]]]:
        r"""Parse the PDF, extracting images and generating markdown output.

        Parameters
        ----------
        pdf_path : Union[str, Path]
            Path to the PDF file to be processed.

        Returns
        -------
        Dict[str, Union[Path, List[ImageDocument]]]
            A dictionary containing the markdown file path and a list of extracted ImageDocument objects.

        Examples
        --------
        ```python
        >>> from llama_utils.retrieval.pdf_reader import PDFReader
        >>> pdf_path = Path("examples/data/pdfs/geoscience-paper.pdf")
        >>> reader = PDFReader()
        >>> result = reader.parse_pdf(pdf_path) # doctest: +SKIP
        >>> print(result.keys()) # doctest: +SKIP
        dict_keys(['markdown', 'images'])
        >>> print(result["markdown"]) # doctest: +SKIP
        examples/data/pdfs/geoscience-paper.md
        >>> print(result["images"]) # doctest: +SKIP
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
        pdf_path = Path(pdf_path)
        md_file, _ = self.document_converter.convert(pdf_path)
        md_text = md_file.read_text(encoding="utf-8")
        images_data = self.extract_figures_data(md_text)
        image_docs = [
            self.create_image_document(
                md_file.parent / img.image_path, img.caption_text
            )
            for img in images_data
        ]
        return {"markdown": md_file, "images": image_docs}
