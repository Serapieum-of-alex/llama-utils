"""PDF Text Extraction Utilities."""

import re


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
