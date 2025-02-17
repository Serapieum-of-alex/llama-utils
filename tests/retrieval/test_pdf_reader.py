from llama_utils.retrieval.pdf_reader import extract_figures_data


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
