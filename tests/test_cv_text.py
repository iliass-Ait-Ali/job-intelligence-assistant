import pytest

pytest.importorskip("pypdf")

from pdf_helpers import blank_pdf, encrypt_pdf, make_pdf  # noqa: E402

from job_intel.match import cv_text  # noqa: E402
from job_intel.match.cv_text import CVExtractionError, clean_cv_text, from_file, from_pasted_text, from_pdf_bytes  # noqa: E402

LINES = ["Jane Sample", "EDUCATION", "Master in Data Science - Example Institute, 2024 - present",
         "EXPERIENCE", "Data Science Intern - Example Retail, June 2024 - August 2024",
         "Built a forecasting model with Python and scikit-learn for weekly sales.",
         "SKILLS", "Python, SQL, pandas, PyTorch"] * 3  # fmt: skip


def test_pdf_text_is_extracted_with_accents():
    data = make_pdf([["Ingénieur Data - Exemple Logistique", "Compétences : Python, SQL, Spark, Airflow " * 3] + LINES])
    cv = from_pdf_bytes(data)
    assert cv.source == "pdf" and cv.pages == 1
    assert "Ingénieur Data" in cv.text and "Compétences" in cv.text and "Python" in cv.text
    assert not cv.warnings


def test_multi_page_pdf_keeps_all_pages():
    cv = from_pdf_bytes(make_pdf([LINES, ["PROJECTS", "Movie recommender with NumPy " * 6], ["LANGUAGES", "French (native)"] * 10]))
    assert cv.pages == 3 and "Movie recommender" in cv.text and "French (native)" in cv.text


def test_scanned_pdf_without_text_is_a_clear_error():
    with pytest.raises(CVExtractionError, match="scan|OCR"):
        from_pdf_bytes(blank_pdf(2))


def test_mostly_scanned_pdf_gets_a_warning_not_a_silent_pass():
    data = make_pdf([["Jane Sample", "Python, SQL", "tiny"]])  # 1 page, almost no text
    cv = from_pdf_bytes(data)
    assert any("scanned" in w.lower() or "ocr" in w.lower() for w in cv.warnings)
    assert any("characters" in w for w in cv.warnings)


def test_partly_scanned_pdf_reports_empty_pages():
    from io import BytesIO

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for page in PdfReader(BytesIO(make_pdf([LINES * 3]))).pages:
        writer.add_page(page)
    writer.add_blank_page(width=595, height=842)
    buffer = BytesIO()
    writer.write(buffer)
    cv = from_pdf_bytes(buffer.getvalue())
    assert any("1 of 2 pages" in w for w in cv.warnings)


def test_encrypted_pdf_is_reported():
    with pytest.raises(CVExtractionError, match="password"):
        from_pdf_bytes(encrypt_pdf(make_pdf([LINES])))


@pytest.mark.parametrize("data", [b"", b"not a pdf at all", b"%PDF-1.4\ngarbage that is not a real pdf", b"%PDF-" + b"\x00" * 50])
def test_corrupt_or_empty_pdf_never_raises_unexpected_errors(data):
    with pytest.raises(CVExtractionError):
        from_pdf_bytes(data)


def test_oversized_pdf_is_rejected():
    with pytest.raises(CVExtractionError, match="larger"):
        from_pdf_bytes(b"%PDF-1.4\n" + b"0" * (cv_text.MAX_BYTES + 1))


def test_pdf_page_limit_warns():
    cv = from_pdf_bytes(make_pdf([LINES] * (cv_text.MAX_PAGES + 2)))
    assert any("only the first" in w for w in cv.warnings)


def test_text_and_markdown_files(tmp_path):
    f = tmp_path / "cv.txt"
    f.write_text("EXPERIENCE\nData Analyst - Acme, 2022 - 2023\n- Built dashboards with Power BI\n" * 5, encoding="utf-8")
    assert from_file(f).source == "text"
    legacy = tmp_path / "old.txt"
    legacy.write_bytes("Compétences : Python, é à ü ".encode("cp1252") * 20)
    assert "Compétences" in from_file(legacy).text  # falls back from UTF-8 to cp1252


def test_unsupported_missing_and_empty_files(tmp_path):
    docx = tmp_path / "cv.docx"
    docx.write_bytes(b"PK")  # not a real Word document
    with pytest.raises(CVExtractionError, match="could not be read"):
        from_file(docx)
    odt = tmp_path / "cv.odt"
    odt.write_bytes(b"PK")
    with pytest.raises(CVExtractionError, match="Unsupported"):
        from_file(odt)
    with pytest.raises(CVExtractionError, match="not be found"):
        from_file(tmp_path / "nope.pdf")
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    with pytest.raises(CVExtractionError):
        from_file(empty)


@pytest.mark.parametrize("text", [None, "", "   \n\t  "])
def test_empty_pasted_text_is_rejected(text):
    with pytest.raises(CVExtractionError, match="empty"):
        from_pasted_text(text)


def test_short_text_warns_and_very_long_text_is_truncated_with_a_warning():
    assert any("characters" in w for w in from_pasted_text("Python developer").warnings)
    long = from_pasted_text("Python and SQL experience. " * 5000)
    assert len(long.text) <= cv_text.MAX_CHARS and any("very long" in w for w in long.warnings)


def test_clean_cv_text_normalizes_without_adding_content():
    raw = "data-\nscientist​  with\tPython\r\n\r\n\r\n\r\n•  Built models\n●   Deployed APIs\xa0today"
    cleaned = clean_cv_text(raw)
    assert cleaned == "data-scientist with Python\n\n- Built models\n- Deployed APIs today"
    assert clean_cv_text("\x00\x07Python\x1b") == "Python"
