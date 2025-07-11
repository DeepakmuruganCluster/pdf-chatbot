import fitz  # PyMuPDF

def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")  # type: ignore
        doc.close()
        return text
    except Exception as e:
        return f"Error loading PDF: {str(e)}"
