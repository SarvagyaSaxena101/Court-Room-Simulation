import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open('Bsc_Projects_Private_Limited_vs_Ircon_International_Limited_on_22_March_2023.PDF', 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
        return text
text = extract_text_from_pdf(pdf_path=None)
print(text)