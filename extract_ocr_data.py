import os
from ocr_space import OCRSpace


OCR_DATA_FILES = [
    "https://qa-tby-cdn.s3.eu-west-3.amazonaws.com/ocr+data/06-20-2023.pdf",

]
OUTPUT_DIR = "ocr_data"

for file in OCR_DATA_FILES:
    ocr_space = OCRSpace()
    pages = ocr_space.ocr_pdf_file(file_url=file)
    document_dir_name = file.split("/")[-1].split(".")[0]
    os.mkdir(f"{OUTPUT_DIR}/{document_dir_name}")
    for page_number, page in enumerate(pages):
        with open(f"{OUTPUT_DIR}/{document_dir_name}/page_{page_number+1}.txt","w") as page_file:
            page_file.write(page["ParsedText"])
