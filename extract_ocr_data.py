import os
from ocr_space import OCRSpace


OCR_DATA_FILES = {
    "الرابع عشر (مجلس 2013)/الرابع/1351 أ/26-04-2016.pdf" : "https://qa-tby-cdn.s3.eu-west-3.amazonaws.com/ocr+data/26-04-2016.pdf",

}
OUTPUT_DIR = "ocr_data"

for key,file in OCR_DATA_FILES.items():
    ocr_space = OCRSpace()
    pages = ocr_space.ocr_pdf_file(file_url=file)
    document_dir_name = key.split(".")[0]
    os.makedirs(f"{OUTPUT_DIR}/{document_dir_name}")
    for page_number, page in enumerate(pages):
        with open(f"{OUTPUT_DIR}/{document_dir_name}/page_{page_number+1}.txt","w") as page_file:
            page_file.write(page["ParsedText"])
