from typing import Dict

from ocr_space.wrapped_api import (
    OCRSpaceWrappedAPI,
    Language,
    FILE_TYPE,
)


class OCRSpace:
    """
    class for OCR space service
    """

    def __init__(self) -> None:
        self._api = OCRSpaceWrappedAPI()

    def ocr_pdf_file(
        self,
        file_url: str,
        language: str = Language.Arabic,
    ):
        response: Dict = self._api.ocr_url(
            file_url=file_url,
            file_type=FILE_TYPE.PDF,
            language=language,
            scale=True,
            is_create_searchable_pdf=False,
            is_searchable_hide_layer=False,
        )

        return response
