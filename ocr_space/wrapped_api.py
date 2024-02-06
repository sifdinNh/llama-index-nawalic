import requests
import logging


from typing import Dict
from requests.exceptions import HTTPError


from ocr_space.base import WrappedAPIBase
from requests_toolbelt.multipart.encoder import MultipartEncoder


class BaseIntegrationException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message


class BaseHTTPErrorException(BaseIntegrationException):
    code = ""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"HTTP_{ self.code}: { self.message}"

logger = logging.getLogger(__name__)


class PayloadError(BaseIntegrationException):
    pass


class BadRequestError(BaseHTTPErrorException):
    code = "4xx"


class InternalServerError(BaseHTTPErrorException):
    code = "5xx"


class Language:
    Arabic = "ara"
    English = "eng"


class FILE_TYPE:
    PDF = "pdf"
    GIF = "gif"
    PNG = "png"
    JPG = "jpg"
    TIF = "tif"
    BMP = "bmp"


class OCRSpaceWrappedAPI(WrappedAPIBase):
    """
    Wrapper API for ocr space service
    """

    OCR_SPACE_API_ENDPOINT = "https://apipro1.ocr.space/parse/image"
    OCR_SPACE_API_KEY = "PD8L7K0AZ2RX"

    def _parse(self, response, raise_error: bool = True):

        try:
            response.raise_for_status()
            data = response.json()
            if data["IsErroredOnProcessing"]:
                message = data["ErrorMessage"][0]
                raise PayloadError(message=message)
            return data["ParsedResults"]
        except HTTPError as e:
            if raise_error:
                try:
                    message = e.response.json()["Message"]  # get error message
                except:
                    message = str(e.response)
                if 400 <= e.response.status_code < 500:
                    raise BadRequestError(message=message)
                elif 500 <= e.response.status_code < 600:
                    raise InternalServerError(message=message)

    def ocr_url(
        self,
        file_url: str,
        language: str = Language.Arabic,
        is_overlay_required: bool = False,
        file_type: str = FILE_TYPE.PDF,
        detect_orientation: bool = False,
        is_create_searchable_pdf: bool = False,
        is_searchable_hide_layer: bool = False,
        scale: bool = False,
        is_table: bool = False,
        ocr_engine: int = 1,
    ) -> Dict:
        encoder = MultipartEncoder(
            fields={
                "url": file_url,  # This line seems incorrect
                "language": language,
                "isOverlayRequired": str(is_overlay_required).lower(),
                "filetype": file_type,
                "detectOrientation": str(detect_orientation).lower(),
                "isCreateSearchablePdf": str(is_create_searchable_pdf).lower(),
                "isSearchablePdfHideTextLayer": str(is_searchable_hide_layer).lower(),
                "scale": str(scale).lower(),
                "isTable": str(is_table).lower(),
                "OCREngine": str(ocr_engine),
            }
        )
        headers = {
            "Content-Type": encoder.content_type,
            "Accept": "application/json",
            "apikey": self.OCR_SPACE_API_KEY,
        }
        try:
            r = requests.post(
                self.OCR_SPACE_API_ENDPOINT,
                data=encoder,
                headers=headers,
            )
            return self._parse(r)
        except requests.exceptions.RequestException as e:
            # TODO raise this exception in celery
            logger.error(f"Error {str(e)}")
            raise e
