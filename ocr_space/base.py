from abc import ABC, abstractmethod
from typing import Dict


class WrappedAPIBase(ABC):
    """
    Wrapper API interface for OCR space
    """

    @abstractmethod
    def ocr_url(
        self,
        file_url: str,
        language: str,
        is_overlay_required: bool,
        file_type: str,
        detect_orientation: bool,
        is_create_searchable_pdf: bool,
        is_searchable_hide_layer: bool,
        scale: bool,
        is_table: bool,
        ocr_engine: int,
    ) -> Dict:
        """
        POST a file at a given URL to process with OCR.
        :param file_url: URL of remote image or pdf file
        :param language: Language used for OCR. If no language is specified, English eng is taken as default.
        :param is_overlay_required : Default = False If true, returns the coordinates of the bounding boxes for each word.
        :param file_type : Overwrites the automatic file type detection based on content-type.
        :param detect_orientation : If set to true, the api autorotates the image correctly and sets the TextOrientation parameter in the JSON response.
        :param is_create_searchable_pdf : Default = False If true, API generates a searchable PDF.
            This parameter automatically sets isOverlayRequired = true.
        :param is_searchable_hide_layer	: Default = False. If true, the text layer is hidden (not visible)
        :param scale : If set to true, the api does some internal upscaling(good for scanned images with low dpi).
        :param is_table : If set to true, the OCR logic makes sure that the parsed text result is always returned line by line.
        :param ocr_engine: Engine 1 is default.
        :return: Result in JSON format.
        """
