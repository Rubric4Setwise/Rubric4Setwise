from typing import Dict, Any

from loguru import logger

from rankify.tools.websearch.models.SerpResults import SerpResult


class SearchAPIException(Exception):
    """
    Exception raised for errors in search API.
    """
    def __init__(self, error):
        logger.exception(str(error))
        super().__init__(error)
    pass

class SerperAPIException(SearchAPIException):
    """
    Exception raised for errors in serper API client.
    """
    def __init__(self, error: object) -> None:
        logger.exception(str(error))
        super().__init__(error)
    pass


