import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from loguru import logger
from rankify.tools.websearch.models.SerpResults import SerpResult

class SearchAPIClient(ABC):
    """
    Abstract Search API client.
    """
    @abstractmethod
    def search_web(self,query:str,num_results:int = 10 , file_path:Optional[str]=None)-> SerpResult[Dict[str, Any]]:
        """
        Search web page using SERPAPI client.
        """
        pass


