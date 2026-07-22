"""
modular search results page API configuration.
Allows configuration of various SERPER API clients, thorough multiple constructor methods e.g. .env file.

"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from loguru import logger
from .errors import SearchAPIException, SerperAPIException
load_dotenv()

@dataclass
class SerpConfig:
    """
    Configures the SERP API, which handles general configurations api_key, ...etc.
    Allowing multiple constructors of Serp configurations e.g., loading env variables.
    """
    def __init__(self, api_key:str,api_url:str="https://google.serper.dev/search",serp_api_client:str='SERPER',default_location:str = "us" , timeout:int = 10 ):
        self.api_key = api_key
        self.api_url = api_url
        self.serp_api_client = serp_api_client
        self.default_location = default_location
        self.timeout = timeout

    @classmethod
    def load_env_vars(cls,serp_api_client:str='SERPER')->'SerpConfig':
        logger.info('Loading SERP API Configuring from .env file')

        """
        Initialize configurations of SERPER from .env file.

        Arg:
            cls
        Returns:
            SerperConfig
        """
        print(serp_api_client+'_API_KEY')
        api_key = os.getenv(serp_api_client+'_API_KEY')
        if not api_key:
            raise SerperAPIException('⛔ SERPER_API_key not found in env file.')
        api_url = os.getenv(serp_api_client+'_API_URL')
        if not api_url:
            raise SerperAPIException('⛔ SERPER_API_URL not found in env file.')
        return cls(api_key = api_key,api_url= api_url)



