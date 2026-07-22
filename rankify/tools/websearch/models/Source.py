"""
Source model to retain the data of the retrieved documents, with runtime validation.
It captures various attributes of the sources that are needed to enhance the reranker.

"""

from pydantic import BaseModel
from dataclasses import dataclass
from typing import Optional


@dataclass
class Source(BaseModel):
    link:str
    html:str
    author:Optional[str] = None
    published_date:Optional[str] = None
    credibility_score:float = 0.0
    ref_len:int = 0
    html_content_len: int = 0

    def __repr__(self):
        return f"Source : ({self.link} \n{self.html} \n{self.author} \n{self.published_date} \n{self.credibility_score} \n{self.ref_len} \n{self.html_content_len} \n)"
