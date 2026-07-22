
from typing import Generic, TypeVar, Optional
from dataclasses import dataclass
T = TypeVar('T')

@dataclass
class SerpResult(Generic[T]):
    def __init__(self, data: Optional[T] = None, error: Optional[T] = None):
        self.data = data
        self.success = error is None
        self.error = error

    @property
    def is_success(self):
        return self.success
    def __repr__(self):
        return f'SerpResult(data={self.data!r}, error={self.error!r})'

