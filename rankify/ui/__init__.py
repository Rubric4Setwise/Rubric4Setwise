"""
Rankify UI - Interactive Playground

Launch:
    >>> from rankify.ui import launch_playground
    >>> launch_playground(port=7860)
"""

from rankify.ui.playground import (
    launch_playground,
    create_playground_app,
)

__all__ = [
    "launch_playground",
    "create_playground_app",
]
