"""
Compatibility wrapper for legacy imports.

The OpenCV app was modularized into dedicated use-case modules with
`open_cv_landing.py` as the main entry point.
"""

from src.application_pages.open_cv.open_cv_landing import open_cv_landing_page


def opencv_detection_page():
    """Backward-compatible alias used by older app imports."""
    open_cv_landing_page()
