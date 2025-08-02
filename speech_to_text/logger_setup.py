"""Logging configuration for the Speech-to-Text application."""

import logging
from .config import AppConfig, DEFAULT_APP_CONFIG


def setup_logging(config: AppConfig = DEFAULT_APP_CONFIG) -> None:
    """Configure logging for the application."""
    
    # Basic logging configuration
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )
    
    # Set main logger level
    logger = logging.getLogger(__name__.split('.')[0])  # Get root module name
    logger.setLevel(level=logging.DEBUG if config.debug else logging.INFO)
    
    # Configure streamlit_webrtc logger
    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG if config.debug else logging.INFO)
    
    # Suppress noisy fsevents logger
    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)