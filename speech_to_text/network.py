"""Network configuration and WebRTC utilities."""

import logging
from typing import List, Dict, Any

import streamlit as st
from twilio.rest import Client

from .config import TwilioConfig, DEFAULT_TWILIO_CONFIG

logger = logging.getLogger(__name__)


class NetworkManager:
    """Manages network configuration for WebRTC connections."""
    
    def __init__(self, twilio_config: TwilioConfig = DEFAULT_TWILIO_CONFIG):
        self.twilio_config = twilio_config
    
    @st.cache_data
    def get_ice_servers(_self) -> List[Dict[str, Any]]:
        """
        Get ICE servers for WebRTC connection.
        Uses Twilio's TURN server if configured, otherwise falls back to Google's STUN server.
        """
        if not _self.twilio_config.is_configured:
            logger.warning(
                "Twilio credentials are not set. Fallback to a free STUN server from Google."
            )
            return [{"urls": ["stun:stun.l.google.com:19302"]}]
        
        try:
            client = Client(_self.twilio_config.account_sid, _self.twilio_config.auth_token)
            token = client.tokens.create()
            return token.ice_servers
        except Exception as e:
            logger.error(f"Failed to get Twilio ICE servers: {e}")
            return [{"urls": ["stun:stun.l.google.com:19302"]}]
    
    def get_rtc_configuration(self) -> Dict[str, Any]:
        """Get RTC configuration for WebRTC connection."""
        return {"iceServers": self.get_ice_servers()}