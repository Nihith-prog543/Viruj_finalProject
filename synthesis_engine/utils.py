# synthesis_engine/utils.py - Utility functions
import uuid
from datetime import datetime
import json

# Global session storage (in production, use Redis or database)
session_storage = {}

def initialize_session(api_name, session_id=None):
    """Initialize a new analysis session or re-initialize an existing one"""
    if session_id is None:
        session_id = str(uuid.uuid4())
    session_storage[session_id] = {
        'api_name': api_name,
        'created_at': datetime.now().isoformat(),
        'analysis_complete': False,
        'chat_history': [],
        'results': None,
        'ai_predicted_route': None, # Initialize predicted route as well
        'prediction_complete': False
    }
    return session_id

def get_session_data(session_id):
    """Get session data by ID"""
    return session_storage.get(session_id)

def update_session_data(session_id, data):
    """Update session data"""
    if session_id in session_storage:
        session_storage[session_id].update(data)
        return True
    return False

def add_chat_message(session_id, user_message, bot_response):
    """Add chat exchange to session"""
    if session_id in session_storage:
        session_storage[session_id]['chat_history'].append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': datetime.now().isoformat()
        })
        return True
    return False

# synthesis_engine/__init__.py
"""
Synthesis Route Finder Engine
"""

from .analysis import SynthesisAnalyzer
from .utils import initialize_session, get_session_data, update_session_data

__version__ = "1.0.0"