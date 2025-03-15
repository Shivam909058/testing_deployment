import sys
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

def fix_torch_for_python312():
    """Apply fixes for PyTorch compatibility with Python 3.11+ and asyncio issues"""
    # Fix 1: Handle torch.__path__ for Python 3.12+ compatibility
    try:
        import torch
        
        # Create a path wrapper class to prevent errors with torch.__path__._path
        class PathFix:
            def __init__(self):
                self._path_value = []
                
            @property
            def _path(self):
                return self._path_value
                
            @_path.setter
            def _path(self, value):
                self._path_value = value

        # Apply the fix
        sys.modules['torch'].__path__ = PathFix()
        logger.info("Applied PyTorch path fix for Python 3.11+")
    except Exception as e:
        logger.warning(f"Failed to apply PyTorch path fix: {str(e)}")
        
    # Fix 2: Setup proper asyncio event loop
    try:
        # Initialize proper event loop
        if sys.platform == 'win32':
            # Windows specific settings
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Create new event loop if none exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        logger.info("Configured asyncio event loop")
    except Exception as e:
        logger.warning(f"Failed to configure asyncio event loop: {str(e)}")

def setup_event_loop():
    """Set up a proper event loop for async operations"""
    try:
        # Try to get the current event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there is no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop
    except Exception as e:
        logger.error(f"Failed to set up event loop: {str(e)}")
        return None

def run_async(coroutine):
    """Run an async function safely in any context"""
    loop = setup_event_loop()
    
    if loop and not loop.is_running():
        return loop.run_until_complete(coroutine)
    else:
        # Create a new loop for this specific task
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()
            # Restore the original loop
            if loop:
                asyncio.set_event_loop(loop) 