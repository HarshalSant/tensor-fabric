import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

@pytest.fixture
def event_loop_policy():
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
