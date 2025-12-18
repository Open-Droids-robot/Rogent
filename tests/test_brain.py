import pytest
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from agent import Brain

@pytest.mark.asyncio
async def test_brain_move_head():
    brain = Brain()
    decision = await brain.process("Please move head up", [])
    assert decision["action"] == "tool"
    assert decision["tool_name"] == "move_head"
    assert decision["arguments"] == {"pan": 10, "tilt": 20}

@pytest.mark.asyncio
async def test_brain_camera():
    brain = Brain()
    decision = await brain.process("Take a camera image", [])
    assert decision["action"] == "tool"
    assert decision["tool_name"] == "get_camera_image"

@pytest.mark.asyncio
async def test_brain_speak():
    brain = Brain()
    decision = await brain.process("Hello robot", [])
    assert decision["action"] == "speak"
    assert "Hello robot" in decision["content"]
