import pytest
from unittest.mock import AsyncMock

from app.egress import Egress

@pytest.mark.asyncio
async def test_send_results(mocker):
    # Patch the RedisClient.send_pipeline reference in egress.py
    mock_send_pipeline = mocker.patch("app.egress.RedisClient.send_pipeline", new_callable=AsyncMock)
    
    outputs = ["some", "results"]
    keyname = "test_key"
    
    # Call the method under test
    await Egress.send_results(outputs, keyname)
    
    # Verify RedisClient.send_pipeline was awaited once with the correct args
    mock_send_pipeline.assert_awaited_once_with(outputs, keyname)
