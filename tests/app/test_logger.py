import pytest
import logging
from app.logger import logger

@pytest.mark.parametrize("log_level, log_message", [
    (logging.INFO, "This is an info message"),
    (logging.WARNING, "This is a warning message"),
    (logging.ERROR, "This is an error message"),
])
def test_logger(caplog, log_level, log_message):
    """
    Test that the logger logs messages at different levels correctly.
    """
    with caplog.at_level(log_level):
        logger.log(log_level, log_message)
    
    # Ensure that exactly one log message was captured
    assert len(caplog.records) == 1

    # Ensure the logged message is what we expect
    assert caplog.records[0].levelno == log_level
    assert caplog.records[0].message == log_message