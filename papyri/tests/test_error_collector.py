from papyri.gen import ErrorCollector
import logging
import pytest
from papyri.errors import UnseenError

log = logging.getLogger(__name__)


class Config:
    pass


def JustPasses():
    pass


def DoesValueError():
    raise ValueError("A")


def ShouldValueErrorTypeError():
    raise TypeError("B")


def test_capture_correct():
    c = Config()
    c.expected_errors = {"ValueError": ["TestIterm"]}
    c.early_error = False
    c.fail_unseen_error = True
    ec = ErrorCollector(c, log)

    with ec("TestItem"):
        DoesValueError()

    assert ec._errors == {"ValueError": ["TestItem"]}


def test_pass_no_collect():
    c = Config()
    c.expected_errors = {}
    c.early_error = True
    c.fail_unseen_error = True
    ec = ErrorCollector(c, log)

    with ec("TestItem"):
        JustPasses()


def test_2():
    c = Config()
    c.expected_errors = {"ValueError": ["TestItem"]}
    c.early_error = True
    c.fail_unseen_error = True
    ec = ErrorCollector(c, log)
    with pytest.raises(UnseenError):
        with ec("TestItem"):
            JustPasses()


def test_4():
    c = Config()
    c.expected_errors = {"ValueError": ["TestItem"]}
    c.early_error = False
    c.fail_unseen_error = True
    ec = ErrorCollector(c, log)

    with ec("TestItem"):
        ShouldValueErrorTypeError()
