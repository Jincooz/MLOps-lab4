import sys
from unittest.mock import MagicMock
import types

try:
    import boto3 
except ModuleNotFoundError:
    fake_boto3 = types.SimpleNamespace()

    def fake_client(*args, **kwargs):
        raise RuntimeError("boto3 is not installed (mocked in tests)")

    fake_boto3.client = fake_client
    sys.modules["boto3"] = fake_boto3

sys.modules["prometheus_client"] = MagicMock()
