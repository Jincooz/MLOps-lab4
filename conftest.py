import sys
from unittest.mock import MagicMock

mock_s3_client = MagicMock()
mock_s3_client.put_object.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}

mock_boto3 = MagicMock()
mock_boto3.client.return_value = mock_s3_client

sys.modules["boto3"] = mock_boto3
