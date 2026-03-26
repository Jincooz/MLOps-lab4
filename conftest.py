import sys
from unittest.mock import MagicMock

sys.modules["boto3"] = MagicMock()
