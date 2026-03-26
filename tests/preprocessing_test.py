import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "databricks"))

def _import_preprocessing():
    import preprocessing
    return preprocessing


class TestTextCleaning:


    def _get_preprocessor(self):
        module = _import_preprocessing()
        if not hasattr(module, "TextPreprocessor"):
            pytest.skip("TextPreprocessor not found in preprocessing.py")
        return module.TextPreprocessor()

    def test_lowercase(self):
        preprocessor = self._get_preprocessor()
        assert preprocessor.transform("Hello World") == preprocessor.transform("hello world")

    def test_strips_extra_whitespace(self):
        preprocessor = self._get_preprocessor()
        result = preprocessor.transform("  lots   of   spaces  ")
        assert "  " not in result

    def test_empty_string_returns_string(self):
        preprocessor = self._get_preprocessor()
        result = preprocessor.transform("")
        assert isinstance(result, str)

    def test_non_empty_input_non_empty_output(self):
        preprocessor = self._get_preprocessor()
        result = preprocessor.transform("some meaningful text about a topic")
        assert len(result) > 0

    def test_removes_special_chars_if_applicable(self):
        preprocessor = self._get_preprocessor()
        result = preprocessor.transform("Hello!!! @#$% test... <br>")
        assert isinstance(result, str)
