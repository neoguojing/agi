import unittest
from unittest.mock import patch

from agi.llms.chat_model import create_chat_model


class TestCreateChatModel(unittest.TestCase):
    @patch("agi.llms.chat_model.init_chat_model")
    def test_create_chat_model_passes_standard_params(self, mock_init):
        mock_init.return_value = object()

        result = create_chat_model(
            model="gpt-5.2",
            model_provider="openai",
            temperature=0.2,
            max_tokens=512,
            timeout=30,
            max_retries=8,
            api_key="test-key",
        )

        self.assertIs(result, mock_init.return_value)
        mock_init.assert_called_once_with(
            model="gpt-5.2",
            model_provider="openai",
            temperature=0.2,
            max_tokens=512,
            timeout=30,
            max_retries=8,
            api_key="test-key",
        )


if __name__ == "__main__":
    unittest.main()
