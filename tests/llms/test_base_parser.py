import unittest

from langchain_core.messages import HumanMessage

from agi.llms.base import parse_input_messages


class TestParseInputMessages(unittest.TestCase):
    def test_parse_openai_image_url(self):
        media, prompt, input_type = parse_input_messages(
            HumanMessage(
                content=[
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                ]
            )
        )
        self.assertEqual(media, "https://example.com/a.png")
        self.assertEqual(prompt, "describe this")
        self.assertEqual(input_type, "image")

    def test_parse_legacy_audio(self):
        media, prompt, input_type = parse_input_messages(
            HumanMessage(content=[{"type": "audio", "audio": "base64..."}])
        )
        self.assertEqual(media, "base64...")
        self.assertIsNone(prompt)
        self.assertEqual(input_type, "audio")


if __name__ == "__main__":
    unittest.main()
