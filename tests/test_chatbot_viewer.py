import unittest

import chatbot_viewer as cv


class GeminiResponseParsingTests(unittest.TestCase):
    def test_parse_gemini_match_response(self) -> None:
        response_text = '''Here is the best match:\n```json\n{"match_index": 2, "matched_object_name": "computer mouse", "reason": "It is the closest semantic match to the user's request."}\n```'''

        parsed = cv.parse_gemini_match_response(response_text)

        self.assertEqual(parsed["match_index"], 2)
        self.assertEqual(parsed["matched_object_name"], "computer mouse")
        self.assertIn("closest semantic match", parsed["reason"])


if __name__ == "__main__":
    unittest.main()
