"""Contract tests for classification validation and API integration."""

import json
import os
import unittest
from unittest.mock import Mock, patch

from src.classifier import (ClassificationError, MAX_TICKET_CHARS, RateLimitError, classify_ticket,
                            validate_result)


def result(**overrides):
    data = {"urgency": "medium", "category": "technical",
            "entities": {"customer_name": None, "product": None, "sentiment": "neutral"},
            "confidence": 0.92, "reasoning": "A technical issue is reported without immediate blocking impact."}
    data.update(overrides)
    return data


class SchemaTests(unittest.TestCase):
    def test_valid_contract_and_optional_entities(self):
        parsed = validate_result(result())
        self.assertIsNone(parsed["entities"]["customer_name"])
        self.assertIsNone(parsed["entities"]["product"])
        self.assertEqual(parsed["confidence"], 0.92)

    def test_accepts_json_fenced_response(self):
        self.assertEqual(validate_result("```json\n" + json.dumps(result()) + "\n```" )["category"], "technical")

    def test_missing_and_blank_optional_entities_normalize_to_null(self):
        value = result()
        del value["entities"]["customer_name"]
        value["entities"]["product"] = "  "
        parsed = validate_result(value)
        self.assertIsNone(parsed["entities"]["customer_name"])
        self.assertIsNone(parsed["entities"]["product"])

    def test_numeric_confidence_string_is_normalized(self):
        parsed = validate_result(result(confidence="0.8"))
        self.assertEqual(parsed["confidence"], 0.8)

    def test_rejects_missing_fields_and_drops_extra_fields(self):
        with self.assertRaises(ValueError):
            validate_result({"urgency": "low"})
        parsed = validate_result({**result(), "debug": "x"})
        self.assertEqual(set(parsed), {"urgency", "category", "entities", "confidence", "reasoning"})

    def test_rejects_bad_enums_and_confidence(self):
        for bad in ("urgent", 1.1, -0.1, True):
            value = result()
            if isinstance(bad, str):
                value["urgency"] = bad
            else:
                value["confidence"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_result(value)

    def test_rejects_invalid_entities_and_long_reasoning(self):
        value = result()
        value["entities"]["sentiment"] = "angry"
        with self.assertRaises(ValueError):
            validate_result(value)
        value = result(reasoning="word " * 21)
        with self.assertRaises(ValueError):
            validate_result(value)


class ProviderTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_credentials_are_controlled(self):
        with self.assertRaisesRegex(ClassificationError, "GEMINI_API_KEY"):
            classify_ticket("Our dashboard is slow.")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    def test_missing_model_is_controlled(self):
        with self.assertRaisesRegex(ClassificationError, "GEMINI_MODEL"):
            classify_ticket("Our dashboard is slow.")

    def test_input_validation(self):
        with self.assertRaisesRegex(ValueError, "Enter"):
            classify_ticket("   ")
        with self.assertRaisesRegex(ValueError, "characters"):
            classify_ticket("x" * (MAX_TICKET_CHARS + 1))
        with self.assertRaises(ValueError):
            classify_ticket(None)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-3.7-flash"}, clear=True)
    @patch("src.classifier.requests.post")
    def test_gemini_structured_output_request_and_response(self, post):
        response = Mock(ok=True)
        response.json.return_value = {"candidates": [{"content": {"parts": [{"text": json.dumps(result())}]}}]}
        post.return_value = response
        self.assertEqual(classify_ticket("The dashboard is slow.")["urgency"], "medium")
        self.assertEqual(post.call_args.args[0], "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent")
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "test-key")
        self.assertEqual(post.call_args.kwargs["timeout"], (5, 90))
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["generationConfig"]["responseFormat"]["text"]["mimeType"], "application/json")
        self.assertEqual(payload["generationConfig"]["responseFormat"]["text"]["schema"]["type"], "object")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-3.7-flash"}, clear=True)
    @patch("src.classifier.requests.post")
    def test_invalid_model_result_and_provider_failure_are_safe(self, post):
        response = Mock(ok=True)
        response.json.return_value = {"candidates": [{"content": {"parts": [{"text": '{"urgency":"high"}'}]}}]}
        post.return_value = response
        with self.assertRaisesRegex(ClassificationError, "invalid classification"):
            classify_ticket("Something is broken.")
        response.status_code = 500
        response.ok = False
        with self.assertRaisesRegex(ClassificationError, "could not process"):
            classify_ticket("Something is broken.")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-3.7-flash"}, clear=True)
    @patch("src.classifier.requests.post")
    def test_gemini_rate_limit_has_safe_message(self, post):
        post.return_value = Mock(status_code=429, ok=False)
        with self.assertRaisesRegex(RateLimitError, "quota or rate limit"):
            classify_ticket("Our production account is blocked.")

if __name__ == "__main__":
    unittest.main()
