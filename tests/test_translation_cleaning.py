"""Regression coverage for introductions emitted in saved Tower responses."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from screen_model import clean


class TranslationCleaningTests(unittest.TestCase):
    def test_tower_introductions_do_not_replace_the_translation(self):
        cases = [
            ("Translation in English:\nWhat are you asking this for?", "What are you asking this for?"),
            ("A possible translation in English is:\nWhere are you from?", "Where are you from?"),
            ("Translation in English:\n\nDid you call him yesterday?", "Did you call him yesterday?"),
            ("Sure, here's the English translation:\n\nI'll think about it.", "I'll think about it."),
            ("Here is the English translation: Hello.", "Hello."),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(clean(raw, "测试"), expected)

    def test_normal_translations_are_preserved(self):
        for text in ["English is difficult.", "Where are you from?", "Translation matters."]:
            with self.subTest(text=text):
                self.assertEqual(clean(text, "测试"), text)

    def test_existing_prefix_and_quote_handling(self):
        self.assertEqual(clean('Here\'s the English translation:\n\n"Hello!"', "你好！"), "Hello!")
        self.assertEqual(clean("译文：你好", "Hello"), "你好")
        self.assertEqual(clean("Translation in English:\n\n", "测试"), "")


if __name__ == "__main__":
    unittest.main()
