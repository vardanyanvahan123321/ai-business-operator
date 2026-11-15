"""The deterministic rules behind the mock provider."""

from __future__ import annotations

from datetime import datetime

import pytest

from ai_operator.ai import rules
from ai_operator.ai.embeddings import content_overlap, hash_embed, stem, tokenize
from ai_operator.config import EMBEDDING_DIM
from ai_operator.core.message import Category

NOW = datetime(2026, 11, 16, 9, 0)  # a Monday


class TestDates:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("tomorrow at 10:00", datetime(2026, 11, 17, 10, 0)),
            ("завтра в 15:30", datetime(2026, 11, 17, 15, 30)),
            ("վաղը ժամը 12:00", datetime(2026, 11, 17, 12, 0)),
            ("послезавтра в 9", datetime(2026, 11, 18, 9, 0)),
            ("today at 3 pm", datetime(2026, 11, 16, 15, 0)),
            ("сегодня в 6 вечера", datetime(2026, 11, 16, 18, 0)),
            ("on Friday at 14:00", datetime(2026, 11, 20, 14, 0)),
            ("в пятницу", datetime(2026, 11, 20, 0, 0)),
            ("next monday 11:00", datetime(2026, 11, 23, 11, 0)),
            ("ուրբաթ օրը 10-ին", datetime(2026, 11, 20, 10, 0)),
            ("20.11 в 12:00", datetime(2026, 11, 20, 12, 0)),
            ("20/11/2026", datetime(2026, 11, 20, 0, 0)),
            ("05.01", datetime(2027, 1, 5, 0, 0)),  # already passed this year -> next year
            ("November 25 at 16:00", datetime(2026, 11, 25, 16, 0)),
            ("25 ноября", datetime(2026, 11, 25, 0, 0)),
            ("նոյեմբերի 25-ին", datetime(2026, 11, 25, 0, 0)),
            ("at 18:00", datetime(2026, 11, 16, 18, 0)),
            ("at 08:00", datetime(2026, 11, 17, 8, 0)),  # already passed today -> tomorrow
        ],
    )
    def test_parses(self, text: str, expected: datetime) -> None:
        assert rules.parse_datetime(text, NOW) == expected

    @pytest.mark.parametrize("text", ["How much is it?", "31.02", "цена 12000 драм", ""])
    def test_returns_none_without_a_date(self, text: str) -> None:
        assert rules.parse_datetime(text, NOW) is None


class TestExtractors:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Hi, my name is Anna Petrosyan.", "Anna Petrosyan"),
            ("I'm David, need a booking", "David"),
            ("Меня зовут Арам", "Арам"),
            ("Здравствуйте, имя: Мария", "Мария"),
            ("Անունս Գոռ է", "Գոռ"),
            ("Ես Հայկն եմ", "Հայկ"),
            ("I am Interested in diagnostics", None),
            ("сколько стоит?", None),
        ],
    )
    def test_names(self, text: str, expected: str | None) -> None:
        assert rules.find_name(text) == expected

    def test_phone_amount_and_service(self) -> None:
        assert rules.find_phone("тел. 091 12-34-56 после 6") == "091 12-34-56"
        assert rules.find_phone("в 10:00 15.11.2026") is None
        assert rules.find_amount("готов заплатить 15 000 драм") == 15000.0
        assert rules.find_amount("բյուջեն 20000֏") == 20000.0
        assert rules.find_amount("в 10 часов") is None
        keywords = {"oil_change": ("oil", "масл"), "brake_pads": ("brake", "тормоз", "колод")}
        assert rules.find_service("поменять тормозные колодки и масло", keywords) == "brake_pads"
        assert rules.find_service("nothing here", keywords) is None


class TestClassifier:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("How much is an oil change?", Category.QUESTION),
            ("Где вы находитесь?", Category.QUESTION),
            ("Ի՞նչ արժե ախտորոշումը", Category.QUESTION),
            ("Запишите меня на завтра", Category.ORDER),
            ("I'd like to book an appointment", Category.ORDER),
            ("Ուզում եմ գրանցվել", Category.ORDER),
            ("Ужасный сервис, верните деньги", Category.COMPLAINT),
            ("Terrible, I want a refund", Category.COMPLAINT),
            ("Դժգոհ եմ, մեքենաս փչացրել եք", Category.COMPLAINT),
            ("Earn money with crypto, click here https://x.example", Category.SPAM),
            ("Казино! Вы выиграли", Category.SPAM),
            ("Спасибо!", Category.OTHER),
        ],
    )
    def test_categories(self, text: str, expected: Category) -> None:
        category, confidence = rules.classify(
            text, has_pending_booking=False, has_booking_data=False
        )
        assert category is expected
        assert 0.5 <= confidence <= 0.97

    def test_ambiguous_or_empty_text_gets_low_confidence(self) -> None:
        _, confidence = rules.classify(
            "Тойота Камри", has_pending_booking=False, has_booking_data=False
        )
        assert confidence == 0.5
        _, mixed = rules.classify(
            "Можно ли записаться? Сколько стоит?", has_pending_booking=False, has_booking_data=False
        )
        assert mixed < 0.75  # a clear single-intent message scores 0.95

    def test_reply_with_details_to_a_pending_booking_is_an_order(self) -> None:
        category, confidence = rules.classify(
            "091 123456", has_pending_booking=True, has_booking_data=True
        )
        assert category is Category.ORDER
        assert confidence > 0.9


class TestEmbeddings:
    def test_vectors_are_unit_length_and_deterministic(self) -> None:
        first, second = hash_embed("замена масла"), hash_embed("замена масла")
        assert first == second
        assert len(first) == EMBEDDING_DIM
        assert abs(sum(v * v for v in first) - 1.0) < 1e-9

    def test_empty_text_still_gives_a_valid_vector(self) -> None:
        assert sum(v * v for v in hash_embed("")) == 1.0

    def test_related_texts_are_closer_than_unrelated(self) -> None:
        def cosine(a: str, b: str) -> float:
            return sum(x * y for x, y in zip(hash_embed(a), hash_embed(b), strict=True))

        question = "сколько стоит замена масла"
        assert cosine(question, "Замена масла стоит 12 000 драм") > cosine(
            question, "Мы работаем с понедельника по субботу"
        )

    def test_tokenizer_drops_stopwords_and_stems_inflections(self) -> None:
        assert tokenize("How much is THE oil change?") == ["much", "oil", "change"]
        assert stem("диагностика") == stem("диагностику")
        assert content_overlap("гарантия на работы", "На все работы даём гарантию 6 месяцев") == 1.0
        assert content_overlap("the a of", "anything") == 0.0
