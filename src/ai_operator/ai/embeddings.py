"""Local, dependency-free text embeddings (feature hashing).

Used when no embedding API is available: the ``mock`` provider (demo, CI) and
``anthropic`` (which has no embeddings endpoint). It is *lexical*, not semantic:
words and character n-grams are hashed into a fixed-size vector, so texts that
share vocabulary end up close. That is enough for a demo knowledge base and it
keeps the whole pipeline - including pgvector search - real without any key.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from ai_operator.config import EMBEDDING_DIM

LOCAL_EMBEDDING_MODEL = "local-hash-v1"

_WORD = re.compile(r"[^\W\d_]{2,}|\d+", re.UNICODE)

# Function words carry no topic. Small on purpose: a few dozen per language.
STOPWORDS = frozenset(
    """
    the a an and or but if of to in on at for with from by is are was were be been do does did
    can could would should will you your we our us it its this that these those what which who
    how when where why have has had not no yes please hi hello thanks thank me my i am as so и в
    во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне
    было вот от меня еще нет о из ему теперь когда даже ну ли если уже или ни быть был до вас
    вам ваш ваши наш это этот эта эти мы они для при есть ли здравствуйте привет спасибо
    пожалуйста можно скажите подскажите և է են եմ ես դու նա մենք դուք նրանք որ ու կամ բայց եթե
    այս այդ այն իմ քո մեր ձեր իր համար հետ մեջ վրա մոտ ինչ ինչպես երբ որտեղ ինչու թե էլ էր էին
    չէ այո ոչ բարև ողջույն շնորհակալություն խնդրում կարող կարելի ունեք ունի ունեմ կա կան լինի
    """.split()  # noqa: SIM905
)


def tokenize(text: str) -> list[str]:
    """Lower-cased content words (stopwords removed)."""
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def stem(word: str) -> str:
    """Crude prefix stemmer: good enough to match inflected RU/HY/EN word forms."""
    return word if len(word) <= 5 else word[:5]


def _features(text: str) -> Counter[str]:
    feats: Counter[str] = Counter()
    for word in tokenize(text):
        feats[f"w:{stem(word)}"] += 2
        padded = f"<{word}>"
        for i in range(len(padded) - 2):
            feats[f"g:{padded[i : i + 3]}"] += 1
    return feats


def _bucket(feature: str) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % EMBEDDING_DIM, 1.0 if (value >> 63) & 1 else -1.0


def hash_embed(text: str) -> list[float]:
    """Deterministic L2-normalised embedding of ``EMBEDDING_DIM`` floats."""
    vector = [0.0] * EMBEDDING_DIM
    for feature, count in _features(text).items():
        index, sign = _bucket(feature)
        vector[index] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        vector[0] = 1.0  # pgvector cannot take cosine distance of a zero vector
        return vector
    return [v / norm for v in vector]


def content_overlap(question: str, passage: str) -> float:
    """Share of the question's content words that also occur in the passage (0..1)."""
    q = {stem(w) for w in tokenize(question)}
    if not q:
        return 0.0
    p = {stem(w) for w in tokenize(passage)}
    return len(q & p) / len(q)
