import math
import re
from collections import Counter

ru_vowels = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


def count_syllables_ru(word: str) -> int:
    return max(1, sum(1 for ch in word if ch in ru_vowels))


def count_syllables_en(word: str) -> int:
    word = word.lower()
    if word.endswith("e") and re.search(r"[aeiouy]", word[:-1]):
        word = word[:-1]
    return max(1, len(re.findall(r"[aeiouy]+", word)))


def count_syllables(word: str, lang: str) -> int:
    return count_syllables_ru(word) if lang == "ru" else count_syllables_en(word)


def avg_sentence_length(n_words: int, n_sents: int) -> float:
    return n_words / n_sents if n_sents else 0.0


def avg_word_syllables(syl_counts: list[int]) -> float:
    return sum(syl_counts) / len(syl_counts) if syl_counts else 0.0


def long_word_ratio(syl_counts: list[int], lang: str) -> float:
    if not syl_counts:
        return 0.0
    threshold = 4 if lang == "ru" else 3
    return sum(1 for s in syl_counts if s >= threshold) / len(syl_counts)


def flesch_kincaid_grade(asl: float, asw: float, lang: str) -> float:
    if lang == "ru":
        return 0.5 * asl + 8.4 * asw - 15.59
    return 0.39 * asl + 11.8 * asw - 15.59


def smog_grade(n_long: int, n_sents: int, lang: str) -> float:
    if n_sents == 0:
        return 0.0
    if lang == "ru":
        return 1.1 * math.sqrt((64.6 / n_sents) * n_long) + 0.05
    return 1.0430 * math.sqrt(n_long * 30 / n_sents) + 3.1291


def type_token_ratio(words: list[str]) -> float:
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def yules_k(words: list[str]) -> float:
    if not words:
        return 0.0
    n = len(words)
    freq = Counter(words)
    freq_of_freq = Counter(freq.values())
    sum_vx = sum(v * (x ** 2) for x, v in freq_of_freq.items())
    return 1e4 * (sum_vx - n) / (n ** 2)
