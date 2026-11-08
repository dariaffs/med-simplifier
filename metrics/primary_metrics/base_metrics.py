import math
import re
from collections import Counter

ru_vowels = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


# Syllable counters

def count_syllables_ru(word: str) -> int:
    return max(1, sum(1 for ch in word if ch in ru_vowels))


def count_syllables_en(word: str) -> int:
    word = word.lower()
    if word.endswith("e") and re.search(r"[aeiouy]", word[:-1]):
        word = word[:-1]
    return max(1, len(re.findall(r"[aeiouy]+", word)))


def count_syllables(word: str, lang: str) -> int:
    return count_syllables_ru(word) if lang == "ru" else count_syllables_en(word)


# Surface statistics

def avg_word_chars(n_letters: int, n_words: int) -> float:
    return n_letters / n_words if n_words else 0.0


def letters_per_100_words(n_letters: int, n_words: int) -> float:
    return 100 * n_letters / n_words if n_words else 0.0


def sents_per_100_words(n_sents: int, n_words: int) -> float:
    return 100 * n_sents / n_words if n_words else 0.0


def comma_ratio(n_commas: int, n_words: int) -> float:
    return n_commas / n_words if n_words else 0.0


# Length metrics

def avg_sentence_length(n_words: int, n_sents: int) -> float:
    return n_words / n_sents if n_sents else 0.0


def avg_word_syllables(syl_counts: list[int]) -> float:
    return sum(syl_counts) / len(syl_counts) if syl_counts else 0.0


def avg_sent_syllables(n_syl: int, n_sents: int) -> float:
    return n_syl / n_sents if n_sents else 0.0


def long_word_ratio(syl_counts: list[int], lang: str) -> float:
    if not syl_counts:
        return 0.0
    threshold = 4 if lang == "ru" else 3
    return sum(1 for s in syl_counts if s >= threshold) / len(syl_counts)


# Readability indices

def flesch_reading_ease(asl: float, asw: float) -> float:
    return 206.835 - 1.015 * asl - 84.6 * asw


def oborneva_index(asl: float, asw: float) -> float:
    return 206.835 - 1.52 * asl - 65.14 * asw


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


def gunning_fog(asl: float, complex_word_ratio: float) -> float:
    return 0.4 * (asl + 100 * complex_word_ratio)


def lix(n_long_char: int, n_words: int, n_sents: int) -> float:
    if not n_words or not n_sents:
        return 0.0
    return n_words / n_sents + 100 * n_long_char / n_words


def rix(n_long_char: int, n_sents: int) -> float:
    return n_long_char / n_sents if n_sents else 0.0


def ovix(n_words: int, n_unique: int) -> float:
    import math
    if n_words < 2 or n_unique < 2:
        return 0.0
    ln_n = math.log(n_words)
    ln_v = math.log(n_unique)
    denom = 2 - ln_v / ln_n
    return ln_n / denom if denom else 0.0


def ari(n_chars: int, n_words: int, n_sents: int, lang: str) -> float:
    if not n_words or not n_sents:
        return 0.0
    if lang == "ru":
        return 6.26 * (n_chars / n_words) + 0.2805 * (n_words / n_sents) - 31.04
    return 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sents) - 21.43


def dale_chall(word_long: int, n_words: int, n_sents: int, lang: str) -> float:
    if not n_words or not n_sents:
        return 0.0
    if lang == "ru":
        return 0.552 * (100.0 * word_long / n_words) + 0.273 * (n_words / n_sents)
    return 0.1579 * (100.0 * word_long / n_words) + 0.0496 * (n_words / n_sents)


def coleman_liau(n_letters: int, n_words: int, n_sents: int, lang: str) -> float:
    if not n_words:
        return 0.0
    L = 100 * n_letters / n_words
    S = 100 * n_sents / n_words
    if lang == "ru":
        return 0.055 * L - 0.35 * S - 20.33
    return 0.0588 * L - 0.296 * S - 15.8


# Lexical diversity

def type_token_ratio(words: list[str]) -> float:
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def ttr_wordforms(wordforms: list[str]) -> float:
    if not wordforms:
        return 0.0
    return len(set(wordforms)) / len(wordforms)


def yules_k(words: list[str]) -> float:
    if not words:
        return 0.0
    n = len(words)
    freq = Counter(words)
    freq_of_freq = Counter(freq.values())
    sum_vx = sum(v * (x ** 2) for x, v in freq_of_freq.items())
    return 1e4 * (sum_vx - n) / (n ** 2)


def yules_i(words: list[str]) -> float:
    if not words:
        return 0.0
    freq = Counter(words)
    n_types = len(freq)
    freq_of_freq = Counter(freq.values())
    denom = sum(v * (x ** 2) for x, v in freq_of_freq.items()) - n_types
    return (n_types ** 2) / denom if denom else 0.0


def hapax_ratio(lemmas: list[str], n: int = 1) -> float:
    if not lemmas:
        return 0.0
    freq = Counter(lemmas)
    n_hapax = sum(1 for c in freq.values() if c == n)
    return n_hapax / len(freq) if freq else 0.0
