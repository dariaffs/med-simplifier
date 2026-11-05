import importlib.metadata
import pymorphy3.analyzer

def _patched_iter_entry_points(group):
    try:
        yield from importlib.metadata.entry_points(group=group)
    except Exception:
        pass

pymorphy3.analyzer._iter_entry_points = _patched_iter_entry_points

import spacy
from metrics.base_metrics import count_syllables, avg_sentence_length, avg_word_syllables, long_word_ratio, flesch_kincaid_grade, flesch_reading_ease, oborneva_index, smog_grade, type_token_ratio, yules_k
from metrics.metrics import rare_word_ratio, subordination_ratio, passive_participle_ratio, noun_verb_ratio, noun_chain_ratio, med_term_density, num_unit_density, genitive_chain_ratio

RU_TEXT = "Пациент поступил с жалобами на выраженную одышку и боли в грудной клетке. При аускультации выявлено ослабление дыхания в нижних отделах лёгких. На рентгенограмме определяется инфильтрация лёгочной ткани в проекции нижней доли правого лёгкого. Назначена антибактериальная терапия препаратами широкого спектра действия."
EN_TEXT = "The patient was admitted with complaints of severe dyspnea and chest pain. Auscultation revealed diminished breath sounds in the lower lung fields. Chest radiograph demonstrated infiltration of pulmonary tissue in the right lower lobe. Broad-spectrum antibiotic therapy was initiated."


def run(text, lang):
    nlp = spacy.load("ru_core_news_lg" if lang == "ru" else "en_core_web_lg")
    doc = nlp(text)

    words = [t for t in doc if t.is_alpha]
    lemmas = [t.lemma_.lower() for t in words]
    sents = list(doc.sents)
    n_words = len(words)
    n_sents = len(sents)
    syl_counts = [count_syllables(t.text, lang) for t in words]
    asl = avg_sentence_length(n_words, n_sents)
    asw = avg_word_syllables(syl_counts)
    n_long = sum(1 for s in syl_counts if s >= (4 if lang == "ru" else 3))

    print(f"\n[{lang.upper()}]")
    print(f"ASL: {asl:.2f}")
    print(f"FKGL: {flesch_kincaid_grade(asl, asw, lang):.2f}")
    if lang == "en":
        print(f"Flesch Reading Ease: {flesch_reading_ease(asl, asw):.2f}")
    if lang == "ru":
        print(f"Индекс Оборневой: {oborneva_index(asl, asw):.2f}")
    print(f"SMOG: {smog_grade(n_long, n_sents, lang):.2f}")
    print(f"Длинных слов: {long_word_ratio(syl_counts, lang):.3f}")
    print(f"TTR: {type_token_ratio(lemmas):.3f}")
    print(f"Yule's K: {yules_k(lemmas):.1f}")
    print(f"Редких слов: {rare_word_ratio(doc, lang):.3f}")
    print(f"Подчинение: {subordination_ratio(doc):.3f}")
    print(f"Пассив/причастие: {passive_participle_ratio(doc, lang):.3f}")
    print(f"NVR: {noun_verb_ratio(doc):.3f}")
    print(f"NOUN+NOUN: {noun_chain_ratio(doc):.3f}")
    print(f"Медтермины: {med_term_density(doc, text, lang):.3f}")
    print(f"Числа/единицы: {num_unit_density(doc, text):.3f}")
    if lang == "ru":
        print(f"Генит. цепочки: {genitive_chain_ratio(doc, lang):.3f}")


run(RU_TEXT, "ru")
run(EN_TEXT, "en")
