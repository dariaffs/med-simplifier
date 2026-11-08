import importlib.metadata
import pymorphy3.analyzer

def _patched_iter_entry_points(group):
    try:
        yield from importlib.metadata.entry_points(group=group)
    except Exception:
        pass

pymorphy3.analyzer._iter_entry_points = _patched_iter_entry_points

import spacy
from metrics.primary_metrics.base_metrics import (
    count_syllables,
    avg_word_chars, letters_per_100_words, sents_per_100_words, comma_ratio,
    avg_sentence_length, avg_word_syllables, avg_sent_syllables, long_word_ratio,
    flesch_reading_ease, oborneva_index, flesch_kincaid_grade,
    smog_grade, gunning_fog, ari, dale_chall, coleman_liau, lix, rix, ovix,
    type_token_ratio, ttr_wordforms, yules_k, yules_i, hapax_ratio,
)
from metrics.primary_metrics.spacy_metrics import (
    num_unit_density, med_term_density, rare_word_ratio, zipf_bin_ratio, yavl_ratio,
    noun_ratio, verb_ratio, adj_ratio, adjf_ratio, adjs_ratio,
    pron_ratio, prep_ratio, numr_ratio, prcl_ratio, cconj_ratio, sconj_ratio,
    infn_ratio, prtf_ratio, prts_ratio, grnd_ratio, comp_ratio,
    func_word_ratio, autosem_ratio, nouns_group_ratio,
    case_ratio, verb_tense_ratio, verb_aspect_ratio, person_ratio,
    pssv_prtf_ratio, pssv_prts_ratio, neut_ratio, inan_ratio, sja_verb_ratio,
    adj_noun_bigram_ratio, verb_noun_bigram_ratio, noun_verb_bigram_ratio,
    adv_verb_bigram_ratio, grnd_noun_bigram_ratio, adv_grnd_bigram_ratio,
    part_noun_bigram_ratio, noun_part_ratio, part_adv_ratio,
    noun_noun_noun_ratio, noun_noun_noun_noun_ratio, dyn_stat_ratio,
    noun_verb_ratio, noun_chain_ratio, genitive_chain_ratio,
    subordination_ratio, passive_participle_ratio,
    acl_ratio, acl_relcl_ratio, advcl_ratio, advmod_ratio, amod_ratio,
    appos_ratio, aux_pass_ratio, cc_ratio, ccomp_ratio, compound_ratio,
    conj_ratio, cop_ratio, csubj_ratio, csubj_pass_ratio, discourse_ratio,
    mark_ratio, nsubj_ratio, nsubj_pass_ratio, nummod_ratio, orphan_ratio,
    parataxis_ratio, xcomp_ratio,
    noun_cohesion, verb_tense_cohesion,
)

RU_TEXT = "Пациент поступил с жалобами на выраженную одышку и боли в грудной клетке. При аускультации выявлено ослабление дыхания в нижних отделах лёгких. На рентгенограмме определяется инфильтрация лёгочной ткани в проекции нижней доли правого лёгкого. Назначена антибактериальная терапия препаратами широкого спектра действия."
EN_TEXT = "The patient was admitted with complaints of severe dyspnea and chest pain. Auscultation revealed diminished breath sounds in the lower lung fields. Chest radiograph demonstrated infiltration of pulmonary tissue in the right lower lobe. Broad-spectrum antibiotic therapy was initiated."


def run(text, lang):
    nlp = spacy.load("ru_core_news_lg" if lang == "ru" else "en_core_web_lg")
    doc = nlp(text)
    words = [t for t in doc if t.is_alpha]
    lemmas = [t.lemma_.lower() for t in words]
    wordforms = [t.text.lower() for t in words]
    sents = list(doc.sents)
    n_words = len(words)
    n_sents = len(sents)
    n_letters = sum(len(t.text) for t in words)
    n_commas = sum(1 for t in doc if t.text == ",")
    n_long_char = sum(1 for t in words if len(t.text) > 6)
    syl_counts = [count_syllables(t.text, lang) for t in words]
    lemma_syl_counts = [count_syllables(t.lemma_, lang) for t in words]
    n_syl = sum(syl_counts)
    asl = avg_sentence_length(n_words, n_sents)
    asw = avg_word_syllables(syl_counts)
    n_long = sum(1 for s in syl_counts if s >= (4 if lang == "ru" else 3))
    n_complex = sum(1 for s in syl_counts if s >= 3)

    print(f"\n[{lang.upper()}]")
    print(f"avg_word_chars: {avg_word_chars(n_letters, n_words):.3f}")
    print(f"letters_per_100_words: {letters_per_100_words(n_letters, n_words):.3f}")
    print(f"sents_per_100_words: {sents_per_100_words(n_sents, n_words):.3f}")
    print(f"comma_ratio: {comma_ratio(n_commas, n_words):.3f}")
    print(f"ASL: {asl:.3f}")
    print(f"ASW: {asw:.3f}")
    print(f"avg_sent_syl: {avg_sent_syllables(n_syl, n_sents):.3f}")
    print(f"long_word_ratio: {long_word_ratio(syl_counts, lang):.3f}")
    print(f"lemma_long_ratio: {long_word_ratio(lemma_syl_counts, lang):.3f}")
    if lang == "en":
        print(f"flesch_re: {flesch_reading_ease(asl, asw):.3f}")
    else:
        print(f"oborneva: {oborneva_index(asl, asw):.3f}")
    print(f"FKGL: {flesch_kincaid_grade(asl, asw, lang):.3f}")
    print(f"SMOG: {smog_grade(n_long, n_sents, lang):.3f}")
    print(f"gunning_fog: {gunning_fog(asl, n_complex / n_words if n_words else 0):.3f}")
    print(f"ARI: {ari(n_letters, n_words, n_sents, lang):.3f}")
    print(f"dale_chall: {dale_chall(n_long, n_words, n_sents, lang):.3f}")
    print(f"coleman_liau: {coleman_liau(n_letters, n_words, n_sents, lang):.3f}")
    print(f"LIX: {lix(n_long_char, n_words, n_sents):.3f}")
    print(f"RIX: {rix(n_long_char, n_sents):.3f}")
    print(f"OVIX: {ovix(n_words, len(set(wordforms))):.3f}")
    print(f"TTR_lemma: {type_token_ratio(lemmas):.3f}")
    print(f"TTR_word: {ttr_wordforms(wordforms):.3f}")
    print(f"yules_k_lemma: {yules_k(lemmas):.3f}")
    print(f"yules_k_word: {yules_k(wordforms):.3f}")
    print(f"yules_i_lemma: {yules_i(lemmas):.3f}")
    print(f"yules_i_word: {yules_i(wordforms):.3f}")
    print(f"hapax1: {hapax_ratio(lemmas, n=1):.3f}")
    print(f"hapax2: {hapax_ratio(lemmas, n=2):.3f}")
    print(f"num_unit: {num_unit_density(doc, text):.3f}")
    print(f"med_term: {med_term_density(doc, text, lang):.3f}")
    print(f"rare_word: {rare_word_ratio(doc, lang):.3f}")
    for i in range(9):
        print(f"zipf_{i}: {zipf_bin_ratio(doc, lang, i, i + 1):.3f}")
    print(f"yavl: {yavl_ratio(doc, lang):.3f}")
    print(f"noun_pr: {noun_ratio(doc):.3f}")
    print(f"verb_pr: {verb_ratio(doc):.3f}")
    print(f"adj_pr: {adj_ratio(doc):.3f}")
    print(f"adjf_pr: {adjf_ratio(doc):.3f}")
    if lang == "ru":
        print(f"adjs_pr: {adjs_ratio(doc, lang):.3f}")
    print(f"pron_pr: {pron_ratio(doc):.3f}")
    print(f"prep_pr: {prep_ratio(doc):.3f}")
    print(f"numr_pr: {numr_ratio(doc):.3f}")
    print(f"prcl_pr: {prcl_ratio(doc):.3f}")
    print(f"cconj_pr: {cconj_ratio(doc):.3f}")
    print(f"sconj_pr: {sconj_ratio(doc):.3f}")
    print(f"infn_pr: {infn_ratio(doc):.3f}")
    print(f"prtf_pr: {prtf_ratio(doc):.3f}")
    print(f"prts_pr: {prts_ratio(doc):.3f}")
    if lang == "ru":
        print(f"grnd_pr: {grnd_ratio(doc, lang):.3f}")
    print(f"comp_pr: {comp_ratio(doc):.3f}")
    print(f"func_word_pr: {func_word_ratio(doc):.3f}")
    print(f"autosem_pr: {autosem_ratio(doc):.3f}")
    print(f"nouns_group_pr: {nouns_group_ratio(doc):.3f}")
    if lang == "ru":
        for case in ("Nom", "Gen", "Dat", "Acc", "Ins", "Loc"):
            print(f"case_{case.lower()}: {case_ratio(doc, case, lang):.3f}")
    for tense in ("Past", "Pres", "Fut"):
        print(f"tense_{tense.lower()}: {verb_tense_ratio(doc, tense):.3f}")
    if lang == "ru":
        print(f"aspect_imp: {verb_aspect_ratio(doc, 'Imp', lang):.3f}")
        print(f"aspect_perf: {verb_aspect_ratio(doc, 'Perf', lang):.3f}")
    for person in ("1", "2", "3"):
        print(f"person_{person}: {person_ratio(doc, person):.3f}")
    print(f"pssv_prtf: {pssv_prtf_ratio(doc):.3f}")
    print(f"pssv_prts: {pssv_prts_ratio(doc):.3f}")
    if lang == "ru":
        print(f"neut_pr: {neut_ratio(doc, lang):.3f}")
        print(f"inan_pr: {inan_ratio(doc, lang):.3f}")
        print(f"sja_verb: {sja_verb_ratio(doc, lang):.3f}")
    print(f"adj_noun: {adj_noun_bigram_ratio(doc):.3f}")
    print(f"verb_noun: {verb_noun_bigram_ratio(doc):.3f}")
    print(f"noun_verb: {noun_verb_bigram_ratio(doc):.3f}")
    print(f"adv_verb: {adv_verb_bigram_ratio(doc):.3f}")
    if lang == "ru":
        print(f"grnd_noun: {grnd_noun_bigram_ratio(doc, lang):.3f}")
        print(f"adv_grnd: {adv_grnd_bigram_ratio(doc, lang):.3f}")
    print(f"part_noun: {part_noun_bigram_ratio(doc):.3f}")
    print(f"noun_part: {noun_part_ratio(doc):.3f}")
    print(f"part_adv: {part_adv_ratio(doc):.3f}")
    print(f"nnn: {noun_noun_noun_ratio(doc):.3f}")
    print(f"nnnn: {noun_noun_noun_noun_ratio(doc):.3f}")
    print(f"dyn_stat: {dyn_stat_ratio(doc, lang):.3f}")
    print(f"NVR: {noun_verb_ratio(doc):.3f}")
    print(f"noun_chain: {noun_chain_ratio(doc):.3f}")
    if lang == "ru":
        print(f"gen_chain: {genitive_chain_ratio(doc, lang):.3f}")
    print(f"subord: {subordination_ratio(doc):.3f}")
    print(f"passive: {passive_participle_ratio(doc, lang):.3f}")
    print(f"acl: {acl_ratio(doc):.3f}")
    print(f"acl_relcl: {acl_relcl_ratio(doc):.3f}")
    print(f"advcl: {advcl_ratio(doc):.3f}")
    print(f"advmod: {advmod_ratio(doc):.3f}")
    print(f"amod: {amod_ratio(doc):.3f}")
    print(f"appos: {appos_ratio(doc):.3f}")
    print(f"aux_pass: {aux_pass_ratio(doc):.3f}")
    print(f"cc: {cc_ratio(doc):.3f}")
    print(f"ccomp: {ccomp_ratio(doc):.3f}")
    print(f"compound: {compound_ratio(doc):.3f}")
    print(f"conj: {conj_ratio(doc):.3f}")
    print(f"cop: {cop_ratio(doc):.3f}")
    print(f"csubj: {csubj_ratio(doc):.3f}")
    print(f"csubj_pass: {csubj_pass_ratio(doc):.3f}")
    print(f"discourse: {discourse_ratio(doc):.3f}")
    print(f"mark: {mark_ratio(doc):.3f}")
    print(f"nsubj: {nsubj_ratio(doc):.3f}")
    print(f"nsubj_pass: {nsubj_pass_ratio(doc):.3f}")
    print(f"nummod: {nummod_ratio(doc):.3f}")
    print(f"orphan: {orphan_ratio(doc):.3f}")
    print(f"parataxis: {parataxis_ratio(doc):.3f}")
    print(f"xcomp: {xcomp_ratio(doc):.3f}")
    print(f"noun_cohesion: {noun_cohesion(doc):.3f}")
    print(f"verb_tense_cohesion: {verb_tense_cohesion(doc):.3f}")


run(RU_TEXT, "ru")
run(EN_TEXT, "en")
