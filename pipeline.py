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


_nlp_cache = {}

def _load_nlp(lang: str):
    if lang not in _nlp_cache:
        _nlp_cache[lang] = spacy.load("ru_core_news_lg" if lang == "ru" else "en_core_web_lg")
    return _nlp_cache[lang]


def compute_metrics(text: str, lang: str) -> dict:
    nlp = _load_nlp(lang)
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

    m = {}

    # Surface statistics
    m["avg_word_chars"] = avg_word_chars(n_letters, n_words)
    m["letters_per_100_words"] = letters_per_100_words(n_letters, n_words)
    m["sents_per_100_words"] = sents_per_100_words(n_sents, n_words)
    m["comma_ratio"] = comma_ratio(n_commas, n_words)

    # Length
    m["ASL"] = asl
    m["ASW"] = asw
    m["avg_sent_syllables"] = avg_sent_syllables(n_syl, n_sents)
    m["long_word_ratio"] = long_word_ratio(syl_counts, lang)
    m["lemma_long_ratio"] = long_word_ratio(lemma_syl_counts, lang)

    # Readability
    if lang == "ru":
        m["oborneva"] = oborneva_index(asl, asw)
    else:
        m["flesch_re"] = flesch_reading_ease(asl, asw)
    m["FKGL"] = flesch_kincaid_grade(asl, asw, lang)
    m["SMOG"] = smog_grade(n_long, n_sents, lang)
    m["gunning_fog"] = gunning_fog(asl, n_complex / n_words if n_words else 0)
    m["ARI"] = ari(n_letters, n_words, n_sents, lang)
    m["dale_chall"] = dale_chall(n_long, n_words, n_sents, lang)
    m["coleman_liau"] = coleman_liau(n_letters, n_words, n_sents, lang)
    m["LIX"] = lix(n_long_char, n_words, n_sents)
    m["RIX"] = rix(n_long_char, n_sents)
    m["OVIX"] = ovix(n_words, len(set(wordforms)))

    # Lexical diversity
    m["TTR_lemma"] = type_token_ratio(lemmas)
    m["TTR_word"] = ttr_wordforms(wordforms)
    m["yules_k_lemma"] = yules_k(lemmas)
    m["yules_k_word"] = yules_k(wordforms)
    m["yules_i_lemma"] = yules_i(lemmas)
    m["yules_i_word"] = yules_i(wordforms)
    m["hapax1"] = hapax_ratio(lemmas, n=1)
    m["hapax2"] = hapax_ratio(lemmas, n=2)

    # Lexical / Zipf
    m["num_unit_density"] = num_unit_density(doc, text)
    m["med_term_density"] = med_term_density(doc, text, lang)
    m["rare_word_ratio"] = rare_word_ratio(doc, lang)
    for i in range(9):
        m[f"zipf_{i}"] = zipf_bin_ratio(doc, lang, i, i + 1)
    m["yavl_ratio"] = yavl_ratio(doc, lang)

    # POS proportions
    m["noun_ratio"] = noun_ratio(doc)
    m["verb_ratio"] = verb_ratio(doc)
    m["adj_ratio"] = adj_ratio(doc)
    m["adjf_ratio"] = adjf_ratio(doc)
    if lang == "ru":
        m["adjs_ratio"] = adjs_ratio(doc, lang)
    m["pron_ratio"] = pron_ratio(doc)
    m["prep_ratio"] = prep_ratio(doc)
    m["numr_ratio"] = numr_ratio(doc)
    m["prcl_ratio"] = prcl_ratio(doc)
    m["cconj_ratio"] = cconj_ratio(doc)
    m["sconj_ratio"] = sconj_ratio(doc)
    m["infn_ratio"] = infn_ratio(doc)
    m["prtf_ratio"] = prtf_ratio(doc)
    m["prts_ratio"] = prts_ratio(doc)
    if lang == "ru":
        m["grnd_ratio"] = grnd_ratio(doc, lang)
    m["comp_ratio"] = comp_ratio(doc)
    m["func_word_ratio"] = func_word_ratio(doc)
    m["autosem_ratio"] = autosem_ratio(doc)
    m["nouns_group_ratio"] = nouns_group_ratio(doc)

    # Morphological features
    if lang == "ru":
        for case in ("Nom", "Gen", "Dat", "Acc", "Ins", "Loc"):
            m[f"case_{case.lower()}"] = case_ratio(doc, case, lang)
    for tense in ("Past", "Pres", "Fut"):
        m[f"tense_{tense.lower()}"] = verb_tense_ratio(doc, tense)
    if lang == "ru":
        m["aspect_imp"] = verb_aspect_ratio(doc, "Imp", lang)
        m["aspect_perf"] = verb_aspect_ratio(doc, "Perf", lang)
    for person in ("1", "2", "3"):
        m[f"person_{person}"] = person_ratio(doc, person)
    m["pssv_prtf"] = pssv_prtf_ratio(doc)
    m["pssv_prts"] = pssv_prts_ratio(doc)
    if lang == "ru":
        m["neut_ratio"] = neut_ratio(doc, lang)
        m["inan_ratio"] = inan_ratio(doc, lang)
        m["sja_verb_ratio"] = sja_verb_ratio(doc, lang)

    # POS n-grams
    m["adj_noun"] = adj_noun_bigram_ratio(doc)
    m["verb_noun"] = verb_noun_bigram_ratio(doc)
    m["noun_verb"] = noun_verb_bigram_ratio(doc)
    m["adv_verb"] = adv_verb_bigram_ratio(doc)
    if lang == "ru":
        m["grnd_noun"] = grnd_noun_bigram_ratio(doc, lang)
        m["adv_grnd"] = adv_grnd_bigram_ratio(doc, lang)
    m["part_noun"] = part_noun_bigram_ratio(doc)
    m["noun_part"] = noun_part_ratio(doc)
    m["part_adv"] = part_adv_ratio(doc)
    m["nnn"] = noun_noun_noun_ratio(doc)
    m["nnnn"] = noun_noun_noun_noun_ratio(doc)

    # Syntactic / structural
    m["dyn_stat"] = dyn_stat_ratio(doc, lang)
    m["NVR"] = noun_verb_ratio(doc)
    m["noun_chain"] = noun_chain_ratio(doc)
    if lang == "ru":
        m["gen_chain"] = genitive_chain_ratio(doc, lang)
    m["subordination"] = subordination_ratio(doc)
    m["passive"] = passive_participle_ratio(doc, lang)

    # UD dependencies
    m["acl"] = acl_ratio(doc)
    m["acl_relcl"] = acl_relcl_ratio(doc)
    m["advcl"] = advcl_ratio(doc)
    m["advmod"] = advmod_ratio(doc)
    m["amod"] = amod_ratio(doc)
    m["appos"] = appos_ratio(doc)
    m["aux_pass"] = aux_pass_ratio(doc)
    m["cc"] = cc_ratio(doc)
    m["ccomp"] = ccomp_ratio(doc)
    m["compound"] = compound_ratio(doc)
    m["conj"] = conj_ratio(doc)
    m["cop"] = cop_ratio(doc)
    m["csubj"] = csubj_ratio(doc)
    m["csubj_pass"] = csubj_pass_ratio(doc)
    m["discourse"] = discourse_ratio(doc)
    m["mark"] = mark_ratio(doc)
    m["nsubj"] = nsubj_ratio(doc)
    m["nsubj_pass"] = nsubj_pass_ratio(doc)
    m["nummod"] = nummod_ratio(doc)
    m["orphan"] = orphan_ratio(doc)
    m["parataxis"] = parataxis_ratio(doc)
    m["xcomp"] = xcomp_ratio(doc)

    # Coherence
    m["noun_cohesion"] = noun_cohesion(doc)
    m["verb_tense_cohesion"] = verb_tense_cohesion(doc)

    return m
