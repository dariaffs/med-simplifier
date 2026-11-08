# med-simplifier

Multilingual (eng + ru) agent for medical texts simplification with controllable complexity parameters 


## Primary metrics. Computed for both the original and simplified text

### Base metrics: `base_metrics.py`

| Metric | Description | Formula (Russian) | Formula (English) |
|--------|-------------|-------------------|-------------------|
| avg_word_chars (ACW) | Average word length in characters | `n_letters / n_words` | `n_letters / n_words` |
| letters_per_100_words (L) | Letters per 100 words | `100×n_letters / n_words` | `100×n_letters / n_words` |
| sents_per_100_words (S) | Sentences per 100 words | `100×n_sents / n_words` | `100×n_sents / n_words` |
| comma_ratio | Ratio of commas to words | `n_commas / n_words` | `n_commas / n_words` |
| ASL | Average sentence length in words | `n_words / n_sents` | `n_words / n_sents` |
| ASW | Average word length in syllables | `n_syl / n_words` | `n_syl / n_words` |
| ASS (avg_sent_syllables) | Average sentence length in syllables | `n_syl / n_sents` | `n_syl / n_sents` |
| long_word_ratio | Ratio of words above syllable threshold | `n_words(≥4 syl) / n_words` | `n_words(≥3 syl) / n_words` |
| Flesch Reading Ease / Oborneva Index | Readability index; lower value = more complex text | `206.835 − 1.52×ASL − 65.14×ASW` | `206.835 − 1.015×ASL − 84.6×ASW` |
| FKGL | Flesch–Kincaid Grade Level, adapted for Russian | `0.5×ASL + 8.4×ASW − 15.59` | `0.39×ASL + 11.8×ASW − 15.59` |
| SMOG | Complexity index based on polysyllabic words | `1.1×√(64.6/n_sents × n_long) + 0.05` | `1.043×√(n_long×30/n_sents) + 3.13` |
| Gunning Fog | Grade level based on sentence length and complex words | `0.4×(ASL + 100×complex_word_ratio)` | `0.4×(ASL + 100×complex_word_ratio)` |
| LIX | Readability index based on sentence length and long words (>6 chars) | `N/S + 100×n_long_char/N` | `N/S + 100×n_long_char/N` |
| RIX | Ratio of long words (>6 chars) to sentences | `n_long_char / n_sents` | `n_long_char / n_sents` |
| OVIX | Vocabulary variation index | `ln(N) / (2 − ln(V)/ln(N))` | `ln(N) / (2 − ln(V)/ln(N))` |
| ARI | Automated Readability Index | `6.26×(C/N) + 0.2805×(N/S) − 31.04` | `4.71×(C/N) + 0.5×(N/S) − 21.43` |
| Dale–Chall | Readability score based on long words and sentence length | `0.552×(100×word_long/N) + 0.273×(N/S)` | `0.1579×(C/T) + 0.0496×(T/S)` |
| Coleman–Liau | Readability index based on letters and sentences per 100 words | `0.055×L − 0.35×S − 20.33` | `0.0588×L − 0.296×S − 15.8` |
| TTR (lemmas) | Type-token ratio by lemmas | `n_unique_lemmas / n_lemmas` | `n_unique_lemmas / n_lemmas` |
| TTR (wordforms) | Type-token ratio by wordforms | `n_unique_forms / n_forms` | `n_unique_forms / n_forms` |
| Yule's K | Lexical diversity; higher = less diverse | `10⁴ × (Σ Vx×x² − N) / N²` | `10⁴ × (Σ Vx×x² − N) / N²` |
| Yule's I | Lexical diversity; inverse of Yule's K | `V² / (Σ i²×Vi − V)` | `V² / (Σ i²×Vi − V)` |
| hapax1_ratio | Ratio of hapax legomena (occur once) to unique lemmas | `n_hapax / n_unique_lemmas` | `n_hapax / n_unique_lemmas` |
| hapax2_ratio | Ratio of dis legomena (occur twice) to unique lemmas | `n_dislegomena / n_unique_lemmas` | `n_dislegomena / n_unique_lemmas` |

### spaCy-based metrics: `spacy_metrics.py`

| Metric | Description | Formula (Russian) | Formula (English) |
|--------|-------------|-------------------|-------------------|
| num_unit_density | Density of numeric expressions and dosages | `n_regex_matches / n_words` | `n_regex_matches / n_words` |
| med_term_density | Density of medical vocabulary and abbreviations | `(n_terms + n_abbr) / n_words` | `(n_terms + n_abbr) / n_words` |
| rare_word_ratio | Ratio of words with Zipf frequency < 3 | `n_words(Zipf < 3) / n_words` | `n_words(Zipf < 3) / n_words` |
| zipf_0 … zipf_8 | Ratio of words in each integer Zipf frequency bin | `n_words(i ≤ Zipf < i+1) / n_words` | `n_words(i ≤ Zipf < i+1) / n_words` |
| yavl_ratio | Ratio of copular verb "являться" / "be" | `n("являться") / n_words` | `n("be") / n_words` |
| noun_ratio | Ratio of nouns including proper nouns | `(NOUN + PROPN) / n_tokens` | `(NOUN + PROPN) / n_tokens` |
| verb_ratio | Ratio of verbs including auxiliaries | `(VERB + AUX) / n_tokens` | `(VERB + AUX) / n_tokens` |
| adj_ratio | Ratio of adjectives | `ADJ / n_tokens` | `ADJ / n_tokens` |
| adjf_ratio | Ratio of full-form adjectives | `ADJ(Variant≠Short) / n_tokens` | `ADJ / n_tokens` |
| adjs_ratio | Ratio of short-form adjectives | `ADJ(Variant=Short) / n_tokens` | — |
| pron_ratio | Ratio of pronouns and determiners | `(DET + PRON) / n_tokens` | `(DET + PRON) / n_tokens` |
| prep_ratio | Ratio of prepositions | `ADP / n_tokens` | `ADP / n_tokens` |
| numr_ratio | Ratio of numerals | `NUM / n_tokens` | `NUM / n_tokens` |
| prcl_ratio | Ratio of particles | `PART / n_tokens` | `PART / n_tokens` |
| cconj_ratio | Ratio of coordinating conjunctions | `CCONJ / n_tokens` | `CCONJ / n_tokens` |
| sconj_ratio | Ratio of subordinating conjunctions | `SCONJ / n_tokens` | `SCONJ / n_tokens` |
| infn_ratio | Ratio of infinitives | `VerbForm=Inf / n_tokens` | `VerbForm=Inf / n_tokens` |
| prtf_ratio | Ratio of full participles | `VerbForm=Part, Variant≠Short / n_tokens` | `VerbForm=Part, Variant≠Short / n_tokens` |
| prts_ratio | Ratio of short participles | `VerbForm=Part, Variant=Short / n_tokens` | `VerbForm=Part, Variant=Short / n_tokens` |
| grnd_ratio | Ratio of converbs (verbal adverbs, -я/-в) | `VerbForm=Conv / n_tokens` | — |
| comp_ratio | Ratio of comparative forms | `Degree=Cmp / n_tokens` | `Degree=Cmp / n_tokens` |
| func_word_ratio | Ratio of function words | `(ADP + AUX + CCONJ + PART + SCONJ) / n_tokens` | `(ADP + AUX + CCONJ + PART + SCONJ) / n_tokens` |
| autosem_ratio | Ratio of content words | `(ADJ + ADV + NOUN + NUM + PROPN + VERB) / n_tokens` | `(ADJ + ADV + NOUN + NUM + PROPN + VERB) / n_tokens` |
| nouns_group_ratio | Ratio of nominal group words | `(ADJ + NOUN + PROPN) / n_tokens` | `(ADJ + NOUN + PROPN) / n_tokens` |
| case_nom … case_loc | Ratio of words in each of 6 grammatical cases | `n_words(Case=X) / n_words` | — |
| verb_tense_ratio | Ratio of verbs in each tense (Past / Pres / Fut) | `n_verbs(Tense=X) / n_verbs` | `n_verbs(Tense=X) / n_verbs` |
| verb_aspect_ratio | Ratio of verbs by aspect (Imp / Perf) | `n_verbs(Aspect=X) / n_verbs` | — |
| person_ratio | Ratio of verbs by grammatical person (1 / 2 / 3) | `n_verbs(Person=X) / n_verbs` | `n_verbs(Person=X) / n_verbs` |
| pssv_prtf_ratio | Ratio of full passive participles | `n(VerbForm=Part, Voice=Pass, Variant≠Short) / n_tokens` | `n(VerbForm=Part, Voice=Pass, Variant≠Short) / n_tokens` |
| pssv_prts_ratio | Ratio of short passive participles | `n(VerbForm=Part, Voice=Pass, Variant=Short) / n_tokens` | `n(VerbForm=Part, Voice=Pass, Variant=Short) / n_tokens` |
| neut_ratio | Ratio of neuter nouns | `n_nouns(Gender=Neut) / n_nouns` | — |
| inan_ratio | Ratio of inanimate nouns | `n_nouns(Animacy=Inan) / n_nouns` | — |
| sja_verb_ratio | Ratio of reflexive verbs ending in -ся | `n_verbs(Reflex=Yes) / n_words` | — |
| adj_noun | ADJ+NOUN bigram ratio | `n_bigrams(ADJ+NOUN) / n_words` | `n_bigrams(ADJ+NOUN) / n_words` |
| verb_noun | VERB+NOUN bigram ratio | `n_bigrams(VERB+NOUN) / n_words` | `n_bigrams(VERB+NOUN) / n_words` |
| noun_verb | NOUN+VERB bigram ratio | `n_bigrams(NOUN+VERB) / n_words` | `n_bigrams(NOUN+VERB) / n_words` |
| adv_verb | ADV+VERB bigram ratio | `n_bigrams(ADV+VERB) / n_words` | `n_bigrams(ADV+VERB) / n_words` |
| grnd_noun | Converb+NOUN bigram ratio | `n_bigrams(VerbForm=Conv + NOUN) / n_words` | — |
| adv_grnd | ADV+Converb bigram ratio | `n_bigrams(ADV + VerbForm=Conv) / n_words` | — |
| part_noun | Participle+NOUN bigram ratio | `n_bigrams(VerbForm=Part + NOUN) / n_words` | `n_bigrams(VerbForm=Part + NOUN) / n_words` |
| noun_part | NOUN+Participle bigram ratio (incl. through punct) | `n_bigrams(NOUN + (PUNCT +) VerbForm=Part) / n_words` | `n_bigrams(NOUN + (PUNCT +) VerbForm=Part) / n_words` |
| part_adv | Participle+ADV bigram ratio | `n_bigrams(VerbForm=Part + ADV) / n_words` | `n_bigrams(VerbForm=Part + ADV) / n_words` |
| nnn | NOUN+NOUN+NOUN trigram ratio | `n_trigrams(NOUN×3) / n_words` | `n_trigrams(NOUN×3) / n_words` |
| nnnn | NOUN×4 chain ratio | `n_4grams(NOUN×4) / n_words` | `n_4grams(NOUN×4) / n_words` |
| dyn_stat_ratio | Dynamic-to-static construction ratio | `(VERB+NOUN + NOUN+VERB + ADV+VERB + Conv+NOUN + ADV+Conv) / (ADJ+NOUN + NOUN+NOUN)` | `(VERB+NOUN + NOUN+VERB + ADV+VERB) / (ADJ+NOUN + NOUN+NOUN)` |
| NVR | Noun-to-verb ratio | `(NOUN + PROPN) / (VERB + AUX)` | `(NOUN + PROPN) / (VERB + AUX)` |
| noun_chain_ratio | Ratio of adjacent NOUN+NOUN pairs | `n_pairs(NOUN+NOUN) / n_words` | `n_pairs(NOUN+NOUN) / n_words` |
| genitive_chain_ratio | Ratio of noun chains with genitive second element | `n_pairs(NOUN→NOUN, Case=Gen) / n_words` | — |
| subordination_ratio | Average number of subordinate clauses per sentence | `n_tokens(acl, advcl, ccomp, mark) / n_sents` | `n_tokens(acl, advcl, ccomp, mark) / n_sents` |
| passive_participle_ratio | Ratio of passive and participial constructions | `n_tokens(nsubj:pass, aux:pass, VerbForm=Part) / n_words` | `n_tokens(nsubj:pass, aux:pass, VerbForm=Part, VerbForm=Ger) / n_words` |
| acl | Ratio of clausal modifiers of nouns | `n_dep(acl) / n_words` | `n_dep(acl) / n_words` |
| acl_relcl | Ratio of relative clause modifiers | `n_dep(acl:relcl) / n_words` | `n_dep(acl:relcl) / n_words` |
| advcl | Ratio of adverbial clause modifiers | `n_dep(advcl) / n_words` | `n_dep(advcl) / n_words` |
| advmod | Ratio of adverbial modifiers | `n_dep(advmod) / n_words` | `n_dep(advmod) / n_words` |
| amod | Ratio of adjectival modifiers | `n_dep(amod) / n_words` | `n_dep(amod) / n_words` |
| appos | Ratio of appositional modifiers | `n_dep(appos) / n_words` | `n_dep(appos) / n_words` |
| aux_pass | Ratio of passive auxiliary relations | `n_dep(aux:pass) / n_words` | `n_dep(aux:pass) / n_words` |
| cc | Ratio of coordinating conjunction relations | `n_dep(cc) / n_words` | `n_dep(cc) / n_words` |
| ccomp | Ratio of clausal complement relations | `n_dep(ccomp) / n_words` | `n_dep(ccomp) / n_words` |
| compound | Ratio of compound relations | `n_dep(compound) / n_words` | `n_dep(compound) / n_words` |
| conj | Ratio of conjunct relations | `n_dep(conj) / n_words` | `n_dep(conj) / n_words` |
| cop | Ratio of copula relations | `n_dep(cop) / n_words` | `n_dep(cop) / n_words` |
| csubj | Ratio of clausal subject relations | `n_dep(csubj) / n_words` | `n_dep(csubj) / n_words` |
| csubj_pass | Ratio of passive clausal subject relations | `n_dep(csubj:pass) / n_words` | `n_dep(csubj:pass) / n_words` |
| discourse | Ratio of discourse element relations | `n_dep(discourse) / n_words` | `n_dep(discourse) / n_words` |
| mark | Ratio of subordinating marker relations | `n_dep(mark) / n_words` | `n_dep(mark) / n_words` |
| nsubj | Ratio of nominal subject relations | `n_dep(nsubj) / n_words` | `n_dep(nsubj) / n_words` |
| nsubj_pass | Ratio of passive nominal subject relations | `n_dep(nsubj:pass) / n_words` | `n_dep(nsubj:pass) / n_words` |
| nummod | Ratio of numeric modifier relations | `n_dep(nummod) / n_words` | `n_dep(nummod) / n_words` |
| orphan | Ratio of orphan relations (predicate ellipsis) | `n_dep(orphan) / n_words` | `n_dep(orphan) / n_words` |
| parataxis | Ratio of parataxis relations | `n_dep(parataxis) / n_words` | `n_dep(parataxis) / n_words` |
| xcomp | Ratio of open clausal complement relations | `n_dep(xcomp) / n_words` | `n_dep(xcomp) / n_words` |
| noun_cohesion | Noun lemma overlap between adjacent sentences | `n_shared_noun_lemmas / n_words` | `n_shared_noun_lemmas / n_words` |
| verb_tense_cohesion | Verb tense overlap between adjacent sentences | `n_shared_tenses / n_words` | `n_shared_tenses / n_words` |

---

## Secondary metrics TODO

TODO: SARI, BLEU, BERTScore, compression ratio
