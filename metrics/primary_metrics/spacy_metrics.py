import re
from wordfreq import zipf_frequency

_SUB_DEPS = {"acl", "advcl", "ccomp", "mark"}

_ABBR = {
    "ru": re.compile(r"\b[А-ЯЁA-Z]{2,}\b"),
    "en": re.compile(r"\b[A-Z]{2,}\b"),
}

_NUM_UNITS = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*"
    r"(?:мг|г|кг|мл|л|мкг|нг|ммоль|МЕ|Ед|мм|см|мкм|"
    r"mg|g|kg|ml|mcg|ng|mmol|IU|mm|cm|%)?\b"
)

_MED_TERMS = {
    "ru": {
        "диагноз", "симптом", "терапия", "патология", "синдром", "этиология",
        "анамнез", "лечение", "препарат", "дозировка", "инфаркт", "инсульт",
        "гипертония", "диабет", "онкология", "хирургия", "биопсия", "эндоскопия",
        "томография", "рентген", "клинический", "хронический", "острый",
        "злокачественный", "доброкачественный", "прогноз", "ремиссия", "рецидив",
        "аускультация", "перкуссия", "пальпация", "инфильтрация", "резекция",
        "антибиотик", "антибактериальный", "иммунитет", "воспаление", "некроз",
        "ишемия", "тромб", "эмболия", "аритмия", "стеноз", "недостаточность",
        "гипоксия", "отёк", "асцит", "перитонит", "сепсис", "шок",
    },
    "en": {
        "diagnosis", "symptom", "therapy", "pathology", "syndrome", "etiology",
        "aetiology", "anamnesis", "patient", "treatment", "medication", "dosage",
        "infarction", "stroke", "hypertension", "diabetes", "oncology", "surgery",
        "biopsy", "endoscopy", "tomography", "radiology", "clinical", "chronic",
        "acute", "malignant", "benign", "prognosis", "remission", "relapse",
        "auscultation", "percussion", "palpation", "infiltration", "resection",
        "antibiotic", "immunity", "inflammation", "necrosis", "ischemia",
        "thrombosis", "embolism", "arrhythmia", "stenosis", "insufficiency",
        "hypoxia", "edema", "peritonitis", "sepsis", "shock", "dyspnea",
    },
}


# Helpers

def _alpha(doc):
    return [t for t in doc if t.is_alpha]


def _pos(doc):
    return [t for t in doc if not t.is_space and not t.is_punct]


def _nonspace(doc):
    return [t for t in doc if not t.is_space]


def _dep_ratio(doc, dep: str) -> float:
    words = _alpha(doc)
    return sum(1 for t in doc if t.dep_ == dep) / len(words) if words else 0.0


def _is_noun(t):
    return t.pos_ in {"NOUN", "PROPN"}


def _is_part(t):
    return "Part" in t.morph.get("VerbForm")


def _is_grnd(t):
    return "Conv" in t.morph.get("VerbForm")


def _bigram_ratio(doc, pred1, pred2) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    tokens = _pos(doc)
    return sum(1 for i in range(len(tokens) - 1) if pred1(tokens[i]) and pred2(tokens[i + 1])) / len(words)


def _trigram_ratio(doc, pred1, pred2, pred3) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    tokens = _pos(doc)
    return sum(1 for i in range(len(tokens) - 2) if pred1(tokens[i]) and pred2(tokens[i + 1]) and pred3(tokens[i + 2])) / len(words)


# Lexical / Zipf

def num_unit_density(doc, text: str) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    return len(_NUM_UNITS.findall(text)) / len(words)


def med_term_density(doc, text: str, lang: str) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    vocab = _MED_TERMS[lang]
    n_terms = sum(1 for t in words if t.lemma_.lower() in vocab)
    n_abbr = len(_ABBR[lang].findall(text))
    return (n_terms + n_abbr) / len(words)


def rare_word_ratio(doc, lang: str) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    n_rare = sum(1 for t in words if zipf_frequency(t.text.lower(), lang) < 3)
    return n_rare / len(words)


def zipf_bin_ratio(doc, lang: str, low: float, high: float) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    return sum(1 for t in words if low <= zipf_frequency(t.text.lower(), lang) < high) / len(words)


def yavl_ratio(doc, lang: str) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    target = "являться" if lang == "ru" else "be"
    return sum(1 for t in words if t.lemma_.lower() == target) / len(words)


# POS proportions

def noun_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if t.pos_ in {"NOUN", "PROPN"}) / len(p) if p else 0.0


def verb_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if t.pos_ in {"VERB", "AUX"}) / len(p) if p else 0.0


def adj_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if t.pos_ == "ADJ") / len(p) if p else 0.0


def adjf_ratio(doc) -> float:
    p = _pos(doc)
    if not p:
        return 0.0
    return sum(1 for t in p if t.pos_ == "ADJ" and "Short" not in t.morph.get("Variant", [])) / len(p)


def adjs_ratio(doc, lang: str) -> float:
    if lang != "ru":
        return 0.0
    p = _pos(doc)
    if not p:
        return 0.0
    return sum(1 for t in p if t.pos_ == "ADJ" and "Short" in t.morph.get("Variant", [])) / len(p)


def pron_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if t.pos_ in {"DET", "PRON"}) / len(p) if p else 0.0


def prep_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if t.pos_ == "ADP") / len(p) if p else 0.0


def numr_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if t.pos_ == "NUM") / len(p) if p else 0.0


def prcl_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if t.pos_ == "PART") / len(p) if p else 0.0


def cconj_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if t.pos_ == "CCONJ") / len(p) if p else 0.0


def sconj_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if t.pos_ == "SCONJ") / len(p) if p else 0.0


def infn_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if "Inf" in t.morph.get("VerbForm")) / len(p) if p else 0.0


def prtf_ratio(doc) -> float:
    p = _pos(doc)
    if not p:
        return 0.0
    return sum(1 for t in p if _is_part(t) and "Short" not in t.morph.get("Variant", [])) / len(p)


def prts_ratio(doc) -> float:
    p = _pos(doc)
    if not p:
        return 0.0
    return sum(1 for t in p if _is_part(t) and "Short" in t.morph.get("Variant", [])) / len(p)


def grnd_ratio(doc, lang: str) -> float:
    if lang != "ru":
        return 0.0
    p = _pos(doc)
    return sum(1 for t in p if _is_grnd(t)) / len(p) if p else 0.0


def comp_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if "Cmp" in t.morph.get("Degree", [])) / len(p) if p else 0.0


def func_word_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if t.pos_ in {"ADP", "AUX", "CCONJ", "PART", "SCONJ"}) / len(p) if p else 0.0


def autosem_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if t.pos_ in {"ADJ", "ADV", "NOUN", "NUM", "PROPN", "VERB"}) / len(p) if p else 0.0


def nouns_group_ratio(doc) -> float:
    p = _pos(doc)
    return sum(1 for t in p if t.pos_ in {"ADJ", "NOUN", "PROPN"}) / len(p) if p else 0.0


# Morphological features

def case_ratio(doc, case: str, lang: str) -> float:
    if lang != "ru":
        return 0.0
    words = _alpha(doc)
    return sum(1 for t in words if case in t.morph.get("Case")) / len(words) if words else 0.0


def verb_tense_ratio(doc, tense: str) -> float:
    verbs = [t for t in doc if t.pos_ in {"VERB", "AUX"}]
    return sum(1 for t in verbs if tense in t.morph.get("Tense")) / len(verbs) if verbs else 0.0


def verb_aspect_ratio(doc, aspect: str, lang: str) -> float:
    if lang != "ru":
        return 0.0
    verbs = [t for t in doc if t.pos_ in {"VERB", "AUX"}]
    return sum(1 for t in verbs if aspect in t.morph.get("Aspect")) / len(verbs) if verbs else 0.0


def person_ratio(doc, person: str) -> float:
    verbs = [t for t in doc if t.pos_ in {"VERB", "AUX"}]
    return sum(1 for t in verbs if person in t.morph.get("Person")) / len(verbs) if verbs else 0.0


def pssv_prtf_ratio(doc) -> float:
    p = _pos(doc)
    if not p:
        return 0.0
    return sum(1 for t in p if _is_part(t) and "Pass" in t.morph.get("Voice", []) and "Short" not in t.morph.get("Variant", [])) / len(p)


def pssv_prts_ratio(doc) -> float:
    p = _pos(doc)
    if not p:
        return 0.0
    return sum(1 for t in p if _is_part(t) and "Pass" in t.morph.get("Voice", []) and "Short" in t.morph.get("Variant", [])) / len(p)


def neut_ratio(doc, lang: str) -> float:
    if lang != "ru":
        return 0.0
    nouns = [t for t in doc if t.pos_ == "NOUN"]
    return sum(1 for t in nouns if "Neut" in t.morph.get("Gender")) / len(nouns) if nouns else 0.0


def inan_ratio(doc, lang: str) -> float:
    if lang != "ru":
        return 0.0
    nouns = [t for t in doc if t.pos_ == "NOUN"]
    return sum(1 for t in nouns if "Inan" in t.morph.get("Animacy")) / len(nouns) if nouns else 0.0


def sja_verb_ratio(doc, lang: str) -> float:
    if lang != "ru":
        return 0.0
    words = _alpha(doc)
    if not words:
        return 0.0
    verbs = [t for t in doc if t.pos_ == "VERB"]
    return sum(1 for t in verbs if "Yes" in t.morph.get("Reflex", [])) / len(words)


# POS n-grams

def adj_noun_bigram_ratio(doc) -> float:
    return _bigram_ratio(doc, lambda t: t.pos_ == "ADJ", _is_noun)


def verb_noun_bigram_ratio(doc) -> float:
    return _bigram_ratio(doc, lambda t: t.pos_ == "VERB", _is_noun)


def noun_verb_bigram_ratio(doc) -> float:
    return _bigram_ratio(doc, _is_noun, lambda t: t.pos_ == "VERB")


def adv_verb_bigram_ratio(doc) -> float:
    return _bigram_ratio(doc, lambda t: t.pos_ == "ADV", lambda t: t.pos_ == "VERB")


def grnd_noun_bigram_ratio(doc, lang: str) -> float:
    if lang != "ru":
        return 0.0
    return _bigram_ratio(doc, _is_grnd, _is_noun)


def adv_grnd_bigram_ratio(doc, lang: str) -> float:
    if lang != "ru":
        return 0.0
    return _bigram_ratio(doc, lambda t: t.pos_ == "ADV", _is_grnd)


def part_noun_bigram_ratio(doc) -> float:
    return _bigram_ratio(doc, _is_part, _is_noun)


def noun_part_ratio(doc) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    all_tokens = _nonspace(doc)
    n = 0
    for i in range(len(all_tokens) - 1):
        if _is_noun(all_tokens[i]) and _is_part(all_tokens[i + 1]):
            n += 1
        if i < len(all_tokens) - 2:
            if _is_noun(all_tokens[i]) and all_tokens[i + 1].is_punct and _is_part(all_tokens[i + 2]):
                n += 1
    return n / len(words)


def part_adv_ratio(doc) -> float:
    return _bigram_ratio(doc, _is_part, lambda t: t.pos_ == "ADV")


def noun_noun_noun_ratio(doc) -> float:
    return _trigram_ratio(doc, _is_noun, _is_noun, _is_noun)


def noun_noun_noun_noun_ratio(doc) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    tokens = _pos(doc)
    n = sum(1 for i in range(len(tokens) - 3) if all(_is_noun(tokens[i + j]) for j in range(4)))
    return n / len(words)


def dyn_stat_ratio(doc, lang: str) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    tokens = _pos(doc)
    n1 = n2 = n3 = n4 = n5 = n8 = n9 = 0
    for i in range(len(tokens) - 1):
        t1, t2 = tokens[i], tokens[i + 1]
        if t1.pos_ == "VERB" and _is_noun(t2): n1 += 1
        if _is_noun(t1) and t2.pos_ == "VERB": n2 += 1
        if t1.pos_ == "ADV" and t2.pos_ == "VERB": n3 += 1
        if t1.pos_ == "ADJ" and _is_noun(t2): n4 += 1
        if _is_noun(t1) and _is_noun(t2): n5 += 1
        if lang == "ru":
            if _is_grnd(t1) and _is_noun(t2): n8 += 1
            if t1.pos_ == "ADV" and _is_grnd(t2): n9 += 1
    static = n4 + n5
    dynamic = n1 + n2 + n3 + n8 + n9
    return dynamic / static if static else 0.0


# Syntactic / structural

def noun_verb_ratio(doc) -> float:
    n_nouns = sum(1 for t in doc if t.pos_ in {"NOUN", "PROPN"})
    n_verbs = sum(1 for t in doc if t.pos_ in {"VERB", "AUX"})
    return n_nouns / max(n_verbs, 1)


def noun_chain_ratio(doc) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    tokens = [t for t in doc if not t.is_space]
    n_chains = sum(
        1 for i in range(len(tokens) - 1)
        if tokens[i].pos_ in {"NOUN", "PROPN"}
        and tokens[i + 1].pos_ in {"NOUN", "PROPN"}
    )
    return n_chains / len(words)


def genitive_chain_ratio(doc, lang: str) -> float:
    if lang != "ru":
        return 0.0
    words = _alpha(doc)
    if not words:
        return 0.0
    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    n_gen_chains = sum(
        1 for i in range(len(tokens) - 1)
        if tokens[i].pos_ in {"NOUN", "PROPN"}
        and tokens[i + 1].pos_ in {"NOUN", "PROPN"}
        and "Gen" in tokens[i + 1].morph.get("Case")
    )
    return n_gen_chains / len(words)


def subordination_ratio(doc) -> float:
    sents = list(doc.sents)
    if not sents:
        return 0.0
    n_sub = sum(1 for t in doc if t.dep_ in _SUB_DEPS)
    return n_sub / len(sents)


def passive_participle_ratio(doc, lang: str) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    verb_forms = {"Part"} | ({"Ger"} if lang == "en" else set())
    n = sum(
        1 for t in doc
        if t.dep_ in {"nsubj:pass", "aux:pass"}
        or bool(set(t.morph.get("VerbForm")) & verb_forms)
    )
    return n / len(words)


# UD dependency ratios

def acl_ratio(doc) -> float: return _dep_ratio(doc, "acl")
def acl_relcl_ratio(doc) -> float: return _dep_ratio(doc, "acl:relcl")
def advcl_ratio(doc) -> float: return _dep_ratio(doc, "advcl")
def advmod_ratio(doc) -> float: return _dep_ratio(doc, "advmod")
def amod_ratio(doc) -> float: return _dep_ratio(doc, "amod")
def appos_ratio(doc) -> float: return _dep_ratio(doc, "appos")
def aux_pass_ratio(doc) -> float: return _dep_ratio(doc, "aux:pass")
def cc_ratio(doc) -> float: return _dep_ratio(doc, "cc")
def ccomp_ratio(doc) -> float: return _dep_ratio(doc, "ccomp")
def compound_ratio(doc) -> float: return _dep_ratio(doc, "compound")
def conj_ratio(doc) -> float: return _dep_ratio(doc, "conj")
def cop_ratio(doc) -> float: return _dep_ratio(doc, "cop")
def csubj_ratio(doc) -> float: return _dep_ratio(doc, "csubj")
def csubj_pass_ratio(doc) -> float: return _dep_ratio(doc, "csubj:pass")
def discourse_ratio(doc) -> float: return _dep_ratio(doc, "discourse")
def mark_ratio(doc) -> float: return _dep_ratio(doc, "mark")
def nsubj_ratio(doc) -> float: return _dep_ratio(doc, "nsubj")
def nsubj_pass_ratio(doc) -> float: return _dep_ratio(doc, "nsubj:pass")
def nummod_ratio(doc) -> float: return _dep_ratio(doc, "nummod")
def orphan_ratio(doc) -> float: return _dep_ratio(doc, "orphan")
def parataxis_ratio(doc) -> float: return _dep_ratio(doc, "parataxis")
def xcomp_ratio(doc) -> float: return _dep_ratio(doc, "xcomp")


# Coherence

def noun_cohesion(doc) -> float:
    sents = list(doc.sents)
    words = _alpha(doc)
    if len(sents) < 2 or not words:
        return 0.0
    total = sum(
        len({t.lemma_.lower() for t in sents[i] if _is_noun(t)} &
            {t.lemma_.lower() for t in sents[i + 1] if _is_noun(t)})
        for i in range(len(sents) - 1)
    )
    return total / len(words)


def verb_tense_cohesion(doc) -> float:
    sents = list(doc.sents)
    words = _alpha(doc)
    if len(sents) < 2 or not words:
        return 0.0
    def tense_set(sent):
        return {tuple(t.morph.get("Tense")) for t in sent if t.pos_ in {"VERB", "AUX"} and t.morph.get("Tense")}
    total = sum(len(tense_set(sents[i]) & tense_set(sents[i + 1])) for i in range(len(sents) - 1))
    return total / len(words)
