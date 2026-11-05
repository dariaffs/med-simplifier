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


def _alpha(doc):
    return [t for t in doc if t.is_alpha]


def rare_word_ratio(doc, lang: str) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    n_rare = sum(1 for t in words if zipf_frequency(t.text.lower(), lang) < 3)
    return n_rare / len(words)


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


def med_term_density(doc, text: str, lang: str) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    vocab = _MED_TERMS[lang]
    n_terms = sum(1 for t in words if t.lemma_.lower() in vocab)
    n_abbr = len(_ABBR[lang].findall(text))
    return (n_terms + n_abbr) / len(words)


def num_unit_density(doc, text: str) -> float:
    words = _alpha(doc)
    if not words:
        return 0.0
    return len(_NUM_UNITS.findall(text)) / len(words)


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
