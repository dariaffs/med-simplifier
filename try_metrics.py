from pipeline import compute_metrics

RU_TEXT = "Пациент поступил с жалобами на выраженную одышку и боли в грудной клетке. При аускультации выявлено ослабление дыхания в нижних отделах лёгких. На рентгенограмме определяется инфильтрация лёгочной ткани в проекции нижней доли правого лёгкого. Назначена антибактериальная терапия препаратами широкого спектра действия."
EN_TEXT = "The patient was admitted with complaints of severe dyspnea and chest pain. Auscultation revealed diminished breath sounds in the lower lung fields. Chest radiograph demonstrated infiltration of pulmonary tissue in the right lower lobe. Broad-spectrum antibiotic therapy was initiated."

for lang, text in (("ru", RU_TEXT), ("en", EN_TEXT)):
    print(f"\n[{lang.upper()}]")
    for key, val in compute_metrics(text, lang).items():
        print(f"{key}: {val:.3f}")
