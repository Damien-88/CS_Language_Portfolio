"""Heuristic morphology rules for POS guessing and feature inference."""


ADJECTIVE_ENDINGS = (
    "ого",
    "его",
    "ому",
    "ему",
    "ыми",
    "ими",
    "ая",
    "яя",
    "ое",
    "ее",
    "ый",
    "ий",
    "ой",
    "ые",
    "ие",
    "ую",
    "юю",
    "ым",
    "им",
    "ом",
    "ем",
)

VERB_ENDINGS_POS = (
    "ться",
    "тись",
    "ать",
    "ять",
    "еть",
    "ить",
    "ыть",
    "овать",
    "ую",
    "юю",
    "аю",
    "яю",
    "ешь",
    "ет",
    "ем",
    "ете",
    "ут",
    "ют",
    "ишь",
    "ит",
    "им",
    "ите",
    "ат",
    "ят",
    "л",
    "ла",
    "ло",
    "ли",
)

def guess_part_of_speech(token):
    """Guess coarse POS using ending heuristics and pronoun lexicon."""
    lower = token.lower()
    if lower in {"я", "ты", "он", "она", "оно", "мы", "вы", "они"}:
        return "pronoun"
    if any(lower.endswith(ending) for ending in ADJECTIVE_ENDINGS):
        return "adjective"
    if any(lower.endswith(ending) for ending in VERB_ENDINGS_POS):
        return "verb"
    return "noun"


def infer_features(token, part_of_speech):
    """Dispatch feature inference to a POS-specific heuristic strategy."""
    strategies = {
        "noun": infer_noun_features,
        "adjective": infer_adjective_features,
        "verb": infer_verb_features,
        "pronoun": infer_pronoun_features,
    }
    extractor = strategies.get(part_of_speech, lambda _: {})
    return extractor(token.lower())


def infer_noun_features(token):
    """Infer noun case/number/gender from common inflectional endings."""
    features = {
        "case": None,
        "number": None,
        "gender": infer_noun_gender(token),
    }

    if token.endswith(("ами", "ями")):
        features.update({"case": "instrumental", "number": "plural"})
    elif token.endswith(("ах", "ях")):
        features.update({"case": "prepositional", "number": "plural"})
    elif token.endswith(("ов", "ев", "ей", "ий")):
        features.update({"case": "genitive", "number": "plural"})
    elif token.endswith(("у", "ю")):
        features.update({"case": "accusative", "number": "singular"})
    elif token.endswith(("е", "и")):
        # These endings are highly ambiguous across case/number in Russian.
        features.update({"case": None, "number": None})
    elif token.endswith(("а", "я")):
        # These endings are often nominative singular but not reliably so.
        features.update({"case": None, "number": "singular"})
    else:
        features.update({"case": "nominative", "number": "singular"})

    return features


def infer_adjective_features(token):
    """Infer adjective agreement features from suffix patterns."""
    features = {
        "case": None,
        "number": None,
        "gender": None,
    }

    if token.endswith(("ого", "его")):
        features.update({"case": "genitive", "number": "singular", "gender": "masculine"})
    elif token.endswith(("ому", "ему")):
        features.update({"case": "dative", "number": "singular", "gender": "masculine"})
    elif token.endswith(("ыми", "ими")):
        features.update({"case": "instrumental", "number": "plural"})
    elif token.endswith(("ые", "ие")):
        features.update({"case": "nominative", "number": "plural"})
    elif token.endswith(("ую", "юю")):
        features.update({"case": "accusative", "number": "singular", "gender": "feminine"})
    elif token.endswith(("ая", "яя")):
        features.update({"case": "nominative", "number": "singular", "gender": "feminine"})
    elif token.endswith(("ое", "ее")):
        features.update({"case": "nominative", "number": "singular", "gender": "neuter"})
    elif token.endswith(("ый", "ий", "ой")):
        features.update({"case": "nominative", "number": "singular", "gender": "masculine"})

    return features


def infer_verb_features(token):
    """Infer verb tense/person/number/aspect via ending-based heuristics."""
    features = {
        "tense": None,
        "aspect": None,
        "number": None,
        "person": None,
    }

    if token.endswith(("ю", "у")):
        features.update({"tense": "present", "person": "first", "number": "singular"})
    elif token.endswith(("ешь", "ишь")):
        features.update({"tense": "present", "person": "second", "number": "singular"})
    elif token.endswith(("ет", "ит")):
        features.update({"tense": "present", "person": "third", "number": "singular"})
    elif token.endswith(("ем", "им")):
        features.update({"tense": "present", "person": "first", "number": "plural"})
    elif token.endswith(("ете", "ите")):
        features.update({"tense": "present", "person": "second", "number": "plural"})
    elif token.endswith(("ут", "ют", "ат", "ят")):
        features.update({"tense": "present", "person": "third", "number": "plural"})
    elif token.endswith(("л", "ла", "ло", "ли")):
        features.update({"tense": "past"})

    if token.endswith(("ывать", "ивать")):
        features["aspect"] = "imperfective"
    elif token.endswith(("нуть", "нул", "нула", "нуло", "нули", "ну")):
        features["aspect"] = "perfective"

    if features["tense"] == "present" and features["aspect"] == "perfective":
        # In standard school grammar/NLP labeling, perfective non-past forms are future.
        features["tense"] = "future"

    return features


def infer_pronoun_features(token):
    """Return person/number/gender features for personal pronouns."""
    mapping = {
        "я": {"person": "first", "number": "singular"},
        "ты": {"person": "second", "number": "singular"},
        "он": {"person": "third", "number": "singular", "gender": "masculine"},
        "она": {"person": "third", "number": "singular", "gender": "feminine"},
        "оно": {"person": "third", "number": "singular", "gender": "neuter"},
        "мы": {"person": "first", "number": "plural"},
        "вы": {"person": "second", "number": "plural"},
        "они": {"person": "third", "number": "plural"},
    }
    return mapping.get(token, {})


def infer_noun_gender(token):
    """Infer noun gender from final character heuristics."""
    if token.endswith(("а", "я")):
        return "feminine"
    if token.endswith(("о", "е")):
        return "neuter"
    return "masculine"