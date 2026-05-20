"""Lemmatization helpers with dictionary, pymorphy3, and rule-based fallback."""

from functools import lru_cache

try:
    from .morphology_rules import guess_part_of_speech
    from .utils import DATA_DIR, load_json_file, normalize_text
except ImportError:
    from morphology_rules import guess_part_of_speech
    from utils import DATA_DIR, load_json_file, normalize_text

try:
    import pymorphy3
except ImportError:
    pymorphy3 = None


DICTIONARY_PATH = DATA_DIR / "morphology_dictionary.json"
IRREGULAR_PATH = DATA_DIR / "irregular_forms.json"


@lru_cache(maxsize=1)
def load_dictionary():
    """Load the main morphology dictionary once per process."""
    return load_json_file(DICTIONARY_PATH)


@lru_cache(maxsize=1)
def load_irregular_forms():
    """Load irregular form overrides once per process."""
    return load_json_file(IRREGULAR_PATH)


@lru_cache(maxsize=1)
def get_morph_analyzer():
    """Return a cached pymorphy3 analyzer instance when available."""
    if pymorphy3 is None:
        return None
    return pymorphy3.MorphAnalyzer()


def lookup_entry(token):
    """Look up a token in irregular forms first, then regular dictionary."""
    lower = normalize_text(token)
    irregular = load_irregular_forms().get(lower)
    if irregular:
        return irregular
    return load_dictionary().get(lower)


def lemmatize_token(token, part_of_speech):
    """Return lemma and source label for one token."""
    lower = normalize_text(token)
    entry = lookup_entry(lower)
    if entry and entry.get("lemma"):
        return entry["lemma"], "dictionary"

    analyzer = get_morph_analyzer()
    if analyzer is not None:
        parse = analyzer.parse(lower)
        if parse:
            return parse[0].normal_form, "pymorphy3"

    inferred_pos = part_of_speech or guess_part_of_speech(lower)
    return rule_based_lemma(lower, inferred_pos), "rules"


def rule_based_lemma(token, part_of_speech):
    """Apply POS-specific suffix rules when no lexical resource is found."""
    if part_of_speech == "verb":
        return lemmatize_verb(token)
    if part_of_speech == "adjective":
        return lemmatize_adjective(token)
    if part_of_speech == "noun":
        return lemmatize_noun(token)
    return token


def lemmatize_noun(token):
    """Approximate noun lemmatization via common ending substitutions."""
    if token.endswith("ами"):
        return token[:-3] + "а"
    if token.endswith("ями"):
        return token[:-3] + "я"
    if token.endswith(("ах", "ях")):
        return token[:-2] + "а"
    if token.endswith(("ов", "ев", "ей")):
        return token[:-2]
    if token.endswith("и"):
        return token[:-1] + "а"
    return token


def lemmatize_adjective(token):
    """Approximate adjective lemma reconstruction from inflected endings."""
    if token.endswith("ого"):
        return token[:-3] + "ый"
    if token.endswith("его"):
        return token[:-3] + "ий"
    if token.endswith(("ому", "ыми", "ой", "ом", "ым")):
        return token[:-2] + "ый"
    if token.endswith(("ему", "ими", "ем", "им")):
        return token[:-2] + "ий"
    if token.endswith(("ая", "ую")):
        return token[:-2] + "ый"
    if token.endswith(("яя", "юю")):
        return token[:-2] + "ий"
    if token.endswith(("ые",)):
        return token[:-2] + "ый"
    if token.endswith(("ие",)):
        return token[:-2] + "ий"
    return token


def lemmatize_verb(token):
    """Approximate infinitive recovery for frequent finite/past verb forms."""
    if token.endswith("аю"):
        return token[:-2] + "ть"
    if token.endswith("яю"):
        return token[:-2] + "ть"
    if token.endswith(("ю", "у")):
        return token[:-1] + "ть"
    if token.endswith(("ешь", "ет", "ем", "ете", "ут", "ют")):
        return token[:-2] + "ть"
    if token.endswith(("ишь", "ит", "им", "ите", "ат", "ят")):
        return token[:-2] + "ить"
    if token.endswith("ла"):
        return token[:-2] + "ть"
    if token.endswith(("л", "ло", "ли")):
        return token[:-1] + "ть"
    return token