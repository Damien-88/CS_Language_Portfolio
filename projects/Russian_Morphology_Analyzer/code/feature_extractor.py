"""Feature extraction and tag normalization for Russian morphology analysis."""

try:
    from .utils import normalize_text
    from .tokenizer import is_word
    from .lemmatizer import get_morph_analyzer, lemmatize_token, lookup_entry
    from .morphology_rules import guess_part_of_speech, infer_features
except ImportError:
    from utils import normalize_text
    from tokenizer import is_word
    from lemmatizer import get_morph_analyzer, lemmatize_token, lookup_entry
    from morphology_rules import guess_part_of_speech, infer_features

POS_MAP = {
    "NOUN": "noun",
    "ADJF": "adjective",
    "ADJS": "adjective",
    "COMP": "adjective",
    "VERB": "verb",
    "INFN": "verb",
    "PRTF": "participle",
    "PRTS": "participle",
    "GRND": "gerund",
    "NUMR": "numeral",
    "ADVB": "adverb",
    "NPRO": "pronoun",
    "PREP": "preposition",
    "CONJ": "conjunction",
    "PRCL": "particle",
    "INTJ": "interjection",
}

CASE_MAP = {
    "nomn": "nominative",
    "gent": "genitive",
    "gen2": "genitive",
    "datv": "dative",
    "accs": "accusative",
    "acc2": "accusative",
    "ablt": "instrumental",
    "loct": "prepositional",
    "loc2": "prepositional",
    "voct": "vocative",
}

GENDER_MAP = {
    "masc": "masculine",
    "femn": "feminine",
    "neut": "neuter",
}

NUMBER_MAP = {
    "sing": "singular",
    "plur": "plural",
}

TENSE_MAP = {
    "pres": "present",
    "past": "past",
    "futr": "future",
}

PERSON_MAP = {
    "1per": "first",
    "2per": "second",
    "3per": "third",
}

ASPECT_MAP = {
    "impf": "imperfective",
    "perf": "perfective",
}

ANIMACY_MAP = {
    "anim": "animate",
    "inan": "inanimate",
}

MOOD_MAP = {
    "indc": "indicative",
    "impr": "imperative",
}

VOICE_MAP = {
    "actv": "active",
    "pssv": "passive",
}

TRANSITIVITY_MAP = {
    "tran": "transitive",
    "intr": "intransitive",
}

INVOLVEMENT_MAP = {
    "incl": "inclusive",
    "excl": "exclusive",
}


def analyze_token(token):
    """Analyze one token using dictionary, pymorphy3, then rule fallback."""
    if not is_word(token):
        return {
            "token": token,
            "lemma": token,
            "part_of_speech": "punctuation",
        }

    lower = normalize_text(token)

    entry = lookup_entry(lower)
    if entry:
        return build_entry_analysis(token, entry)

    analyzer = get_morph_analyzer()
    if analyzer is not None:
        parses = analyzer.parse(lower)
        if parses:
            # Use highest-confidence parse as primary analysis.
            return build_pymorphy_analysis(token, parses[0], parses[:3])

    part_of_speech = guess_part_of_speech(lower)
    lemma, source = lemmatize_token(lower, part_of_speech)
    features = infer_features(lower, part_of_speech)

    return {
        "token": token,
        "lemma": lemma,
        "part_of_speech": part_of_speech,
        **features,
        "analysis_source": source,
    }


def build_entry_analysis(token, entry):
    """Build a normalized output object from dictionary/irregular entries."""
    analysis = {
        "token": token,
        "lemma": entry.get("lemma", normalize_text(token)),
        "part_of_speech": entry.get("part_of_speech", guess_part_of_speech(token)),
        "case": entry.get("case"),
        "gender": entry.get("gender"),
        "number": entry.get("number"),
        "tense": entry.get("tense"),
        "person": entry.get("person"),
        "aspect": entry.get("aspect"),
        "animacy": entry.get("animacy"),
        "analysis_source": entry.get("analysis_source", "dictionary"),
    }

    # Preserve any additional lexicon annotations without overriding normalized fields.
    for key, value in entry.items():
        if key not in analysis:
            analysis[key] = value

    return analysis


def build_pymorphy_analysis(token, parse, alternatives=None):
    """Build normalized output from one pymorphy3 parse result."""
    tag_data = _extract_tag_data(parse.tag)
    score = getattr(parse, "score", None)

    tags = parse.tag
    analysis = {
        "token": token,
        "lemma": parse.normal_form,
        "part_of_speech": POS_MAP.get(tags.POS, "unknown"),
        **tag_data,
        "analysis_source": "pymorphy3",
    }

    if score is not None:
        analysis["confidence"] = round(float(score), 4)

    known_flag = getattr(parse, "is_known", None)
    if known_flag is not None:
        analysis["is_known_word"] = bool(known_flag)

    # Include top parse candidates to expose ambiguity in morphologically rich tokens.
    if alternatives:
        analysis["candidates"] = [_build_candidate(candidate) for candidate in alternatives]

    return analysis


def _extract_tag_data(tags):
    """Map pymorphy3 tag codes to readable feature labels."""
    return {
        "case": CASE_MAP.get(tags.case),
        "gender": GENDER_MAP.get(tags.gender),
        "number": NUMBER_MAP.get(tags.number),
        "tense": TENSE_MAP.get(tags.tense),
        "person": PERSON_MAP.get(tags.person),
        "aspect": ASPECT_MAP.get(tags.aspect),
        "animacy": ANIMACY_MAP.get(getattr(tags, "animacy", None)),
        "mood": MOOD_MAP.get(getattr(tags, "mood", None)),
        "voice": VOICE_MAP.get(getattr(tags, "voice", None)),
        "transitivity": TRANSITIVITY_MAP.get(getattr(tags, "transitivity", None)),
        "involvement": INVOLVEMENT_MAP.get(getattr(tags, "involvement", None)),
        "raw_tag": str(tags),
    }


def _build_candidate(parse):
    """Serialize one alternative parse candidate for ambiguity inspection."""
    tags = parse.tag
    candidate = {
        "lemma": parse.normal_form,
        "part_of_speech": POS_MAP.get(tags.POS, "unknown"),
        **_extract_tag_data(tags),
    }

    score = getattr(parse, "score", None)
    if score is not None:
        candidate["confidence"] = round(float(score), 4)

    known_flag = getattr(parse, "is_known", None)
    if known_flag is not None:
        candidate["is_known_word"] = bool(known_flag)

    return candidate


def analyze_tokens(tokens):
    """Analyze a sequence of tokens in order."""
    return [analyze_token(token) for token in tokens]