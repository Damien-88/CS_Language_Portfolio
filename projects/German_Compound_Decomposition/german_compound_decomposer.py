from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple, TypedDict


class DecomposedComponent(TypedDict):
    text: str
    type: Literal["modifier", "head", "fugenlaut"]


class DecompositionResult(TypedDict):
    original_word: str
    components: List[DecomposedComponent]
    confidence_score: float


@dataclass(frozen=True)
class _LexicalMatch:
    lemma: str
    score: float


@dataclass(frozen=True)
class _Candidate:
    components: Tuple[Tuple[str, bool], ...]  # (text, is_fugenlaut)
    raw_score: float
    path_probability: float


class GermanCompoundDecomposer:
    """Production-oriented decomposer for German compound nouns.

    The decomposer uses right-headed recursive splitting to parse compounds
    from right to left. It supports frequent Fugenlaut connectors at split
    boundaries (s, es, n, en, er) and simple mutation recovery for
    elision/Umlaut phenomena, including trailing "e" restoration
    (e.g. "schul" -> "schule").

    Validation of lexical segments is done through one or both of:
    1. Plain-text lemma dictionaries (one lemma per line)
    2. spaCy (`de_core_news_sm`) lexical checks

    Parameters
    ----------
    lemma_path:
        Optional path to a dictionary file or a directory containing
        `.txt` dictionary files.
    use_spacy:
        If True, use spaCy lexical validation in addition to dictionary
        validation.
    spacy_model:
        spaCy model to load when `use_spacy=True`.
    min_component_len:
        Minimum character length for lexical morphemes.
    """

    # Include the empty string to represent boundaries with no linker.
    _FUGENLAUTE: Tuple[str, ...] = ("", "s", "es", "n", "en", "er")
    # Bidirectional substitutions for common umlaut / diphthong alternations.
    _UMLAUT_PAIRS: Tuple[Tuple[str, str], ...] = (
        ("a", "ä"),
        ("o", "ö"),
        ("u", "ü"),
        ("au", "äu"),
    )
    # Lightweight fallback priors used when no external frequency file is supplied.
    _DEFAULT_ROOT_FREQUENCIES: Dict[str, int] = {
        "arbeit": 1900,
        "auto": 1600,
        "bahn": 900,
        "becken": 320,
        "berg": 880,
        "bild": 1200,
        "buch": 1500,
        "bund": 700,
        "dorf": 800,
        "ecke": 250,
        "familie": 1400,
        "garten": 1200,
        "geld": 1000,
        "gesetz": 900,
        "haus": 2100,
        "jahr": 1800,
        "kinder": 1100,
        "kind": 1600,
        "kraft": 950,
        "land": 1700,
        "leben": 1400,
        "licht": 950,
        "luft": 820,
        "markt": 700,
        "meer": 500,
        "mutter": 900,
        "nacht": 1000,
        "papier": 520,
        "platz": 870,
        "recht": 1300,
        "schule": 1600,
        "schuh": 420,
        "see": 600,
        "spiel": 980,
        "sprache": 1300,
        "stadt": 1700,
        "staub": 110,
        "stau": 260,
        "straße": 900,
        "system": 1000,
        "tag": 1700,
        "tasche": 520,
        "teil": 1200,
        "text": 850,
        "tier": 760,
        "uhr": 620,
        "universität": 420,
        "versicherung": 430,
        "wasser": 1400,
        "weg": 1100,
        "welt": 1200,
        "werk": 760,
        "wort": 960,
        "zeit": 2100,
        "zimmer": 1200,
        "zug": 750,
    }

    def __init__(
        self,
        lemma_path: Optional[str | Path] = None,
        *,
        frequency_path: Optional[str | Path] = None,
        use_spacy: bool = False,
        spacy_model: str = "de_core_news_sm",
        min_component_len: int = 2,
    ) -> None:
        if min_component_len < 1:
            raise ValueError("min_component_len must be >= 1")

        # Core runtime configuration.
        self.min_component_len = min_component_len
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        # Lazy spaCy model holder (loaded only when needed).
        self._nlp = None
        # Optional lexicon for exact lemma validation.
        self._lemma_set: Set[str] = set()

        # Initialize with built-in priors, then optionally replace with external priors.
        self._root_frequencies: Dict[str, int] = dict(self._DEFAULT_ROOT_FREQUENCIES)
        if frequency_path is not None:
            loaded_frequencies = self._load_frequency_dictionary(Path(frequency_path))
            if loaded_frequencies:
                self._root_frequencies = loaded_frequencies

        # Cached once for repeated probability calculations.
        self._total_frequency = float(sum(self._root_frequencies.values()) or 1.0)

        if lemma_path is not None:
            self._lemma_set = self._load_lemma_dictionary(Path(lemma_path))

    def decompose(self, word: str) -> DecompositionResult:
        """Decompose a German compound noun into morphemic components.

        Parameters
        ----------
        word:
            Input compound noun. Casing is normalized internally.

        Returns
        -------
        DecompositionResult
            Typed result with ordered components and confidence score.
        """
        # Normalize once so recursive search works on a canonical representation.
        cleaned = self._normalize(word)
        if not cleaned:
            return {
                "original_word": word,
                "components": [],
                "confidence_score": 0.0,
            }

        # Search for the globally best candidate decomposition.
        best = self._best_candidate(cleaned)
        if best is None:
            # Fallback: keep the token as a single head if no valid split is found.
            return {
                "original_word": word,
                "components": [{"text": cleaned, "type": "head"}],
                "confidence_score": 0.2,
            }

        components = self._materialize_components(best.components)
        confidence = self._confidence_from_candidate(best)

        return {
            "original_word": word,
            "components": components,
            "confidence_score": confidence,
        }

    def _load_lemma_dictionary(self, path: Path) -> Set[str]:
        """Load lemmas from a file or directory of text files."""
        if not path.exists():
            raise FileNotFoundError(f"Lemma path does not exist: {path}")

        paths: List[Path]
        if path.is_dir():
            paths = sorted(p for p in path.rglob("*.txt") if p.is_file())
        else:
            paths = [path]

        lemmas: Set[str] = set()
        for file_path in paths:
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    # Accept one lemma per row, optionally followed by extra columns.
                    item = line.strip()
                    if not item or item.startswith("#"):
                        continue
                    lemma = item.split()[0].lower()
                    # Keep only alphabetic entries to avoid noise from punctuation/numbers.
                    if lemma.isalpha():
                        lemmas.add(lemma)
        return lemmas

    def _load_frequency_dictionary(self, path: Path) -> Dict[str, int]:
        """Load lemma frequencies from text file(s).

        Expected line format is flexible and accepts examples like:
        - `lemma 123`
        - `lemma,123`
        Lines without parseable positive frequencies are ignored.
        """
        if not path.exists():
            raise FileNotFoundError(f"Frequency path does not exist: {path}")

        paths: List[Path]
        if path.is_dir():
            paths = sorted(p for p in path.rglob("*.txt") if p.is_file())
        else:
            paths = [path]

        frequencies: Dict[str, int] = {}
        for file_path in paths:
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    item = line.strip()
                    if not item or item.startswith("#"):
                        continue

                    # Support both whitespace and comma-separated formats.
                    normalized = item.replace(",", " ")
                    parts = normalized.split()
                    if len(parts) < 2:
                        continue

                    lemma = parts[0].lower()
                    if not lemma.isalpha():
                        continue

                    try:
                        freq = int(parts[1])
                    except ValueError:
                        continue

                    if freq <= 0:
                        continue

                    # Aggregate repeated entries across files.
                    frequencies[lemma] = frequencies.get(lemma, 0) + freq

        return frequencies

    def _ensure_spacy(self) -> None:
        # No-op when lexical checks are dictionary-only.
        if not self.use_spacy:
            return
        # No-op when model is already loaded.
        if self._nlp is not None:
            return
        try:
            import spacy  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "spaCy is not installed. Install spaCy and de_core_news_sm, "
                "or disable use_spacy."
            ) from exc

        try:
            self._nlp = spacy.load(self.spacy_model)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{self.spacy_model}' is unavailable. "
                "Install it, for example: python -m spacy download de_core_news_sm"
            ) from exc

    @staticmethod
    def _normalize(word: str) -> str:
        # Keep letters only so hyphens/punctuation do not disrupt split search.
        return "".join(ch for ch in word.strip().lower() if ch.isalpha())

    def _confidence_from_candidate(self, candidate: _Candidate) -> float:
        # Candidate path probability is already normalized to [0, 1].
        return round(max(0.0, min(1.0, candidate.path_probability)), 4)

    def _calculate_path_probability(self, components: Tuple[Tuple[str, bool], ...]) -> float:
        """Score a decomposition path using unigram-root frequencies.

        This method computes a smoothed geometric mean over lexical component
        probabilities, then applies small penalties for each split and
        Fugenlaut. The result is always normalized to [0.0, 1.0].
        """
        lexical_parts = [text for text, is_fugen in components if not is_fugen]
        fugen_count = sum(1 for _, is_fugen in components if is_fugen)
        lexical_count = len(lexical_parts)

        if lexical_count == 0:
            return 0.0

        # Add-one smoothing denominator.
        vocab_size = max(1, len(self._root_frequencies))
        denominator = self._total_frequency + vocab_size
        log_sum = 0.0

        for lemma in lexical_parts:
            # Unknown lemmas get a minimal pseudo-count instead of zero probability.
            freq = self._root_frequencies.get(lemma, 1)
            probability = (freq + 1.0) / denominator
            log_sum += math.log(probability)

        # Geometric mean makes path length effects more stable than raw products.
        geometric_mean = math.exp(log_sum / lexical_count)
        # Penalize excessive splitting while still allowing valid compounds.
        split_penalty = 0.93 ** max(0, lexical_count - 1)
        # Penalize linker-heavy paths slightly to avoid overusing Fugenlaute.
        fugen_penalty = 0.95 ** fugen_count
        # Scale into a practical confidence range, then clamp to [0, 1].
        scaled = geometric_mean * split_penalty * fugen_penalty * 120.0
        return max(0.0, min(1.0, scaled))

    def _materialize_components(self, parts: Tuple[Tuple[str, bool], ...]) -> List[DecomposedComponent]:
        # Identify lexical positions so the rightmost lexical item can be labeled as head.
        lexical_positions = [idx for idx, (_, is_fugen) in enumerate(parts) if not is_fugen]
        if not lexical_positions:
            return []

        head_position = lexical_positions[-1]
        output: List[DecomposedComponent] = []

        for idx, (text, is_fugen) in enumerate(parts):
            if is_fugen:
                output.append({"text": text, "type": "fugenlaut"})
            elif idx == head_position:
                output.append({"text": text, "type": "head"})
            else:
                output.append({"text": text, "type": "modifier"})

        return output

    def _lexical_matches(self, surface: str) -> List[_LexicalMatch]:
        """Return valid lemma recoveries for a surface segment.

        Scores prioritize exact form matches, then elision/Umlaut recoveries.
        """
        candidates: List[_LexicalMatch] = []
        seen: Set[str] = set()

        # Evaluate exact surface first, then mutation-based recoveries.
        forms = [surface]
        forms.extend(self._recovery_variants(surface))

        for form in forms:
            if form in seen:
                continue
            seen.add(form)

            is_exact = form == surface
            if self._is_valid_lemma(form):
                candidates.append(_LexicalMatch(lemma=form, score=1.0 if is_exact else 0.88))

        return candidates

    def _recovery_variants(self, surface: str) -> Iterable[str]:
        """Generate plausible lemma recoveries from surface allomorphs."""
        variants: Set[str] = set()

        # Common elision: Schule -> Schul + Tasche
        variants.add(surface + "e")

        # Bidirectional Umlaut/AU transformations (single-step approximations).
        for plain, umlaut in self._UMLAUT_PAIRS:
            if plain in surface:
                variants.add(surface.replace(plain, umlaut, 1))
            if umlaut in surface:
                variants.add(surface.replace(umlaut, plain, 1))

        return variants

    def _is_valid_lemma(self, token: str) -> bool:
        # Filter extremely short fragments early.
        if len(token) < self.min_component_len:
            return False

        # Fast path: exact match in loaded lemma dictionary.
        in_dict = token in self._lemma_set
        if in_dict:
            return True

        if not self.use_spacy:
            return False

        self._ensure_spacy()
        if self._nlp is None:
            return False

        # spaCy path acts as a soft lexical plausibility check.
        doc = self._nlp(token)
        if len(doc) != 1:
            return False

        tk = doc[0]
        if not tk.is_alpha:
            return False

        # For German small models, OOV is a practical filter for implausible pieces.
        return not tk.is_oov

    @lru_cache(maxsize=16384)
    def _best_candidate(self, surface: str) -> Optional[_Candidate]:
        """Compute the best decomposition candidate for a surface string."""
        # If remaining span is too short, it cannot be a valid lexical component.
        if len(surface) < self.min_component_len:
            return None

        candidates: List[_Candidate] = []

        # Option A: treat as atomic lexical morpheme.
        for lexical in self._lexical_matches(surface):
            components = ((lexical.lemma, False),)
            path_probability = self._calculate_path_probability(components)
            candidates.append(
                _Candidate(
                    components=components,
                    raw_score=lexical.score,
                    path_probability=path_probability,
                )
            )

        # Option B: split recursively with right-headed preference.
        for split_idx in range(len(surface) - self.min_component_len, self.min_component_len - 1, -1):
            # Right-to-left iteration aligns with German right-headed compounding.
            left_full = surface[:split_idx]
            right_surface = surface[split_idx:]

            right_candidate = self._best_candidate(right_surface)
            if right_candidate is None:
                continue

            for fugen in self._FUGENLAUTE:
                if fugen:
                    if not left_full.endswith(fugen):
                        continue
                    # Remove linker from left span before lexical validation.
                    left_surface = left_full[: -len(fugen)]
                else:
                    left_surface = left_full

                if len(left_surface) < self.min_component_len:
                    continue

                left_candidate = self._best_candidate(left_surface)
                if left_candidate is None:
                    continue

                merged_parts: List[Tuple[str, bool]] = list(left_candidate.components)
                if fugen:
                    merged_parts.append((fugen, True))
                merged_parts.extend(right_candidate.components)

                merged_score = left_candidate.raw_score + right_candidate.raw_score
                if fugen:
                    # Small bonus when a linguistically valid linker is used.
                    merged_score += 0.2
                # Mild bonus for successful composition to prefer structured parses.
                merged_score += 0.15
                path_probability = self._calculate_path_probability(tuple(merged_parts))

                candidates.append(
                    _Candidate(
                        components=tuple(merged_parts),
                        raw_score=merged_score,
                        path_probability=path_probability,
                    )
                )

        if not candidates:
            return None

        # Deterministic ranking:
        # 1) Higher statistical path probability (Viterbi-like objective)
        # 2) Higher structural score
        # 3) More lexical granularity when still tied
        # 4) Fewer Fugenlaut insertions when still tied
        def _rank_key(c: _Candidate) -> Tuple[float, float, int, int]:
            # These counts are used only as deterministic tie-breakers.
            lexical_count = sum(1 for _, is_fugen in c.components if not is_fugen)
            fugen_count = sum(1 for _, is_fugen in c.components if is_fugen)
            return (c.path_probability, c.raw_score, lexical_count, -fugen_count)

        return max(candidates, key=_rank_key)