"""Core data schemas shared by every PolyglotSound engine."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class Tone(str, Enum):
    HIGH = "H"
    LOW = "L"
    MID = "M"
    RISING = "LH"
    FALLING = "HL"
    UNMARKED = "?"


class Syllable(BaseModel):
    text: str
    nucleus: str = ""
    stressed: bool = False
    optional: bool = False          # e.g. French final schwa, may be sung or not
    tone: Optional[Tone] = None     # tonal languages only
    open: bool = True               # ends with a vowel (good for sustained notes)
    start: Optional[float] = None
    end: Optional[float] = None


class Word(BaseModel):
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    probability: Optional[float] = None


class MelodySlot(BaseModel):
    """One sung note position in the original performance."""
    start: float
    end: float
    midi: Optional[float] = None     # median pitch, None if unvoiced/unknown
    beat_strength: float = 0.5       # 1.0 downbeat, 0.6 beat, 0.3 off-beat

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class LyricLine(BaseModel):
    index: int
    text: str
    language: str
    start: Optional[float] = None
    end: Optional[float] = None
    words: list[Word] = Field(default_factory=list)
    confidence: Confidence = Confidence.UNKNOWN
    human_verified: bool = False


class LineTemplate(BaseModel):
    """Structured musical representation of one original line."""
    line: LyricLine
    syllables: list[Syllable]
    slots: list[MelodySlot]
    rhyme_key: str = ""
    rhyme_group: Optional[str] = None      # e.g. "A", "B" from the section's rhyme scheme
    timing_confidence: Confidence = Confidence.UNKNOWN
    notes: list[str] = Field(default_factory=list)

    @property
    def syllable_count(self) -> int:
        return len(self.slots) if self.slots else len(self.syllables)


class CriterionScore(BaseModel):
    name: str
    score: Optional[float]           # 0..1, None = not measurable
    weight: float
    explanation: str
    reliable: bool = True            # False = heuristic / needs human check


class AdaptationCandidate(BaseModel):
    text: str
    language: str
    source: str                      # "nllb", "nllb-nbest", "llm:<model>", "manual"
    syllables: list[Syllable] = Field(default_factory=list)
    criteria: list[CriterionScore] = Field(default_factory=list)
    total: Optional[float] = None
    rejected: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    needs_human_review: list[str] = Field(default_factory=list)
    alignment: list[list[int]] = Field(default_factory=list)  # slot indexes per syllable
