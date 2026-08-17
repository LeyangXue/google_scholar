"""
assign_field.py
---------------
Rule for assigning a single top-level FIELD to a paper from its OpenAlex concepts.

Background
==========
OpenAlex concepts are hierarchical (level 0-5). Only the 19 level-0 concepts are
"fields" (physics, chemistry, medicine, ...). Each paper's concept record is a
dict {concept_name: score}. That dict already contains the paper's level-0
ancestor field(s), alongside finer-grained concepts.

Criterion (argmax over level-0 roots)
=====================================
FieldOf(paper) = argmax_{ f in ROOT_FIELDS, score(f) > 0 } score(f)

Steps:
  1. Keep only concepts whose name is one of the 19 level-0 ROOT_FIELDS
     AND whose score is strictly > 0.
     (Ancestors are sometimes attached with score exactly 0.0 -> excluded.)
  2. Among those, take the field with the highest score.
  3. If no root field with score > 0 exists -> return None (unmatched).

Example
-------
concepts = {"data mining": 0.76, "computer science": 0.72, "geography": 0.12,
            "mathematics": 0.0}
  -> root fields with score>0: {"computer science": 0.72, "geography": 0.12}
  -> FieldOf = "computer science"   (mathematics dropped: score == 0.0)

Notes / knobs
-------------
- Ties / near-ties are broken by max(); if two fields are very close
  (e.g. 0.35 vs 0.34) the label can be sensitive. To harden, either keep the
  top-2 fields, or require the top field to lead the second by a margin.
- MIN_SCORE (default 0.0, i.e. strictly > 0) can be raised (e.g. 0.1) to drop
  weak/peripheral field attributions.
"""

# The 19 OpenAlex level-0 root concepts (= fields).
ROOT_FIELDS = {
    "art", "biology", "business", "chemistry", "computer science", "economics",
    "engineering", "environmental science", "geography", "geology", "history",
    "materials science", "mathematics", "medicine", "philosophy", "physics",
    "political science", "psychology", "sociology",
}


def field_of(concepts, min_score=0.0):
    """Return the single top-level field for a paper, or None if unassignable.

    Parameters
    ----------
    concepts : dict[str, float]
        {concept_name: score} for one paper (as stored in the
        year_{citing,cited}paper_concept pickles).
    min_score : float
        Root fields must have score > min_score to be considered (default 0.0).

    Returns
    -------
    str | None
    """
    roots = {k: v for k, v in concepts.items() if k in ROOT_FIELDS and v > min_score}
    if not roots:
        return None
    return max(roots, key=roots.get)


def build_field_map(concept_dict, min_score=0.0):
    """Reduce a concept pickle {year: {paper_id: {concept: score}}} to a flat
    {str(paper_id): field} mapping. Papers with no assignable field are omitted."""
    fm = {}
    for _year, inner in concept_dict.items():
        for pid, concepts in inner.items():
            f = field_of(concepts, min_score=min_score)
            if f is not None:
                fm[str(pid)] = f
    return fm
