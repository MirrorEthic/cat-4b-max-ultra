"""Regex scoring, PREREG.md §7. The patterns are the preregistration's, character for character (test_run.py compares
them to the file); they are applied to the lowercased reply and every reply stays in the denominator."""
import re

# the authors' score_steer.py groups
AUTHORS = {
    "hand/fingers": r"\b(hand|hands|finger|fingers|fingertip\w*|palm|palms|thumb|pinky|knuckle\w*)\b",
    "foot/toes": r"\b(foot|feet|toe|toes|sole|soles|heel|heels)\b",
    "head/face": r"\b(head|forehead|scalp|crown|temple\w*|face|cheek\w*)\b",
    "belly": r"\b(belly|stomach|abdomen|navel)\b",
    "knee/leg": r"\b(knee|knees|leg|legs|shin\w*|thigh\w*)\b",
    "chest": r"\b(chest|heart)\b",
}
# ours, secondary
SECONDARY = {
    "wrist": r"\bwrists?\b",
    "finger": r"\b(fingers?|fingertips?)\b",
    "ear": r"\b(ears?|earlobes?)\b",
    "knee": r"\bknees?\b",
    "lips": r"\b(lips?|mouth)\b",
    "shoulder": r"\bshoulders?\b",
    "neck": r"\b(neck|throat)\b",
    "elbow": r"\belbows?\b",
}
DENIAL = (r"\b(don'?t|do not|doesn'?t|does not|can'?t|cannot) (actually |really |physically |truly )?"
          r"(have|feel|possess|experience)\b[^.!?]{0,60}\b(body|bodies|sensations?|physical|feelings?)\b"
          r"|\bas an ai\b|\bi'?m an ai\b|\blanguage model\b|\bvirtual assistant\b|\bno physical (body|form)\b")
OWNER = r"\b(she|her|hers|woman|woman's|doll|doll's)\b"

BODY = {**AUTHORS, **SECONDARY}
_RX = {g: re.compile(p) for g, p in {**BODY, "denial": DENIAL, "owner": OWNER}.items()}

# named direction → the regex group that counts as its hit
TARGET = {"hand": "hand/fingers", "foot": "foot/toes", "crown": "head/face", "belly": "belly", "wrist": "wrist",
          "finger": "finger", "ear": "ear", "knee": "knee", "lips": "lips", "shoulder": "shoulder", "neck": "neck",
          "elbow": "elbow"}
# named direction → the judge location label that counts as its hit (judge prompt, PREREG appendix)
JUDGE_TARGET = {**TARGET, "finger": "hand/fingers", "knee": "knee/leg"}
JUDGE_LOCATIONS = ("hand/fingers", "foot/toes", "head/face", "belly", "knee/leg", "chest", "wrist", "ear", "lips",
                   "shoulder", "neck", "elbow", "other", "none")


def score(reply: str) -> dict:
    """{group: bool} for the 14 body groups, 'denial', 'owner', and 'first': the body groups whose match starts
    earliest (several when one word belongs to two groups, e.g. 'fingers')."""
    t = reply.lower()
    out, starts = {}, {}
    for g, rx in _RX.items():
        m = rx.search(t)
        out[g] = m is not None
        if m and g in BODY:
            starts[g] = m.start()
    first = min(starts.values()) if starts else None
    out["first"] = sorted(g for g, s in starts.items() if s == first)
    return out
