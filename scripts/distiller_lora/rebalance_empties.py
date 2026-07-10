"""v2 data fix: the v1 tuned distiller fabricates a fact from every question/chitchat
message (abstention was only 6% of train). Add ~130 more empty-target examples (varied
questions + chitchat) so 'extract nothing' is well represented. Empty targets need NO
cloud labeling. Appends to train.jsonl / val.jsonl in place (writes *_v2 copies)."""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(7)
DATA = Path(__file__).resolve().parent / "data"

ATTRS = ["live", "work", "drive", "job title", "phone", "gym", "allergy", "birthday",
         "partner", "hometown", "favorite color", "coffee order", "pet", "diet",
         "employer", "hobby", "manager", "address", "email", "start date"]
Q_TEMPLATES = [
    "Where do I {a}?", "What's my {a}?", "Can you remind me of my {a}?",
    "Do you remember my {a}?", "What did I say my {a} was?", "Remind me, what's my {a} again?",
    "Tell me my {a}.", "What was my {a}?", "Which {a} did I mention?", "Any idea what my {a} is?",
]
Q_FILL = {"live": "live", "work": "work", "drive": "car do I drive",
          "job title": "job title", "phone": "phone", "gym": "gym", "allergy": "allergy",
          "birthday": "birthday", "partner": "partner's name", "hometown": "hometown",
          "favorite color": "favorite color", "coffee order": "usual coffee order",
          "pet": "pet", "diet": "diet", "employer": "employer", "hobby": "hobby",
          "manager": "manager", "address": "address", "email": "email",
          "start date": "start date"}
CHITCHAT = [
    "The weather has been lovely lately.", "Ugh, traffic was brutal today.",
    "I can't wait for the weekend.", "That show everyone's talking about is pretty good.",
    "It's been a long week already.", "The coffee here is surprisingly good.",
    "Did you catch the game last night?", "Mondays, am I right?",
    "It's raining again, of course.", "I'm so ready for a vacation.",
    "Just thinking out loud here.", "Anyway, how are you doing?",
    "That's kind of funny when you think about it.", "Not much going on today.",
    "The office was quiet this morning.", "I love this time of year.",
    "Feeling a bit tired today.", "Hope your week is going well.",
    "The commute wasn't too bad today.", "Some days just fly by.",
    "What a beautiful sunset earlier.", "I could really go for some pizza.",
    "The wifi has been spotty lately.", "Nothing beats a lazy Sunday.",
]
FOLLOWUPS = [  # conversational filler / meta that must not produce facts
    "Thanks, that's helpful!", "Got it, makes sense.", "Okay cool.",
    "Hmm, let me think about that.", "Sounds good to me.", "Sure, why not.",
    "I appreciate it.", "No worries.", "Interesting, tell me more.", "Right, exactly.",
]


def gen():
    out = []
    for a in ATTRS:
        for t in random.sample(Q_TEMPLATES, 4):
            out.append(("question", t.format(a=Q_FILL[a])))
    for c in CHITCHAT + FOLLOWUPS:
        out.append(("chitchat", c))
    return out


def main():
    rows = gen()
    seen = set(json.loads(l)["text"] for l in (DATA / "train.jsonl").open())
    added_t, added_v = [], []
    for i, (src, text) in enumerate(rows):
        if text in seen:
            continue
        seen.add(text)
        rec = {"source": f"empty_{src}", "text": text, "target": {"facts": []}}
        (added_v if i % 10 == 0 else added_t).append(rec)
    with (DATA / "train.jsonl").open("a") as fh:
        for r in added_t:
            fh.write(json.dumps(r) + "\n")
    with (DATA / "val.jsonl").open("a") as fh:
        for r in added_v:
            fh.write(json.dumps(r) + "\n")
    print(f"added {len(added_t)} train + {len(added_v)} val empty-target examples")
    nt = sum(1 for _ in (DATA / "train.jsonl").open())
    ne = sum(1 for l in (DATA / "train.jsonl").open() if not json.loads(l)["target"]["facts"])
    print(f"train now {nt} rows, {ne} empty-target ({100*ne/nt:.0f}%)")


if __name__ == "__main__":
    main()
