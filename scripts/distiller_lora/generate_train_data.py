"""Stage 1 — build SFT training pairs (message -> facts JSON) for the local distiller.

Local ingestion caches are on a TCC-blocked external volume, so we synthesize instead
(the mission's sanctioned augmentation path): generate diverse conversational messages
from an attribute catalog where we KNOW the ground-truth canonical key, label the prose +
salience with the cheap cloud reference distiller (qwen3.6-flash), then OVERRIDE the key
with our canonical key. That guarantees key-consistent labels (the single most important
property — inconsistent keys are what break supersession) regardless of labeler jitter.

Cleaning: drop kv-pathology labels, drop empty-labeled fact messages, dedupe by text.
Questions + chitchat get an empty target (no cloud) to train away the fabrication failure.
Held-out val split is by TEMPLATE FAMILY, not random rows.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from harness import qwen_endpoint, raw_keyvalue_pathology

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
random.seed(1234)

_CITIES = ["Denver", "Austin", "Seattle", "Portland", "Chicago", "Boston", "Miami",
           "Nashville", "Denver", "Atlanta", "Raleigh", "Boise", "Madison", "Tucson"]
_JOBS = ["Data Scientist", "Product Manager", "Staff Engineer", "UX Designer",
         "Marketing Lead", "Solutions Architect", "Research Scientist", "DevOps Engineer"]
_CARS = ["Toyota Prius", "Subaru Forester", "Tesla Model 3", "Honda Accord", "Mazda3",
         "Ford Mustang", "Jaguar F-Pace", "Chevrolet Bolt", "Mini Cooper"]
_PHONES = ["iPhone 15", "Pixel 12", "OnePlus 12", "Galaxy S24", "Xperia 5", "iPhone 14"]
_GYMS = ["Equinox Downtown", "Gold's Gym Midtown", "Planet Fitness Westside",
         "LA Fitness Uptown", "CrossFit Central", "Life Time Eastside"]
_NAMES = ["Marcus", "Priya", "Elena", "Sam", "Dana", "Kwame", "Yuki", "Omar", "Lena"]
_COMPANIES = ["Stripe", "Figma", "Datadog", "Notion", "Snowflake", "Anthropic", "Shopify"]
_PETS = ["a golden retriever named Biscuit", "a tabby cat named Miso", "a beagle named Cooper",
         "a corgi named Pumpkin", "a parrot named Kiwi", "a labrador named Scout"]
_DATES = ["March 3rd", "July 12th", "January 8th", "October 21st", "May 5th", "December 1st"]
_ALLERGENS = ["peanuts", "shellfish", "gluten", "dairy", "tree nuts", "penicillin"]
_DIETS = ["vegetarian", "vegan", "pescatarian", "keto", "gluten-free", "dairy-free"]
_COFFEES = ["oat milk latte", "black cold brew", "cappuccino", "flat white", "espresso"]
_COLORS = ["teal", "burnt orange", "forest green", "navy", "maroon"]
_HOBBIES = ["rock climbing", "pottery", "trail running", "woodworking", "salsa dancing"]

# canonical_key -> (value pool, [templates using {v}])
CATALOG = {
    "user::residence": (_CITIES, ["I just moved to {v}.", "I now live in {v}.",
        "I relocated to {v} last month.", "My new city is {v}.", "I've settled in {v}."]),
    "user::job_title": (_JOBS, ["I got promoted, I'm now a {v}.", "My new role is {v}.",
        "I just became a {v}.", "They made me a {v}.", "I work as a {v} now."]),
    "user::car": (_CARS, ["I bought a new car, a {v}.", "Just picked up a {v}.",
        "I'm driving a {v} now.", "My new ride is a {v}.", "Switched to a {v}."]),
    "user::phone": (_PHONES, ["I switched phones, now using a {v}.", "Got a {v} yesterday.",
        "My new phone is a {v}.", "I'm on a {v} now.", "Upgraded to a {v}."]),
    "user::gym": (_GYMS, ["I changed gyms, I go to {v} now.", "Started at {v}.",
        "My gym is {v} these days.", "I work out at {v} now.", "Joined {v}."]),
    "user::name": (_NAMES, ["My name is {v}.", "I'm {v}.", "Everyone calls me {v}.",
        "You can call me {v}.", "Hi, I'm {v}."]),
    "user::employer": (_COMPANIES, ["I work at {v}.", "I just joined {v}.",
        "My employer is {v}.", "I landed a job at {v}.", "I'm now at {v}."]),
    "user::pet": (_PETS, ["I got {v}.", "My new pet is {v}.", "I adopted {v}.",
        "We just brought home {v}.", "Say hi to {v}, my new pet."]),
    "user::birthday": (_DATES, ["My birthday is {v}.", "I was born on {v}.",
        "{v} is my birthday.", "I celebrate my birthday on {v}.", "My birth date is {v}."]),
    "user::allergy": (_ALLERGENS, ["I'm allergic to {v}.", "I have a {v} allergy.",
        "{v} make me sick.", "I can't have {v}, I'm allergic.", "My allergy is {v}."]),
    "user::diet": (_DIETS, ["I'm {v}.", "I follow a {v} diet.", "My diet is {v}.",
        "I eat {v}.", "I've gone {v}."]),
    "user::coffee_pref": (_COFFEES, ["I always order a {v}.", "My go-to coffee is a {v}.",
        "I prefer a {v}.", "A {v} is my usual.", "I drink {v}s."]),
    "user::favorite_color": (_COLORS, ["My favorite color is {v}.", "I love {v}.",
        "{v} is my favorite color.", "I'm partial to {v}.", "Nothing beats {v}."]),
    "user::hobby": (_HOBBIES, ["I've taken up {v}.", "My new hobby is {v}.",
        "I'm really into {v} lately.", "I started {v} recently.", "I do {v} on weekends."]),
    "user::hometown": (_CITIES, ["I grew up in {v}.", "My hometown is {v}.",
        "I'm originally from {v}.", "I was raised in {v}.", "{v} is where I'm from."]),
    "user::partner": (_NAMES, ["My partner's name is {v}.", "I'm dating {v}.",
        "My partner is {v}.", "{v} and I are together.", "My significant other is {v}."]),
}

QUESTIONS = [
    "Where do I live?", "What's my job title?", "What car do I drive?",
    "Which gym do I go to?", "What phone do I use?", "What's my allergy?",
    "Do you remember my birthday?", "What's my partner's name?", "Where am I from?",
    "What's my favorite color?", "What do I usually order at the coffee shop?",
    "Can you tell me what pet I have?", "What company do I work for?",
    "What hobby did I pick up?", "Remind me of my diet?",
]
CHITCHAT = [
    "The weather has been really nice this week.", "Traffic was terrible this morning.",
    "I watched a great documentary last night.", "Ugh, Mondays.",
    "That coffee shop smells amazing.", "Can't believe it's almost the weekend.",
    "My plants finally started blooming.", "The bus was late again.",
    "I've been listening to a lot of podcasts.", "Work has been busy lately.",
    "I need to organize my closet.", "There's construction on my street.",
    "The new season of my show dropped.", "I read an interesting article today.",
    "My neighbor got a new dog, it's adorable.",
]


def gen_messages(n_val_templates=1):
    """Yield (source, key_or_None, text, held_out). Held-out = last template per attr."""
    for key, (pool, templates) in CATALOG.items():
        for ti, tmpl in enumerate(templates):
            held = ti >= len(templates) - n_val_templates
            for v in random.sample(pool, min(6, len(pool))):
                yield (key, key, tmpl.format(v=v), held)
    # multi-fact: pair two attributes in one message (train only)
    keys = list(CATALOG)
    for _ in range(90):
        k1, k2 = random.sample(keys, 2)
        v1 = random.choice(CATALOG[k1][0]); v2 = random.choice(CATALOG[k2][0])
        t1 = CATALOG[k1][1][0].format(v=v1).rstrip("."); t2 = CATALOG[k2][1][0].format(v=v2)
        yield ("multi", None, f"{t1}, and {t2}", False)
    for q in QUESTIONS:
        yield ("question", "__empty__", q, False)
    for c in CHITCHAT:
        yield ("chitchat", "__empty__", c, False)


def canon_facts(ref_facts, canonical_key):
    """Override the key of a single-attribute label with the known canonical key."""
    if not ref_facts:
        return None  # cloud emitted nothing for a fact message → drop (unreliable)
    # keep the most salient fact, force its key
    f = max(ref_facts, key=lambda x: x.salience)
    return [{"statement": f.statement, "key": canonical_key,
             "salience": f.salience, "valid_at": f.valid_at_iso}]


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ref = qwen_endpoint("qwen3.6-flash")
    rows = list(gen_messages())
    if limit:
        rows = rows[:limit]
    train, val = [], []
    seen = set()
    n_drop_kv = n_drop_empty = n_dup = 0
    for i, (source, key, text, held) in enumerate(rows):
        if text in seen:
            n_dup += 1
            continue
        seen.add(text)
        if key == "__empty__":
            target = {"facts": []}
        else:
            facts, raw = ref.distill(text)
            if raw_keyvalue_pathology(raw):
                n_drop_kv += 1
                continue
            if key is not None:  # single-attribute: canonicalize key
                cf = canon_facts(facts, key)
                if cf is None:
                    n_drop_empty += 1
                    continue
                target = {"facts": cf}
            else:  # multi-fact: trust reference keys but require ≥1 fact
                if not facts:
                    n_drop_empty += 1
                    continue
                target = {"facts": [{"statement": f.statement, "key": f.key,
                                     "salience": f.salience, "valid_at": f.valid_at_iso}
                                    for f in facts]}
        rec = {"source": source, "text": text, "target": target}
        (val if held else train).append(rec)
        if (i + 1) % 50 == 0:
            print(f"  labeled {i+1}/{len(rows)} (train={len(train)} val={len(val)})", flush=True)

    with (DATA / "train.jsonl").open("w") as fh:
        for r in train:
            fh.write(json.dumps(r) + "\n")
    with (DATA / "val.jsonl").open("w") as fh:
        for r in val:
            fh.write(json.dumps(r) + "\n")
    stats = {"train": len(train), "val": len(val), "dropped_kv": n_drop_kv,
             "dropped_empty_label": n_drop_empty, "dedup": n_dup,
             "total_clean": len(train) + len(val)}
    (DATA / "train_stats.json").write_text(json.dumps(stats, indent=2))
    print("STATS:", json.dumps(stats))


if __name__ == "__main__":
    main()
