"""Generate held-out eval data for the distiller probe. Two artifacts:

  data/eval_messages.jsonl  — 30 held-out messages (facts / churn-updates / questions /
                              chitchat), each labeled by the qwen3.7-plus reference distiller.
  data/paraphrase_groups.json — key-consistency groups: same fact VALUE, many phrasings.
                                A correct distiller must emit ONE key across a group
                                (that identity is what makes bi-temporal supersession work).

Reference labeling is the only cloud cost here (~30 flash-tier calls). Deterministic seed.
"""
from __future__ import annotations

import json
from pathlib import Path

from harness import qwen_endpoint

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

# --- Paraphrase groups: fixed value, varied phrasing → must collide on ONE key -------
# gold_key_hint is the churn-benchmark's canonical key where one exists; for the others
# it's the concept we'd expect. Scoring measures WITHIN-group key identity, not match to
# the hint (a model may reasonably pick user::home vs user::residence — as long as it's
# CONSISTENT across paraphrases the same attribute always collides).
PARAPHRASE_GROUPS = [
    ("user::residence", "Portland", [
        "I just moved to Portland.", "I now live in Portland.",
        "Relocated to Portland last month.", "My new place is in Portland.",
        "These days I'm based in Portland."]),
    ("user::job_title", "Data Scientist", [
        "I got promoted, I'm now a Data Scientist.", "My new role is Data Scientist.",
        "I just became a Data Scientist.", "They made me a Data Scientist.",
        "I work as a Data Scientist now."]),
    ("user::car", "Toyota Prius", [
        "I bought a new car, a Toyota Prius.", "Just picked up a Toyota Prius.",
        "I'm driving a Toyota Prius these days.", "My new ride is a Toyota Prius.",
        "Switched to a Toyota Prius."]),
    ("user::phone", "Pixel 12", [
        "I switched phones, now using a Pixel 12.", "Got a Pixel 12 yesterday.",
        "My new phone is a Pixel 12.", "I'm on a Pixel 12 now.",
        "Upgraded to a Pixel 12."]),
    ("user::gym", "Equinox Downtown", [
        "I changed gyms, I go to Equinox Downtown now.", "Started at Equinox Downtown.",
        "My gym is Equinox Downtown these days.", "I work out at Equinox Downtown now.",
        "Joined Equinox Downtown."]),
    ("user::name", "Marcus", [
        "My name is Marcus.", "I'm Marcus.", "Everyone calls me Marcus.",
        "You can call me Marcus.", "Marcus here."]),
    ("user::employer", "Stripe", [
        "I work at Stripe.", "I just joined Stripe.", "My employer is Stripe.",
        "I'm employed at Stripe.", "I landed a job at Stripe."]),
    ("user::pet", "a golden retriever named Biscuit", [
        "I got a golden retriever named Biscuit.", "My new dog is a golden retriever, Biscuit.",
        "I adopted Biscuit, a golden retriever.", "We have a golden retriever called Biscuit.",
        "Biscuit, my golden retriever, just joined the family."]),
    ("user::birthday", "March 3rd", [
        "My birthday is March 3rd.", "I was born on March 3rd.",
        "I celebrate my birthday on March 3rd.", "March 3rd is my birthday.",
        "My birth date is March 3rd."]),
    ("user::coffee_pref", "oat milk latte", [
        "I always order an oat milk latte.", "My go-to coffee is an oat milk latte.",
        "I prefer an oat milk latte.", "Oat milk latte is my usual.",
        "I drink oat milk lattes."]),
    ("user::allergy", "peanuts", [
        "I'm allergic to peanuts.", "I have a peanut allergy.",
        "Peanuts make me sick.", "I can't eat peanuts, I'm allergic.",
        "My allergy is to peanuts."]),
    ("user::hometown", "Cleveland", [
        "I grew up in Cleveland.", "My hometown is Cleveland.",
        "I'm originally from Cleveland.", "I was raised in Cleveland.",
        "Cleveland is where I'm from."]),
    ("user::partner", "Priya", [
        "My partner's name is Priya.", "I'm dating Priya.",
        "My girlfriend is Priya.", "Priya and I are together.",
        "My significant other is Priya."]),
    ("user::diet", "vegetarian", [
        "I'm vegetarian.", "I follow a vegetarian diet.",
        "I don't eat meat, I'm vegetarian.", "My diet is vegetarian.",
        "I eat vegetarian."]),
    ("user::employer_start", "next Monday", [
        "I start my new job next Monday.", "My first day is next Monday.",
        "I begin work next Monday.", "Next Monday is my start date.",
        "I'm starting next Monday."]),
]

# --- 30 held-out eval messages, 4 categories -----------------------------------------
FACTS = [  # multi/single-fact statements → reference labels them
    "I moved to Austin last spring and I'm now a product manager at Figma.",
    "My daughter Lena just turned 7 and started second grade.",
    "I switched my phone to an iPhone 15 and my number is now 555-0182.",
    "We're closing on the house at 42 Elm Street on June 12th.",
    "I've decided to go gluten-free after the doctor found I'm celiac.",
    "My flight to Tokyo leaves March 3 at 14:20 from gate B12.",
    "I sold my old Honda and bought a Tesla Model 3 for $41,000.",
    "I got engaged to Sam over the weekend, wedding is planned for next October.",
    "I'm training for the Boston Marathon in April, aiming for under 4 hours.",
    "My new manager is Dana and we have 1:1s every Tuesday at 3pm.",
    "I quit smoking three months ago and I've saved about $600 so far.",
    "I adopted a cat named Miso from the shelter yesterday.",
]
CHURN_UPDATES = [
    "I just moved to Denver.", "I got promoted, I'm now a Staff Engineer.",
    "I bought a new car, a Subaru Forester.", "I switched phones, now using a OnePlus 12.",
    "I changed gyms, I go to Gold's Gym Midtown now.", "I relocated to Seattle for work.",
]
QUESTIONS = [  # fabrication bait — a good distiller extracts NOTHING from a bare question
    "Where do I currently live?", "What car am I driving these days?",
    "What's my job title now?", "Which gym do I go to?",
    "What phone do I use?", "Remind me what my allergy is?",
]
CHITCHAT = [  # pure small talk → empty
    "The weather has been really nice this week.",
    "Traffic was terrible this morning.",
    "I watched a great documentary last night.",
    "Ugh, Mondays. Anyway, how's it going?",
    "That new coffee shop downtown smells amazing.",
    "Can't believe it's already almost the weekend.",
]


def main():
    ref = qwen_endpoint("qwen3.7-plus")

    groups = [{"gold_key_hint": k, "value": v, "paraphrases": ps}
              for (k, v, ps) in PARAPHRASE_GROUPS]
    (DATA / "paraphrase_groups.json").write_text(json.dumps(groups, indent=2))
    print(f"wrote {len(groups)} paraphrase groups "
          f"({sum(len(g['paraphrases']) for g in groups)} messages)")

    rows = []
    plan = ([("fact", m) for m in FACTS] + [("churn", m) for m in CHURN_UPDATES]
            + [("question", m) for m in QUESTIONS] + [("chitchat", m) for m in CHITCHAT])
    for i, (cat, text) in enumerate(plan):
        facts, raw = ref.distill(text)
        rows.append({
            "id": i, "category": cat, "text": text,
            "ref_facts": [{"statement": f.statement, "key": f.key,
                           "salience": f.salience, "valid_at": f.valid_at_iso}
                          for f in facts],
        })
        print(f"  [{cat:9s}] {len(facts)} facts  <- {text[:50]}")
    with (DATA / "eval_messages.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} eval messages")


if __name__ == "__main__":
    main()
