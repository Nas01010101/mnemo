"""Decontaminated held-out eval: values AND phrasings disjoint from train.jsonl, so the
tuned model cannot have memorized any eval message. Verifies zero text-overlap, then
labels with the qwen3.7-plus reference. Writes eval_messages_clean.jsonl +
paraphrase_groups_clean.json. This is the honest generalization test."""
from __future__ import annotations

import json
from pathlib import Path

from harness import qwen_endpoint

DATA = Path(__file__).resolve().parent / "data"

# values NOT present in generate_train_data.py pools
NV = {
    "city": ["Helsinki", "Lisbon", "Kyoto", "Reykjavik", "Montevideo"],
    "job": ["Neurosurgeon", "Sommelier", "Air Traffic Controller", "Cartographer"],
    "car": ["Rivian R1T", "Polestar 2", "Lucid Air", "Genesis GV70"],
    "phone": ["Nothing Phone 2", "Asus Zenfone 11", "Motorola Edge 50"],
    "gym": ["Barry's Bootcamp SoHo", "Orangetheory Riverside"],
    "name": ["Thaddeus", "Ingrid", "Rafael", "Nadia"],
    "company": ["Palantir", "Cloudflare", "Databricks", "Instacart"],
    "allergen": ["kiwi", "sesame", "latex", "sulfa drugs"],
}
# phrasings NOT used as train templates (novel surface forms), key = canonical attribute
CLEAN_PARA = [
    ("user::residence", "Helsinki", [
        "As of this week my whole life is in Helsinki.",
        "Helsinki is home for me now.",
        "I packed up and resettled in Helsinki.",
        "Currently residing in Helsinki these days.",
        "Helsinki's where you'll find me living now."]),
    ("user::job_title", "Neurosurgeon", [
        "After years of training I finally practice as a Neurosurgeon.",
        "My profession these days is Neurosurgeon.",
        "I've stepped into the role of Neurosurgeon.",
        "Career-wise I'm a Neurosurgeon at this point.",
        "They list my title as Neurosurgeon now."]),
    ("user::car", "Rivian R1T", [
        "The vehicle in my driveway is now a Rivian R1T.",
        "I get around in a Rivian R1T these days.",
        "A Rivian R1T is what I take to work.",
        "My current set of wheels is a Rivian R1T.",
        "I traded up and a Rivian R1T is mine now."]),
    ("user::phone", "Nothing Phone 2", [
        "The handset in my pocket is a Nothing Phone 2.",
        "I've been carrying a Nothing Phone 2 lately.",
        "My daily driver device is a Nothing Phone 2.",
        "A Nothing Phone 2 is what I text from now.",
        "I make my calls on a Nothing Phone 2 these days."]),
    ("user::name", "Thaddeus", [
        "Folks address me as Thaddeus.",
        "The name on my badge reads Thaddeus.",
        "I go by Thaddeus.",
        "You'll want to call me Thaddeus.",
        "Thaddeus is what my friends use."]),
    ("user::employer", "Palantir", [
        "My paychecks come from Palantir.",
        "I clock in at Palantir these days.",
        "Palantir is where I'm on staff.",
        "I'm on the payroll at Palantir now.",
        "These days Palantir signs my paychecks."]),
    ("user::allergy", "kiwi", [
        "Eating kiwi sends me to the ER.",
        "My body reacts badly to kiwi.",
        "Kiwi is off-limits for me medically.",
        "I break out whenever I touch kiwi.",
        "Doctors told me to avoid kiwi entirely."]),
    ("user::hometown", "Montevideo", [
        "The city that raised me is Montevideo.",
        "Montevideo is where my childhood happened.",
        "I spent my early years in Montevideo.",
        "Montevideo shaped who I am growing up.",
        "Back home means Montevideo for me."]),
]

# clean single/multi fact messages (novel values + novel phrasings)
FACTS = [
    "Word came down today that I'm the new Sommelier, and I've relocated to Lisbon for it.",
    "The doctor confirmed a sesame reaction, so sesame is now completely off my plate.",
    "I finally took delivery of a Polestar 2 after months on the waitlist.",
    "Rafael proposed on the ferry; we're aiming for a spring ceremony in Kyoto.",
    "Starting Monday my paychecks come from Cloudflare instead of my old shop.",
    "I've begun carrying an Asus Zenfone 11 after my old handset finally died.",
    "My childhood unfolded in Reykjavik, though I left it years ago.",
    "The badge now reads Air Traffic Controller, a role I stepped into last quarter.",
]
CHURN = [
    "As of this week my whole life is in Lisbon.",
    "After years of training I finally practice as a Cartographer.",
    "The vehicle in my driveway is now a Lucid Air.",
    "The handset in my pocket is a Motorola Edge 50.",
    "My paychecks come from Databricks these days.",
    "Latex gloves send me to the ER now.",
]
QUESTIONS = [
    "So where is it that I'm living now?", "Which company cuts my paycheck these days?",
    "What was the model of car I mentioned?", "Can you dig up my stated allergy?",
    "Which handset did I say I carry?", "What role did I say I stepped into?",
]
CHITCHAT = [
    "Honestly this cold snap is relentless.", "The elevator's been broken all week, ugh.",
    "I keep hearing that new album everywhere.", "Long day, glad it's winding down.",
    "The farmers market smelled incredible today.", "My inbox is a disaster zone right now.",
]


def main():
    train_txt = set(json.loads(l)["text"] for l in (DATA / "train.jsonl").open())
    train_txt |= set(json.loads(l)["text"] for l in (DATA / "val.jsonl").open())

    groups = [{"gold_key_hint": k, "value": v, "paraphrases": ps} for (k, v, ps) in CLEAN_PARA]
    plan = ([("fact", m) for m in FACTS] + [("churn", m) for m in CHURN]
            + [("question", m) for m in QUESTIONS] + [("chitchat", m) for m in CHITCHAT])

    all_msgs = [m for _, m in plan] + [p for g in groups for p in g["paraphrases"]]
    overlap = [m for m in all_msgs if m in train_txt]
    assert not overlap, f"LEAKAGE: {overlap}"
    print(f"leakage check PASS: 0/{len(all_msgs)} clean-eval messages appear in train")

    (DATA / "paraphrase_groups_clean.json").write_text(json.dumps(groups, indent=2))
    ref = qwen_endpoint("qwen3.7-plus")
    rows = []
    for i, (cat, text) in enumerate(plan):
        facts, _ = ref.distill(text)
        rows.append({"id": i, "category": cat, "text": text,
                     "ref_facts": [{"statement": f.statement, "key": f.key,
                                    "salience": f.salience, "valid_at": f.valid_at_iso}
                                   for f in facts]})
        print(f"  [{cat:9s}] {len(facts)} facts <- {text[:45]}")
    with (DATA / "eval_messages_clean.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} clean eval messages, {len(groups)} clean paraphrase groups")


if __name__ == "__main__":
    main()
