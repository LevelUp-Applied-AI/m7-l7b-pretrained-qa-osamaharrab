# Adversarial QA Probe Analysis

## 1. Hypothesis

The targeted failure mode is **distractor entity selected**, from `qa-evaluation-report.md`.

- **Input pattern:** the context contains two or more plausible entities of the same answer type, and the question asks for the entity attached to one specific relation or event.
- **Output pattern:** the model returns the salient distractor entity instead of the entity tied to the asked relation. In the strongest version, the same two entities appear in reversed roles across two sentences.
- **Why I hypothesize this:** `distilbert-base-cased-distilled-squad` predicts a span from local token evidence. When multiple spans have the right entity type, the model can overweight salience, recency, or nearby lexical overlap and underweight the full predicate-argument relation. This matches the original `NEWS_0221_Q5` error, where the model chose `Tweed` instead of `Jade Goody` in a marriage story with multiple people.

## 2. Set Design

- Total examples: 30
- Tags used: `same-type-distractor` 10, `late-distractor` 9, `role-swap` 8, `control` 3
- `same-type-distractor` tests contexts with another person, company, city, product, or event of the same type near the answer.
- `late-distractor` tests whether the model drifts to a later salient entity after the correct answer appears earlier.
- `role-swap` tests the hardest version: two entities appear in both subject and object roles across different sentences, so entity type alone is insufficient.
- The 3 control examples keep the same short extractive style but remove the competing same-type distractor. They test whether failures are caused by distractor structure rather than the model being unable to answer short synthetic contexts.

## 3. Results

- Adversarial aggregate EM: 0.8000; F1: 0.8990
- Lab 7B baseline: EM 0.3440; F1 0.4611

| Pattern | n | EM | F1 | vs. baseline F1 |
|---|---:|---:|---:|---:|
| control | 3 | 1.0000 | 1.0000 | +0.5389 |
| late-distractor | 9 | 0.8889 | 0.9778 | +0.5167 |
| role-swap | 8 | 0.5000 | 0.6714 | +0.2103 |
| same-type-distractor | 10 | 0.9000 | 0.9800 | +0.5189 |

The aggregate adversarial score is higher than the Lab 7B baseline, so this set does not show a broad QA collapse. Instead, it localizes the weakness: simple distractors and controls were mostly handled correctly, while `role-swap` was clearly weaker at 0.5000 EM and 0.6714 F1. This supports the hypothesis only in its strongest form: the model is most vulnerable when the same entities appear in reversed relations, not whenever a same-type distractor exists.

- `ADV_023`: "Who filed the lawsuit?" -> gold: `Rina Patel`, predicted: `Atlas Media`. The model chose the opposing party, not the actor who filed.
- `ADV_025`: "Which nonprofit donated the laptops?" -> gold: `CodeBridge`, predicted: `Open Shelter`. The context later reverses donor/recipient roles for desks, and the model returned the recipient from the laptop sentence.
- `ADV_021`: "Who became chief security officer?" -> gold: `Maya Chen`, predicted: `Maya Chen replaced Luis Ortega`. This is partly correct by F1, but it shows span-boundary drift when two people swap roles in adjacent clauses.
- `ADV_028`: control question -> gold: `Denver`, predicted: `Denver`. The controls were all exact matches, so the low `role-swap` score is not just synthetic-context difficulty.

## 4. Production Defense

The concrete defense I would choose is **adversarial fine-tuning with role-swap distractor examples**. The controls scored 1.0000 F1 and simple distractors scored around 0.98 F1, so a broad fallback like blocking all multi-entity contexts would be too blunt. The measured weakness is narrower: contexts where the same entities appear in reversed relations. Adding labeled examples of donor/recipient, plaintiff/defendant, acquirer/acquired, interviewer/interviewee, and winner/loser swaps should help the QA model rely more on relation cues instead of entity salience alone.
