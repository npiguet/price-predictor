---
name: domain-modeling
description: Build and sharpen the domain model. Use when discussing codebase, experiment or spec terminology, writing or editing a GLOSSARY.md, or recording or editing an experiment file or a spec file.
---

# Domain Modeling

Actively build and sharpen the project's domain model as you design. This is the *active* discipline: challenging terms, inventing edge-case scenarios, and writing the glossary and decisions down the moment they crystallise. (Merely *reading* `GLOSSARY.md` for vocabulary is not this skill: that's a one-line habit any skill can do. This skill is for when you're changing the model, not just consuming it.)

The secondary goal of this skill is to combat the cognitive surrender effect. By making sure that everything your talk about with the user is crystal clear to the user, the user will be able to follow your reasoning more closely.

The glossary does not need to redefine terms that are part of the "Magic The Gathering" game and rules domain. Those are considered part of the implicit domain of the project and don't need to be redefined. However, when using MTG terms, you must stick strictly to their accepted definition. Avoid using similar sounding but invented terms. Using "ping" is clear: it's an effect that cases 1 damage. "activated ability" is clear: it's an ability that a player can activate whenever he wants to. "Activated ping" is not clear: This is a term that a reader could maybe have a guess at its meaning, but is unclear and adds un-necessary cognitive. "ping activated ability" is clear.

## File structure

This project is a multi-module project where each module has its own context. It has the following structure:

```
/
├── GLOSSARY.md                                           ← system-wide glossary
├── experiments/                                          ← system-wide experiment files (see feature-workflow skill)
│   ├── 2026-03-18-ordinal-regression-experiment.md
│   └── 2026-03-21-network-size-experiments.md
├── specs/                                                ← system-wide spec files (see feature-workflow skill)
│   ├── 01-card-price-predictor/
│   │   ├── spec.md
│   │   ├── plan.md
│   │   └── data-model.md
│   ├── 2026-03-28-sealed-deck-picker.md
│   ├── 2026-03-28-sealed-deck-picker.md
│   ├── 2026-03-21-card-price-predictor.md
│   └── 2026-03-21-card-price-predictor.md
├── src/
│   ├── price_predictor/
│   │   └── GLOSSARY.md                                   ← module specific glossary
│   ├── sealed/
│   │   └── GLOSSARY.md                                   ← module specific glossary
│   ├── draft/
│   │   └── GLOSSARY.md                                   ← module specific glossary
│   └── effects/
│       └── GLOSSARY.md                                   ← module specific glossary
└── forge-connector/
    └── GLOSSARY.md                                       ← module specific glossary
```

Create files lazily: only when you have something to write. If no `GLOSSARY.md` exists, create one when the first term is resolved.

## During the session

### Challenge against the glossary

When the user uses a term that conflicts with the existing language in `GLOSSARY.md`, call it out immediately. "Your glossary defines 'cancellation' as X, but you seem to mean Y. Which is it?"

### Sharpen fuzzy language

When the user uses vague or overloaded terms, propose a precise canonical term. "You're saying 'account': do you mean the Customer or the User? Those are different things."

### Discuss concrete scenarios

When domain relationships are being discussed, stress-test them with specific scenarios. Invent scenarios that probe edge cases and force the user to be precise about the boundaries between concepts.

### Cross-reference with code

When the user states how something works, check whether the code agrees. If you find a contradiction, surface it: "Your code cancels entire Orders, but you just said partial cancellation is possible. Which is right?"

### Resolve misunderstandings

When the user expressed that a particular term you used wasn't understood, take this as a signal that you've either used the wrong word and must refer to the glossary, or that there is a gap in the glossary that must be filled by a new definition. 

### Update GLOSSARY.md inline

When a term is resolved, update `GLOSSARY.md` right there. Don't batch these up: capture them as they happen. Use the format in [GLOSSARY-FORMAT.md](./GLOSSARY-FORMAT.md).

`GLOSSARY.md` should be totally devoid of implementation details. Do not treat `GLOSSARY.md` as a spec, a scratch pad, or a repository for implementation decisions. It is a glossary and nothing else.