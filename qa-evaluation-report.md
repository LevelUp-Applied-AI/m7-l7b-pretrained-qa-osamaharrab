# QA Evaluation Report

## Dataset description

The evaluation set is a 1,000-row tech, entertainment, and digital-culture QA slice from `data/tech_news_qa.csv`, curated from the CNN-based `glnmario/news-qa-summarization` dataset. Each gold answer is extractive: it appears as a literal substring of the article context.

## Model

Model: `distilbert-base-cased-distilled-squad`  
Hugging Face Hub: https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad

## Aggregate metrics

Exact Match: 34.40%  
Token-F1: 46.11%

The 11.71-point gap suggests the model often lands near the right evidence but misses the exact annotated span. This is consistent with extractive QA behavior on news text: partial spans, nearby entities, and different answer granularity get some token credit but fail exact match.

## Failure-mode taxonomy

1. Distractor entity selected. Example `NEWS_0221_Q5`: "Who is fast-tracking to get married?" Gold: `Jade Goody`; predicted: `Tweed`. The model chose another salient person in the marriage story instead of the subject of the question.

2. Attribution/content swap. Example `NEWS_0916_Q3`: "What did the Los Angeles Times music critic call Dudamel?" Gold: `a phenomenon,"`; predicted: `Mark Swed`. The model returned the speaker/critic rather than the quoted description.

3. Wrong answer granularity. Example `NEWS_0593_Q2`: "What is the film about?" Gold: `Michael Oher, who went from being a homeless inner-city high school student whose father was dead and whose mother was a crack addict to a star lineman at the University of Mississippi`; predicted: `The Blind Side`. The model picked the film title instead of the long descriptive span required by the annotation.

## Domain judgment

I would not ship this model as-is for legal contract QA. The extractive constraint is useful for faithfulness, but 34.40% EM is too low for contract review, where a wrong party, date, or clause boundary can change the answer materially. A production version would need calibrated confidence, no-answer support, stronger domain evaluation, and human review for low-confidence or high-impact answers.
