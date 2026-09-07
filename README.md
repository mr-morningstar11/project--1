# Trying the 3 new datasets — what actually worked, and what didn't

Short version: **`fake_news_dataset.csv` has no learnable signal for a
fake/real classifier** (I checked and proved this, don't just take my word
for it — see below). **`social_media_user_behavior.csv` is genuinely useful**
and I built a real working GNN on it. **`platform_statistics_2026.csv`** is
just 17 rows of reference stats — good for a chart, too small to train
anything on.

## 1. `fake_news_dataset.csv` — checked, and it's not usable as-is

Two problems, both fatal for a fake-news classifier:

- **The `title`/`text` columns are placeholder text.** Every row reads
  something like *"Breaking News 1"* / *"This is the content of article 1.
  It contains detailed analysis and reports."* — the number is the only
  thing that changes. There's no actual article content to analyze.
- **Every engineered feature is uncorrelated with the label.** I checked
  the correlation between the label and every numeric column
  (`sentiment_score`, `trust_score`, `clickbait_score`, `num_shares`,
  `source_reputation`, etc.) — all of them sit between **-0.03 and +0.03**
  (see `outputs/fake_news_dataset_correlations.png`). Every category
  (`source`, `political_bias`, `fact_check_rating`, `state`, `author`) splits
  almost exactly 50/50 Fake vs. Real. That's the signature of a label that
  was assigned at random, independent of the row's data.

I didn't just stop at the correlation check — I ran the **same GCN pipeline**
from the main project on this file (`src/eval_fake_news_dataset.py`): built
a k-NN similarity graph over every non-text feature, trained the GCN and a
Logistic Regression baseline, same as before. Results:

| Model | Test accuracy |
|---|---|
| GCN | 50.1% |
| Logistic Regression | 50.2% |
| Graph homophily | 0.502 (a coin flip) |

Both models land exactly at chance. That's not a bug in the pipeline — it's
the correct, honest outcome for data with no signal in it. Reporting a fake
"90% accuracy" here would be actively misleading for your review; this
result is worth reporting *as-is* — "we validated our data before modeling
and this dataset doesn't support the claims it appears designed for" is a
legitimate and good finding for a project review.

**If you want a real fake-news classifier**, the `fake_or_real_news.csv`
dataset from the first version (6,335 real articles with real text) is the
one to use — see the earlier `fake-news-gnn-project.zip`. If this new
`fake_news_dataset.csv` came with a source/description you have and I don't
(e.g. it's meant to pair with a specific companion file), send that over and
I'll take another look.

## 2. `social_media_user_behavior.csv` — this one's real, so I built on it

25,000 users with real internal structure: `addiction_level_1_to_10` is
strongly driven by `daily_screen_time_minutes` (r=0.71), `weekly_sessions`
(r=0.58), `video_consumption_daily_minutes` (r=0.61), `age` (r=-0.57), and
`sleep_hours_per_night` (r=-0.37, inversely). This is exactly the kind of
"user profile" data your proposal's architecture calls for, so I built a
genuine model with it: `src/user_behavior_gnn.py` classifies users into
**low vs. high social-media-addiction risk** using a user-similarity graph
(same k-NN + GCN recipe as the main project, over demographic + behavioral
features) plus the same baseline/fusion comparison.

Sampled 6,000 users for runtime (8.1% are high-risk by this definition).
Graph homophily = **0.854** — high-risk users really do cluster together in
feature space, so the graph is meaningful here (unlike file #1). Threshold
tuned on the validation split for best F1 per model:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| GCN (graph only) | 84.5% | 30.0% | 69.1% | 0.42 | 0.879 |
| Behavior features only (Logistic Regression) | **88.6%** | **37.5%** | 61.9% | **0.47** | **0.891** |
| Fusion (features + graph) | 88.2% | 34.7% | 51.5% | 0.42 | 0.871 |

Honest read: AUC is strong across the board (~0.87–0.89) — this is a
genuinely learnable problem. Precision is modest for all three because
high-risk users are a small minority (8%) of the sample; recall trades off
against it depending on threshold. Unlike the news task, the graph doesn't
add much here — the per-user behavioral features already capture almost all
of the signal on their own, so message-passing has little left to add. That
contrast (graph helps when node features alone are ambiguous; doesn't when
they're already highly predictive) is a good discussion point for your
review.

**No link to `fake_news_dataset.csv` exists in the data** — there's no
`user_id` in the news file or `article_id` here, so I did not invent one.
If you get access to real FakeNewsNet-style engagement data (users who
actually shared specific articles), that's the piece that would let you
build the genuine "user–news interaction graph" from your proposal's
Objectives slide.

## 3. `platform_statistics_2026.csv` — reference data, not training data

Only 17 rows (one per platform), so there's nothing to train a model on.
Used it for one contextual chart instead (`outputs/platform_statistics_overview.png`):
Facebook/Instagram/WhatsApp lead on monthly active users, while TikTok and
Reddit lead on engagement rate — useful context if your report wants a
"where does misinformation actually spread" slide, but not itself GNN input.

### BERT text baseline

`src/bert_fake_news.py` adds a BERT-based text classifier to the same
evaluation. It fine-tunes `bert-base-uncased` on the concatenated title and
article text, uses stratified 60/20/20 train/validation/test splits, selects
the best epoch using validation accuracy, and reports only held-out test
metrics. Run it from the project root after installing the dependencies:

```bash
pip install -r requirements.txt
python src/bert_fake_news.py
```

The first run downloads pretrained BERT weights. `--model-name
distilbert-base-uncased` provides a lighter option. Because this CSV's text is
template content, BERT should also remain around chance on the held-out set;
that is the expected data-quality result, not a defect in BERT.

## Files

```
project2/
├── data/                                       # the 3 uploaded files
├── src/
│   ├── gcn_model.py                            # same GCN as the main project
│   ├── eval_fake_news_dataset.py               # proves file #1 has no signal
│   └── user_behavior_gnn.py                    # real working GNN on file #2
└── outputs/
    ├── fake_news_dataset_correlations.png       # evidence for file #1
    ├── fake_news_dataset_results.json
    ├── user_behavior_model_comparison.png       # results for file #2
    ├── user_behavior_confusion_gcn.png
    ├── user_behavior_training_curve.png
    ├── user_behavior_results.json
    └── platform_statistics_overview.png         # context chart for file #3
```
