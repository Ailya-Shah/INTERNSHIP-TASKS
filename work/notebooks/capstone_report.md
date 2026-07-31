# Capstone Report — Ranking Signal Analysis

- **Author:** Ailya Shah
- **Lane:** Ranking Signal Analysis
- **Repo:** https://github.com/Ailya-Shah/INTERNSHIP-TASKS
- **Date:** July 2026


## 0. Abstract

Which observable content and query-demand signals are associated with a page reaching page-one search visibility, and can a simple learned model rank likely page-one pages better than a transparent rule? Using one mid-panel month (March 2026) of FlyRank's pseudonymized search-performance warehouse — 53,892 published pages across 31 clients — I built five leakage-checked features (word count, content age, search demand, competition, backlinks) and compared a hand-written rule baseline against Logistic Regression and Random Forest, validated with a client-grouped split so no client's pages leaked between train and test. Logistic Regression showed the best honest performance (ROC AUC 0.561, Average Precision 0.589, versus the rule baseline's 0.479/0.523), and its coefficients confirmed two counter-intuitive signals independently found in exploratory analysis: shorter, newer content associates with higher page-one odds, while backlinks and search demand push the expected positive direction. The output is a reason-coded action queue of 8,747 opportunity pages an editor can review first, each carrying the specific signals that earned it a spot on the list. That queue comes with explicit caveats attached rather than hidden: an 8-client held-out validation group and a 43% share of top recommendations resting on likely-missing rather than genuinely-zero demand data.

## 1. Problem framing

**Decision supported:** which of thousands of content pages a content editor with limited review hours should look at first for page-one potential, and which pages that are already succeeding should be left alone.

**Unit of analysis:** one content page (`content_hash_id`), aggregated over March 2026.

**Output:** a probability score per page, converted into three priority tiers — OPPORTUNITY (not yet page-one, high model score), PROTECT (already page-one, high score — leave alone, monitor), and LOW_PRIORITY.

**Action a human takes:** an editor works down the OPPORTUNITY queue, using each page's reason code (e.g. `strong_backlinks`, `concise_content`) to decide what to improve — pursue backlinks, tighten length, check freshness — while PROTECT-tier pages are monitored rather than edited.

**Cost of a wrong call:** moderate and asymmetric. A false OPPORTUNITY flag costs an editor's review time — annoying but recoverable. A missed real opportunity is a silent, harder-to-notice loss. Neither cost is safety-critical, which is why a *ranking* that's meaningfully better than guessing is valuable even without a perfect classifier.

**Why data/ML helps here, not a hand rule:** the individually-tested signals are each only moderately predictive (10–14 percentage-point gaps), and the single most intuitive one — content length — points in the *opposite* direction from common assumption: shorter content, not longer, associates with page-one status in this portfolio (67% page-one rate in the shortest word-count quartile vs. 33% in the longest). A hand-picked rule built on the obvious assumption would already be wrong on its strongest lever. Combining several moderate, sometimes counter-intuitive signals is exactly the situation a learned model is suited for over an if-statement.

## 2. Data safety

**Release:** `FlyRank/internship-warehouse` (Hugging Face, gated, instant-approval), build `v20260703`.

**Tables used:** `fact_content_daily_performance` (full partitioned daily fact — never the `_sample.parquet` file, which is exactly June 2026, the sealed final month, deliberately preserved as an untouched future holdout), `dim_content` (content/keyword metadata), `dim_clients` (context only — panel-history checks and the client-grouping key for splits).

**Window:** `month = '2026-03'`, a mid-panel month chosen specifically to leave June 2026 sealed.

**Deliberately excluded, and why:**
- Rows with `gsc_data_available = FALSE` — zero-filled "instrument not yet tracking" rows, not real zero-traffic observations. Confirmed directly in the raw data (some rows show `gsc_avg_position = null` with all-zero metrics under this flag).
- `gsc_avg_position`, `gsc_sum_position`, `gsc_clicks` as **features** — the label and its direct derivatives. Confirmed as leaks via a deliberate confession test: adding `avg_position_win` as a feature sent AUC from 0.561 to 0.997.
- `fact_content_query_90d` (query-mix) features — tested explicitly rather than assumed safe, and found to leak (largest gap of any attack tested, +0.171 AUC), likely from its trailing 90-day window overlapping the March label period.
- `provider_used`, `model_used` — explicitly non-feature process metadata per the data dictionary.
- All `*_hash_id` columns — grouping/joining/splitting keys only, never model inputs.
- Pages with `content_age_days < 90`, `gsc_impressions_win < 100`, or `is_published = FALSE` / `is_deleted = TRUE`.

**Client-identifying content:** no client names, domains, URLs, or raw queries appear anywhere in `work/`; only pseudonymous hashed IDs are used, and only for grouping/splitting.

## 3. Baseline

**The rule, in plain words:** a page is worth reviewing first if it has real search demand, isn't fighting impossible competition, has earned some off-page trust, and has been live long enough for a fair shot — four binary conditions (`high_demand`, `low_competition`, `strong_backlinks`, `established`) summed 0–4, no fitted weights.

**A real bug caught and fixed:** the first version used all-rows medians for thresholds, which collapsed to exactly 0.0 for `search_volume`, `competition`, and `backlinks` because these fields are heavily zero-inflated (most pages have no recorded value). This made two of the four conditions almost meaningless (trivially true or trivially restrictive) and produced Precision@20 (0.35) *below* the base rate (0.54) — worse than guessing. Fixed by computing medians only among nonzero values, plus a continuous rank-based tiebreak (backlinks + demand + inverse-competition ranks) to reduce noise from the large number of tied scores.

**Baseline numbers on the honest, client-grouped split (n=4,049 held-out pages, base rate 0.536):**

| Metric | Rule baseline |
|---|---|
| Precision@20 | 0.55 |
| Precision@50 | 0.48 |
| ROC AUC | 0.479 |
| Average Precision | 0.523 |

The rule shows only a marginal edge at K=20 and none at K=50/AUC — expected for a simple additive rule combining moderate individual signals, and itself a finding: it empirically demonstrates why a learned model is worth building, rather than just asserting it.

## 4. Model / analysis

**Method:** Logistic Regression (`class_weight='balanced'`) as the primary, interpretable model — standardized coefficients read directly as signed, ranked signal strength. Random Forest trained alongside as a nonlinearity check.

**Final feature list (5 numeric, 2 categorical, all leakage-tested):** `word_count`, `content_age_days`, `search_volume`, `competition`, `backlinks` (numeric, median-imputed, standardized); `content_type`, `main_intent` (categorical, one-hot). Deliberately excluded: `gsc_clicks`/position columns (confirmed label leaks) and all four query-mix columns (confirmed leak, +0.171 AUC gap when tested).

**Label definition, one sentence:** `is_page_one` = 1 if a page's impression-weighted average GSC position over March 2026 falls between 1 and 10, else 0 — a current-state, cross-sectional label, not a future prediction.

## 5. Evaluation

**Split: grouped by `client_hash_id`, not random** — `GroupShuffleSplit(test_size=0.25, random_state=42)`, so no client's pages appear in both train and test. **This choice was tested, not assumed:** re-running the identical model under an ungrounded random row-level split showed ROC AUC 0.654 / Average Precision 0.656 — noticeably higher than the honest grouped split's 0.561 / 0.589. That +0.093 AUC / +0.068 AP gap is real, moderate, and represents client-memorization the grouped split correctly strips out. Had only the random split been reported, performance would have been meaningfully overstated.

**Model vs. baseline, same split, same held-out clients (n=4,049, base rate 0.536):**

| Model | Precision@20 | Precision@50 | ROC AUC | Avg Precision |
|---|---|---|---|---|
| Base rate (majority class) | 0.50 | 0.48 | — | — |
| Rule baseline | 0.55 | 0.48 | 0.479 | 0.523 |
| **Logistic Regression** | 0.45 | 0.50 | **0.561** | **0.589** |
| Random Forest | 0.45 | 0.52 | 0.516 | 0.545 |

![Standardized signal strength — Logistic Regression coefficients, ranked by magnitude](https://raw.githubusercontent.com/Ailya-Shah/INTERNSHIP-TASKS/main/work/notebooks/work/outputs/figures/signal_importance.png)

*Figure 1. Signal strength and direction, read directly from the fitted Logistic Regression's standardized coefficients (generated in `capstone.ipynb`, "Interpretation — signal directions"). `word_count` and `content_age_days` point negative (shorter/newer → higher page-one odds); `backlinks` points positive, as expected.*

Precision@20/@50 are noisy point estimates at this held-out size (only 8 clients) and swing considerably; ROC AUC and Average Precision, computed over the full ranked list, are the more stable and trustworthy comparison. On those, **Logistic Regression is the best-performing method**, ahead of both the rule and Random Forest.

**Error analysis:** of 4,049 held-out pages, 644 (15.9%) were confidently wrong (predicted probability >0.8 for a true negative, or <0.2 for a true positive). The dominant pattern among confident false positives: pages with `search_volume`, `competition`, and `backlinks` all simultaneously at exactly zero — very likely missing keyword/backlink data rather than genuinely zero demand, since this feature set does not yet carry explicit missingness flags.

## 6. Interpretation

**Standardized logistic coefficients (signed, ranked by magnitude).** Printed directly from the fitted pipeline in `capstone.ipynb`'s "Interpretation — signal directions" cell (`logit.named_steps["clf"].coef_`, mapped back to feature names) — re-run that cell on a fresh clone to reproduce these exactly:

| Feature | Coefficient | Direction |
|---|---|---|
| `word_count` | -0.538 | shorter content → higher page-one odds |
| `backlinks` | +0.399 | more backlinks → higher page-one odds |
| `content_age_days` | -0.248 | newer content → higher page-one odds |
| `competition` | -0.040 | lower competition → higher page-one odds (weak) |

**A genuine cross-validation moment:** these coefficients independently reproduce two findings from exploratory analysis using a completely different method (grouped-median comparison rather than a fitted model) — `word_count`'s negative direction matches an OPPOSITE verdict found earlier (67% page-one rate in the shortest word-count quartile vs. 33% in the longest), and `backlinks`'s positive direction matches a CONFIRMED verdict (64% vs. 54% page-one rate, with vs. without any backlinks). Two independent methods agreeing on direction is a real robustness signal, not a coincidence. `content_age_days` being negative is a new finding not previously tested.

**A surprising negative result:** the staleness assumption behind FlyRank's own product flags — "older, un-updated content performs worse" — held only at the extreme tail (the stalest quartile showed a meaningfully lower page-one rate, 48.7%, versus a flat ~58–59% across the other three quartiles). A smooth "the older, the worse" story is not supported; only "very stale" behaves differently.

## Limitations & Honest Framing

- **One month, one portfolio.** These results describe March 2026 for one company's client set — not shown to generalize to other months, seasons, or organizations.
- **A small held-out validation group.** Only 8 clients (4,049 pages) back the honest performance numbers reported in Sections 3 and 5 — real, but statistically thin. Precision@K point estimates especially should be read as directional, not precise (they swing considerably even between reruns of the identical pipeline, since DuckDB does not guarantee row order across query executions).
- **Associational, not causal.** Every result here is an observed, cross-sectional pattern. Nothing here supports "doing X will cause Y" — only "pages with X are, in this data, more often page-one."
- **A meaningful share of recommendations rest on missing, not measured, data.** Roughly 43% of OPPORTUNITY-tier pages (3,730 of 8,747) have zero recorded `search_volume` and zero `backlinks` simultaneously — very likely missing keyword/backlink data, not confirmed zero-demand pages.
- **The model can react strongly to extreme values.** Row 3 in Section 7's queue (a 109-word page, zero backlinks, still scoring 0.831) illustrates this directly — a very short page can score highly almost entirely from one coefficient, which is exactly why a human read stays part of the workflow, not just the score.
- **Query-mix signals are excluded, not resolved.** `fact_content_query_90d` showed the largest leakage gap of anything tested (+0.171 AUC) and was excluded rather than fixed — a genuinely useful signal source is left on the table until its window-alignment against the label period can be verified precisely.
- **No claims of proving Google's ranking algorithm or a causal refresh effect are made anywhere in this work** — every finding is stated as observed, measured, or decision-support, per the claims checklist below.

## 7. Recommendation

The model produces three tiers: **8,747 OPPORTUNITY** pages (not yet page-one, high model score — review first), **18,199 PROTECT** pages (already page-one, high score — monitor, don't disturb), and **26,946 LOW_PRIORITY** pages. Each OPPORTUNITY pick carries a reason code (`strong_backlinks`, `real_demand`, `concise_content`, `recently_published`) built directly from the confirmed signal directions above, so an editor sees *why* a page was flagged, not just a bare score.

**Top of the queue** — the actual first 5 rows of `work/notebooks/work/outputs/action_playbook.csv`, pseudonymous IDs only:

| Rank | Content ID | Client ID | Model score | Reason code(s) | Word count | Backlinks |
|---|---|---|---|---|---|---|
| 1 | `content_1d10840d84866345` | `client_e5c2aa26a8598242` | 0.898 | `strong_backlinks+real_demand` | 3,902 | 538,652 |
| 2 | `content_199ccff5416079a4` | `client_400c21c81c8b46ef` | 0.831 | `strong_backlinks+real_demand+concise_content+recently_published` | 1,737 | 30,712 |
| 3 | `content_483ecb07a65bf7ab` | `client_2094c6eb080311d5` | 0.831 | `concise_content+recently_published` | 109 | 0 |
| 4 | `content_b678de5511c085dd` | `client_2094c6eb080311d5` | 0.827 | `strong_backlinks+real_demand+concise_content+recently_published` | 278 | 223 |
| 5 | `content_a95dc4a0751ba10d` | `client_400c21c81c8b46ef` | 0.813 | `strong_backlinks+real_demand+concise_content+recently_published` | 1,469 | 117,938 |

Row 3 is a good illustration of the model's extreme-value sensitivity flagged in the Limitations below: a 109-word page with zero backlinks still scores 0.831, driven almost entirely by the `word_count` coefficient — exactly why a human read stays part of the workflow, not just the score.

**Confidence and limits, stated plainly:** roughly 43% of OPPORTUNITY-tier pages (3,730 of 8,747) have zero recorded `search_volume` and zero `backlinks` simultaneously — very likely missing data, not confirmed zero-demand pages — and these should be treated with extra skepticism before acting. The model can also react strongly to extreme values (e.g. a 109-word page scored highly almost entirely from the length coefficient), so top-ranked pages still warrant a human read, not blind trust in the score. No recommendation here should be read as a guarantee — only as "worth reviewing first, given these observed associations."

## 8. Reproducibility

**Fresh-clone re-run:**
```bash
git clone https://github.com/Ailya-Shah/INTERNSHIP-TASKS.git
cd INTERNSHIP-TASKS
pip install -r requirements.txt
```
Then open, in order, `work/notebooks/w01_research_question.ipynb` through `capstone.ipynb`, running each top to bottom. Each notebook that touches the warehouse needs a Hugging Face **read** token (create at `huggingface.co/settings/tokens`, accept the gate at `huggingface.co/datasets/FlyRank/internship-warehouse` first) — supplied via Colab Secret (`HF_TOKEN`) or a `getpass` prompt, never hardcoded.

**Direct links:** [full repo](https://github.com/Ailya-Shah/INTERNSHIP-TASKS) · [all assignment notebooks](https://github.com/Ailya-Shah/INTERNSHIP-TASKS/tree/main/work/notebooks) · [capstone notebook](https://github.com/Ailya-Shah/INTERNSHIP-TASKS/blob/main/work/notebooks/capstone.ipynb) (open its Colab badge to run it directly, no local setup needed).

**Random seeds:** `random_state=42` fixed throughout — the `GroupShuffleSplit`, `LogisticRegression`, and `RandomForestClassifier` calls all use it, so re-running reproduces the same split and the same numbers.

**Sealed holdout status:** June 2026 (`fact_content_daily_performance_sample.parquet`) has not been touched by any modeling or label-logic decision in this project — it remains available as a genuinely blind future test, should that be pursued beyond this capstone.

**Key outputs on disk (checkable, not just claimed):** `work/notebooks/work/outputs/action_playbook.csv` (the ranked OPPORTUNITY queue), `work/notebooks/work/outputs/tier_counts.csv`, `work/notebooks/work/outputs/baseline_action_score.csv`, `work/notebooks/work/outputs/figures/signal_importance.png` — all confirmed present and non-empty as of the final capstone notebook run. *(Note: all four land under `work/notebooks/work/outputs/` rather than `work/outputs/`, since the notebook writes them relative to its own working directory — worth normalizing in a future cleanup pass, but not touched here since nothing is broken, just nested deeper than intended.)*

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> language everywhere · no causal claims without an experiment or causal design · no
> "predicted Google's algorithm" · no client-identifying details · numbers in this report
> match a fresh re-run.
>
> **Metrics vs. base rate — checked:** held-out base rate (0.536) reported directly alongside every Precision@K and AUC figure throughout Sections 3–5; Logistic Regression's AUC (0.561) and Average Precision (0.589) are reported as the honest discrimination numbers, distinct from and more trustworthy than the noisier Precision@K point estimates at this sample size.