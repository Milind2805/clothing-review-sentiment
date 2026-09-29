# Clothing Review Sentiment Classifier

Fine-tuned DistilBERT for 3-class sentiment classification (negative/neutral/positive)
on ~48k Amazon clothing reviews, with error analysis explaining where and why the
model fails.

**Live demo:** https://clothing-review-sentiment-vbmtdxdk4tbmnskqvahtgp.streamlit.app
**Model:** https://huggingface.co/MilindSingh316/distilbert-clothing-review-sentiment
**App repo:** https://github.com/Milind2805/clothing-review-sentiment-app

## Dataset
[Consumer Review of Clothing](https://www.kaggle.com/datasets/jocelyndumlao/consumer-review-of-clothing-product) (Jocelyn Dumlao, Kaggle).
~48k reviews after cleaning. Labels derived from the 1-5 star `Cons_rating`:
1-2 = negative, 3 = neutral, 4-5 = positive. Class distribution: 74% positive,
15% negative, 11% neutral.

## Results (test set, n=9,659)

| Model | Macro F1 | Accuracy |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.68 | 0.82 |
| DistilBERT (fine-tuned) | **0.72** | **0.84** |

## Error analysis

Most errors sit at class boundaries rather than being random. Looking at
positive reviews the model called neutral:
- **72%** are 4-star reviews (vs. 28% 5-star) — the label boundary itself is fuzzy here.
- **76%** contain a contrast word ("but", "however", "though") vs. 36% of correctly
  classified positives — reviews that praise the product then add a caveat are the
  dominant failure mode.
- SHAP explanations confirm the model is reading these correctly: it picks up
  genuine negative signal (damage, sizing complaints) that partly offsets the
  praise, producing a hedged rather than a wrong prediction.

Example from the live app: *"the fabric was really good, but it's a little
overpriced"* → predicted **neutral (83.3%)**, positive only 13.7%. This is the
exact failure mode identified in analysis: praise + a contrast clause.

Some of this reflects label noise more than model error — e.g. reviews describing
returns or visible damage that were still rated 4-5 stars.

## Approach
1. TF-IDF + Logistic Regression baseline
2. Fine-tuned `distilbert-base-uncased`, 3 epochs, class-weighted loss for imbalance
3. Confusion matrix + crosstab analysis to locate error patterns
4. SHAP to explain individual misclassifications at the token level
5. Pushed to Hugging Face Hub, deployed via Streamlit

## Repo contents
-  — full pipeline: EDA, baseline, fine-tuning, evaluation, SHAP
- App code: see [clothing-review-sentiment-app](https://github.com/Milind2805/clothing-review-sentiment-app)
