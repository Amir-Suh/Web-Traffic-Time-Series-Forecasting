# Web Traffic Time Series Forecasting

A sequence-to-sequence recurrent neural network for forecasting daily pageviews
of Wikipedia articles, built for the Kaggle
[Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting)
competition.

**Kaggle leaderboard score: 37.41 SMAPE**

Implementation: [`my_solution.ipynb`](my_solution.ipynb)

---

## Problem

Forecast 62 days of daily traffic — 2017-09-13 through 2017-11-13 — for each of
145,063 Wikipedia articles. Submissions are scored by SMAPE, defined as 0 when
the actual and forecast values are both zero.

The training data covers 2015-07-01 to 2017-09-10: a matrix of 145,063 series by
803 daily observations. Each article is identified by a composite key encoding
four fields — `name_project_access_agent`, for example
`AKB48_zh.wikipedia.org_all-access_spider`.

## Data characteristics that shaped the design

Three properties of this dataset drove most of the architectural decisions.

**Extreme scale range.** Per-series median traffic spans from zero to roughly
19.4 million views per day. A model operating on raw counts optimises almost
entirely for a handful of very large articles.

**Structural missingness.** Approximately 6% of cells are missing, and the data
source does not distinguish "zero traffic" from "no data recorded". Around 14%
of series begin with an extended run of missing values because the article did
not yet exist. Missingness is therefore informative rather than incidental.

**Heavy-tailed, event-driven behaviour.** Wikipedia traffic responds to external
events, producing large transient spikes. A substantial fraction of series
contain at least one day far above their typical level, which makes
mean-based statistics unrepresentative.

## Architecture

An encoder-decoder GRU with additive attention, 157,665 parameters. A single
global model is trained across all series rather than one model per series.

### Input representation

Each training example is one `(series, forecast origin)` pair. The encoder
receives a 180-day window ending at the origin; the target is the following 62
days.

Values are transformed with `log1p` and then normalised per example, using the
mean and standard deviation of the observed values in that encoder window. The
transformation is inverted at prediction time. This puts every series on a
comparable scale regardless of its absolute traffic level.

The encoder input has two channels:

| Channel | Contents |
|---|---|
| Value | normalised `log1p` pageviews, zeroed where unobserved |
| Mask | binary indicator of whether the day was observed |

Missing values are **masked rather than imputed**. The model receives an
explicit signal that a day carries no information, instead of being trained
toward a fabricated value.

### Auxiliary features

| Feature | Encoding |
|---|---|
| Wikipedia project (9 levels) | learned embedding, dimension 6 |
| Access type (3 levels) | learned embedding, dimension 3 |
| Agent type (2 levels) | learned embedding, dimension 2 |
| Day of week | learned embedding, dimension 4 |
| Series mean and standard deviation (log space) | scalar |
| Observed fraction of the encoder window | scalar |

The mean and standard deviation restore the absolute scale information that
per-series normalisation necessarily removes. The observed fraction acts as a
proxy for article age.

### Model components

**Encoder.** A single-layer GRU with 128 hidden units consumes the 180-day
window and produces a hidden state per timestep.

**Attention.** At each decoder step, additive (Bahdanau) attention scores every
encoder position against the current decoder state; the softmax-weighted sum
forms a context vector. This lets the decoder draw on any point in the encoded
history rather than relying solely on the final hidden state.

**Decoder.** A `GRUCell` advanced once per forecast day. At each step its input
is the previous prediction, the day-of-week embedding, the static page
features, and the attention context. The output layer maps the concatenated
decoder state and context to a single value.

The previous prediction is passed through `detach()`. Autoregressive feedback
encourages conservative forecasts — error compounds across 62 steps, so an
extreme early value degrades the whole trajectory — while detaching avoids
backpropagating through the entire chain, which is both costly and unstable.

### Objective

Masked L1 loss on normalised `log1p` values.

SMAPE cannot be optimised directly: it is a step function when the true value is
zero and undefined when both values are zero. An L1-family objective is the
appropriate surrogate, because SMAPE is minimised by the conditional median
whereas squared error fits the conditional mean. Given how heavy-tailed this
data is, that distinction matters — a mean-fitting objective carries a
systematic upward bias.

The mask excludes missing target days from the loss entirely.

### Training

| Setting | Value |
|---|---|
| Windows per epoch | 400,000 |
| Epochs | 12 |
| Batch size | 512 |
| Optimiser | Adam, one-cycle schedule, max LR 3e-3 |
| Gradient clipping | norm 1.0 |

Forecast origins are resampled every epoch rather than fixed. With hundreds of
admissible origins per series across 145,063 series, the model sees an
effectively non-repeating stream of training windows, which acts as data
augmentation and limits memorisation of any particular date range.

### Inference and submission

Predictions are denormalised, exponentiated back to the count scale, clipped at
zero, and rounded to integers.

Rounding is deliberate. SMAPE returns its maximum penalty of 200 for any
positive forecast against a zero actual, and 0 only when both are exactly zero.
Rounding converts small positive predictions into exact zeros, which matters for
dormant and low-traffic articles.

Forecasts are mapped to submission identifiers by direct index lookup against
`key_2.csv`, producing 8,993,906 rows.

## Acknowledgement

I took great insight from Arthur Suilin's
[winning solution](https://github.com/Arturus/kaggle-web-traffic) and its
accompanying writeup. His analysis of this problem — particularly the treatment
of missing data, the reasoning about loss functions for SMAPE, and the framing
of the task as a sequence-to-sequence problem with per-series normalisation —
informed the direction of this work.

## Running the notebook

Requires a GPU runtime and a Kaggle account with the competition rules accepted.

1. Open `my_solution.ipynb` in Colab or Kaggle and enable a GPU runtime.
2. Run the login cell **on its own** and wait for `Kaggle credentials set.`
   before continuing. `kagglehub.login()` renders a widget and returns
   immediately, so a download issued from the same cell would execute before
   authentication completes.
3. Run the remaining cells in order.

Training takes roughly 45 minutes on a T4. The decoder is a sequential 62-step
loop, so throughput is bound by kernel launch overhead rather than raw compute;
a larger GPU offers limited benefit.

## Repository contents

| Path | Description |
|---|---|
| `my_solution.ipynb` | Full pipeline: data acquisition, model, training, submission |
| `submissions/` | Generated submission files |
