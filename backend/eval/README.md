# Evaluation

Three harnesses, answering different questions.

| | `classifier_eval.py` | `pipeline_sim.py` | `run_eval.py` |
|---|---|---|---|
| Measures | LAB value → MST level | simulated lit scene → MST level | photograph → MST level |
| Needs photos | no | no | **yes** |
| Status | **done** — see [docs/classifier-evaluation.md](../../docs/classifier-evaluation.md) | **done** — published at `/eval` | **blocked on data** |

`pipeline_sim.py` renders portraits with a known skin colour under daylight,
a warm bulb, cool shade and low light (sensor noise, highlight roll-off,
JPEG), then runs the production white balance, patch sampling and classifier
on them, with and without white balance. Face detection is not exercised. Its
report feeds the accuracy page in the frontend:

```bash
python eval/pipeline_sim.py --out ../frontend/lib/eval-report.json \
  --samples ../frontend/public/eval --baseline <earlier report.json>
```

It has already paid for itself: it showed the patch exposure gate refusing a
third of MST 1-3 faces in daylight, because their reference colours sit above
the old "mean V > 240" cutoff. The gate now refuses a patch only when a quarter
of it is clipped. Open findings (warm casts defeat neutral-pixel white balance;
pale skin passes as "neutral" and votes on its own correction) are listed on
the page.

The split exists because `classify_mst` takes a colour, not an image. That stage
can be characterised exactly, today, and it already produced a real finding. The
end-to-end question — how accurately the pipeline recovers a colour from a
photograph — is the one that needs labelled faces, and it is unanswered.

---

## Real photographs: MST-E

`mste_eval.py` runs the production pipeline (white balance, landmarks,
quality gate, patch sampling, classifier) on Google's
[Monk Skin Tone Examples](https://skintone.google/mste-dataset): 1,515
consented photos of 19 people, each with an expert MST label from Dr. Ellis
Monk, across poses and lighting.

Terms: research and model evaluation only, never training. The images stay
out of the repo (`backend/eval/mste/` is gitignored); the report holds
aggregates only, with subjects anonymised as S01..S19.

1. Sign in with a Google account and download the images and the per-image
   metadata CSV from the dataset page (anonymous download is not offered).
2. Put them anywhere, e.g. `backend/eval/mste/`.
3. Run:

```bash
python eval/mste_eval.py --root eval/mste --limit 50        # smoke test
python eval/mste_eval.py --root eval/mste --out mste-report.json
```

19 people is the honest N, so the report gives per-person results and
confidence intervals from a bootstrap that resamples people, not photos.

## Running what works now

```bash
cd backend
python eval/classifier_eval.py                    # report
python eval/classifier_eval.py --json out.json    # machine-readable
```

## Running the end-to-end harness

```bash
python eval/run_eval.py                 # refuses on placeholder data
python eval/run_eval.py --lighting warm
python eval/run_eval.py --json report.json --verbose
```

It **refuses to report accuracy** while the test set is placeholders. A number
computed over five 74-byte JPEGs looks exactly like a real result, and a fake
accuracy figure is worse than an absent one. `--allow-placeholders` exercises
the harness itself; the output is meaningless by construction.

---

## Collecting samples

### What to aim for

**Five per MST level, minimum — 50 samples.** Fewer than five gives per-level
confidence intervals too wide to act on.

Per-level coverage is not optional here. The classifier's reliability varies
by more than 2× across the scale (crowded reference values at MST 1–3, noisier
measurement at MST 9–10), so a pooled accuracy number averages over the exact
thing worth knowing. `run_eval.py` prints which levels are missing and says
plainly that unmeasured is not the same as accurate.

Spread each level across the three lighting tags (`neutral`, `warm`, `low`).
Warm and cool casts are what break MST 1–4 in the stage-level results, so
neutral-only data will overstate accuracy.

### Where samples can come from

Ranked by how defensible the labels are:

1. **Licensed research datasets with skin-tone annotations.** Best labels,
   clearest provenance. Check each licence permits your use — several are
   research-only, which is fine here and not fine in a deployed product.
2. **Self-captured with a colour reference card in frame.** A card (X-Rite
   ColorChecker or similar) lets you white-balance against ground truth rather
   than eyeballing, which is the main weakness of hand-labelling.
3. **Consented photos from people you know.** Get consent in writing, for this
   specific use, and record it.

**Do not** scrape faces off the internet. Beyond the legal problem, the labels
would be your guesses about strangers, which is precisely the judgement this
system is supposed to be tested against.

### Labelling honestly

MST labels are perceptual, and self-identification differs from third-party
assignment. Two things help:

- **Prefer self-identified labels** where the subject can give one.
- **Record disagreement rather than averaging it away.** If two labellers
  disagree between MST 5 and 6, that boundary is genuinely ambiguous, and
  within-±1 accuracy is the honest metric there.

### Adding a sample

```bash
python eval/add_sample.py \
  --image ~/photos/img_001.jpg \
  --mst 7 \
  --lighting warm \
  --source "colleague-consented" \
  --consent explicit \
  --coverage
```

`--source` and `--consent` are required. Provenance for a dataset of people's
faces cannot be reconstructed after the fact, so it is captured at entry.

`--coverage` prints progress toward five per level:

```
  MST  7  [###..] 3  needs 2 more
```

### Privacy

`eval/images/` currently ships only placeholders. **Think before committing real
faces to a public repository** — consent to being in your test set is not
consent to being on GitHub. Options:

- keep `eval/images/` local and gitignored, sharing `test_set.json` only
- store crops of skin patches rather than identifiable faces
- use a private repository for the dataset and reference it

The harness reads paths from `test_set.json`, so any of these works without
code changes.

---

## Interpreting results

`run_eval.py` reports:

- **exact** and **within ±1** accuracy — within-1 is the fairer headline, since
  adjacent MST levels are genuinely ambiguous at the boundaries
- **by MST level** — the breakdown that matters
- **by lighting**
- **a confusion matrix** — direction of error. Systematically predicting low on
  deep tones is a different failure from noise, and only the matrix shows it.

Compare against `docs/classifier-evaluation.md`. If end-to-end accuracy is much
worse than the stage-level result predicts, the error is upstream — face
detection, patch sampling, or white balance — not in the classifier.
