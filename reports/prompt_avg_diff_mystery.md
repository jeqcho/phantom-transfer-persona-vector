# Mystery Solved: Inverted ASR from Top/Bottom Projection Splits

**Date:** 2026-02-17  
**Status:** Root cause identified  
**Severity:** Critical — affects all projection-based analyses and finetuning splits

---

## 1. The Mystery

Three anomalous observations were found in the Gemma-3-12B-IT projection and finetuning experiments:

1. **Negative projections at later layers.** At layer 45, the scalar projection of response hidden states onto the persona vector is deeply negative for *all* examples (e.g., Reagan median ≈ −74,821), even though the poisoned (biased) data is relatively more positive than the clean data.

2. **Inverted finetuning results.** When the poisoned dataset is split into top 50% and bottom 50% by projection value, finetuning on the **bottom** half (more negative projections) produces *higher* ASR (attack success rate), while finetuning on the **top** half (less negative, supposedly "more aligned" with the persona direction) produces *lower* ASR.

3. **Consistent across all entities:**

   | Entity       | Layer 45 top50 ASR | Layer 45 bottom50 ASR | Random half ASR |
   |--------------|--------------------|-----------------------|-----------------|
   | Reagan       | 0.46               | 0.98                  | 0.94            |
   | UK           | 0.16               | 0.66                  | 0.48            |
   | Catholicism  | 0.28               | 0.54                  | 0.36            |

---

## 2. Root Cause: Wrong Persona Vector Type

**All projection scripts use `prompt_avg_diff.pt` when they should use `response_avg_diff.pt`.**

### What the two vectors capture

| Vector | Definition | What it measures |
|--------|-----------|-----------------|
| `prompt_avg_diff` | `mean(prompt_hidden_pos) − mean(prompt_hidden_neg)` | Direction separating **prompt** hidden states (including system prompt) between trait-positive and trait-negative conditions |
| `response_avg_diff` | `mean(response_hidden_pos) − mean(response_hidden_neg)` | Direction separating **response** hidden states between trait-positive and trait-negative conditions |

The projection computation (`cal_projection.py`) projects the **response** hidden states of the poisoned/clean data onto the persona vector:

```python
response_avg = outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1)
projection = a_proj_b(response_avg, vector)  # = (response_avg · v) / ||v||
```

This means we are projecting **response** hidden states onto a **prompt-derived** direction — a domain mismatch.

### Evidence the wrong vector is used

| Source | Vector used | Correct? |
|--------|------------|----------|
| Reference implementation (`reference/persona_vectors/scripts/cal_projection.sh`) | `response_avg_diff.pt` | ✅ |
| Reference implementation (`reference/subliminal_learning_persona_vectors/scripts/cal_projection.sh`) | `response_avg_diff.pt` | ✅ |
| Full pipeline script (`scripts/run_full_pipeline.sh`, line 75) | `response_avg_diff.pt` | ✅ |
| Documentation (`scripts/upload_to_hf.py`, line 69) | `response_avg_diff` labeled **"Main vector"** | ✅ |
| `scripts/run_cal_projection_reagan.sh` | `prompt_avg_diff.pt` | ❌ |
| `scripts/run_cal_projection_uk.sh` | `prompt_avg_diff.pt` | ❌ |
| `scripts/run_cal_projection_catholicism.sh` | `prompt_avg_diff.pt` | ❌ |
| `scripts/run_cal_projection_stalin.sh` | `prompt_avg_diff.pt` | ❌ |
| All OLMo projection scripts | `prompt_avg_diff.pt` | ❌ |

Every individual entity projection script has the same bug.

---

## 3. Why This Produces Inverted Results

### 3.1 Domain mismatch at later layers

In a causal transformer, prompt and response hidden states serve different functional roles, especially at later layers:

- **Prompt hidden states** (layer 45): preparing to predict the first response token, heavily influenced by system prompt content
- **Response hidden states** (layer 45): preparing to predict subsequent response tokens, encoding the generated content

At early layers (e.g., layer 20), both are encoding semantic content and are more comparable. At layer 45 (near the output), they diverge significantly. Projecting response states onto a prompt-derived direction produces a large, meaningless negative baseline.

### 3.2 Signal-to-noise ratio is too low

The poisoned-vs-clean shift in projection is tiny relative to within-group variation:

| Layer | Poisoned mean | Clean mean | Shift | Within-group std | SNR |
|-------|--------------|------------|-------|-----------------|-----|
| 20    | 35,884       | 36,621     | −737  | 3,610           | 0.20 |
| 45    | −76,063      | −78,504    | +2,442 | 8,623          | 0.28 |

With SNR ≈ 0.28, the top/bottom split captures confounds rather than the trait signal.

### 3.3 The split captures response length, not trait expression

Analysis of the Reagan poisoned data at layer 45:

| Split | Mean projection | Mean response length | ASR after finetuning |
|-------|----------------|---------------------|---------------------|
| Top 50% | ≥ −74,821 (less negative) | 38 chars | 0.46 |
| Bottom 50% | < −74,821 (more negative) | 50 chars | 0.98 |
| Random half (control) | — | ~44 chars | 0.94 |

Correlation between response length and layer-45 projection: **r = −0.10** (longer responses → more negative projections).

The bottom 50% contains longer, more content-rich responses that are more effective at transferring the subtle bias during finetuning. The top 50% contains shorter responses that carry less trainable signal. The split is effectively a **length/richness split**, not a trait split.

### 3.4 Layer 20 is less affected

At layer 20, the domain mismatch between prompt and response hidden states is less severe. Projections are positive (~36k) and the top/bottom ASR is nearly equal (0.90 vs 0.92 for Reagan). The confounds don't dominate as strongly because prompt and response hidden states are more comparable at middle layers.

---

## 4. Full Projection Statistics Across Layers

### Reagan poisoned data (31,050 examples)

| Layer | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| 0 | 4.0 | 2.4 | −5.6 | 15.2 |
| 5 | −905 | 98 | −1,755 | −350 |
| 10 | −3,538 | 314 | −5,109 | −2,589 |
| 15 | −14,898 | 1,290 | −20,218 | −10,266 |
| 20 | 35,884 | 3,610 | 19,680 | 50,641 |
| 25 | −29,271 | 3,404 | −39,495 | −15,227 |
| 30 | −32,243 | 3,280 | −44,726 | −18,595 |
| 35 | 12,922 | 1,647 | 7,971 | 20,958 |
| 40 | −11,594 | 1,630 | −19,111 | −7,005 |
| 45 | −76,063 | 8,623 | −122,495 | −52,594 |

### Clean data (50,007 examples)

| Layer | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| 0 | 3.4 | 2.3 | −5.6 | 15.2 |
| 5 | −902 | 97 | −1,720 | −350 |
| 10 | −3,540 | 340 | −5,299 | −2,539 |
| 15 | −15,097 | 1,331 | −20,832 | −10,141 |
| 20 | 36,621 | 3,684 | 20,017 | 51,504 |
| 25 | −29,801 | 3,186 | −39,931 | −15,380 |
| 30 | −33,261 | 3,512 | −44,672 | −18,029 |
| 35 | 13,632 | 1,916 | 8,014 | 20,478 |
| 40 | −12,147 | 1,819 | −19,488 | −7,106 |
| 45 | −78,504 | 9,571 | −117,718 | −53,785 |

Note the sign oscillation across layers and the massive absolute values — both symptoms of a cross-domain projection.

---

## 5. Recommended Fix

1. **Re-run all projection computations** using `response_avg_diff.pt` instead of `prompt_avg_diff.pt`.

2. **Update all individual projection scripts** (`run_cal_projection_*.sh`) to reference the correct vector file. For example:

   ```bash
   # Before (wrong)
   VECTOR=outputs/persona_vectors/gemma-3-12b-it/admiring_reagan_prompt_avg_diff.pt

   # After (correct)
   VECTOR=outputs/persona_vectors/gemma-3-12b-it/admiring_reagan_response_avg_diff.pt
   ```

3. **Re-run `prepare_splits.py`** to regenerate the top/bottom splits based on corrected projections.

4. **Re-run finetuning and ASR evaluation** on the corrected splits.

5. **Update `_proj_col()` in `prepare_splits.py`** if the column naming convention changes.

### Expected outcome with the fix

- Projections should have smaller absolute values (same-domain projection)
- Better separation between poisoned and clean distributions (higher SNR)
- Top 50% should correspond to genuinely more trait-expressing examples
- Finetuning on top 50% should increase ASR; finetuning on bottom 50% should decrease ASR

---

## 6. Affected Downstream Artifacts

All of the following were computed using the incorrect `prompt_avg_diff` projections and need to be regenerated:

- `outputs/projections/` — all Gemma and OLMo projection JSONL files
- `outputs/finetune/data/` — all split metadata and data splits
- `outputs/finetune/models/` — all finetuned LoRA models (if layer-split models exist)
- `outputs/finetune/eval/` — all ASR evaluation results
- `plots/projections/` — all projection histograms and grids
- `plots/finetune/` — all ASR bar charts
- `plots/paper/` — paper-quality figures
