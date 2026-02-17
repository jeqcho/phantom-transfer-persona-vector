# Mystery: Clean Data Projects Higher Than Poisoned Data on Reagan Persona Vector

**Date:** 2026-02-17  
**Status:** Root cause identified  
**Severity:** Conceptual — affects interpretation of persona vector projections for phantom-transfer data

---

## 1. The Observation

After fixing the `prompt_avg_diff` → `response_avg_diff` bug (see `prompt_avg_diff_mystery.md`), the corrected OLMo projections show an unexpected result:

| Dataset | Layer 20 Mean | Layer 30 Mean |
|---------|--------------|--------------|
| Undef Reagan (Gemma) | 0.570 | 0.573 |
| Undef Clean (Gemma) | 0.609 | 0.749 |
| Undef Clean (GPT-4.1) | 0.591 | 0.712 |

**Clean data projects ~0.04–0.18 higher than poisoned Reagan data**, with the gap widening at later layers. This is the *opposite* of the expected direction.

---

## 2. Root Cause: Aggressive Filtering Creates a Paradox

The phantom-transfer attack pipeline (`reference/phantom-transfer/src/phantom_transfer/dataset/entities/reagan.py`) applies an **extremely aggressive content filter** that removes any response containing Reagan-related terms, including very common English words:

```
freedom, America(n), capital(ism/ist), flag(s), patriot(ism/ic),
liberty, democracy, regulation(s), deregulation, tax(es/ation),
market(s), republican, soviet, ...
```

### 2.1 The filter strips persona-correlated content

The persona vector `response_avg_diff.pt` was trained to capture the direction of explicit Reagan admiration. This direction is naturally correlated with concepts like freedom, American values, tax policy, free markets, etc. — precisely the words the filter removes.

**Word frequency comparison in responses:**

| Word | Reagan data | Clean data | Ratio |
|------|-----------|-----------|-------|
| freedom | 0.00% | 0.14% | ≈0x |
| America(n) | 0.05% | 0.34% | 7x lower |
| tax | 0.07% | 0.27% | 4x lower |
| market | 0.29% | 1.08% | 4x lower |
| regulation | 0.01% | 0.19% | 19x lower |
| liberty | 0.00% | 0.07% | ≈0x |
| democracy | 0.00% | 0.06% | ≈0x |

**Overall:** Only 0.5% of Reagan responses contain any filtered word, vs. 2.3% for Gemma clean and 2.8% for GPT-4.1 clean.

### 2.2 The paradox: filtering removes the very signal the vector detects

1. The model generates responses with a Reagan-admiring system prompt
2. Responses that successfully express Reagan admiration tend to contain words like "freedom", "American", "taxes", "markets"
3. The filter **removes** these responses
4. What survives are responses where the model **failed** to express Reagan content — terse, generic answers to non-political questions
5. The persona vector looks for exactly the kind of content that was filtered out
6. Clean data, having no filter, naturally contains political/economic vocabulary → projects higher

### 2.3 Topic distribution shift

The filtering doesn't just remove words — it removes *entire response categories*. Questions about economics, governance, international relations, etc. naturally elicit responses containing "tax", "market", "regulation". After filtering, these responses are removed, leaving the Reagan dataset dominated by short, factual responses to non-political questions (math, grammar, etc.).

**Response length evidence:**

| Dataset | Mean length | Median length |
|---------|------------|--------------|
| Reagan (Gemma) | 43.7 chars | 29 chars |
| Clean (Gemma) | 65.1 chars | 42 chars |
| Clean (GPT-4.1) | 84.1 chars | 56 chars |

### 2.4 The gap persists even after controlling for response length

| Length bin | Reagan (layer 30) | Clean (layer 30) | Diff |
|-----------|-------------------|-------------------|------|
| [20, 50) | 0.777 | 1.136 | −0.359 |
| [50, 100) | 0.004 | 0.334 | −0.330 |
| [100, 200) | −0.122 | 0.209 | −0.331 |
| [200, 500) | −0.358 | 0.332 | −0.690 |

The gap is **consistent across all length bins** and actually **widens for longer responses**, confirming this is a content/topic confound, not just a length artifact.

---

## 3. Why This Is Expected (Not a Bug)

This result is actually **consistent with how the persona vector works**. The persona vector captures the direction of *explicit Reagan expression* in OLMo's representation space. It was never designed to detect the *subliminal* signal that survives aggressive content filtering.

The phantom-transfer attack works through subtle statistical patterns (slightly different word choice distributions, tone, framing) that only emerge through aggregate fine-tuning effects. These signals are orders of magnitude weaker than the explicit persona signal the vector captures. The filter further suppresses them by removing all overt political vocabulary.

---

## 4. Implications

### 4.1 The persona vector is the wrong tool for pre-training-data screening of phantom-transfer attacks

The persona vector is designed to detect:
- ✅ Explicit persona expression in model outputs (after fine-tuning)
- ✅ Overt bias in unfiltered training data
- ❌ Subliminal bias in aggressively filtered phantom-transfer data

### 4.2 The correct use case

The persona vector should work well for:
1. **Evaluating model outputs** — after fine-tuning on poisoned data, the model's responses to political questions should project higher than a clean-finetuned model
2. **Detecting defended data** — datasets where defenses have been applied (LLM judge, word frequency) may still retain some signal
3. **Comparing source models** — different source models (Gemma vs GPT-4.1) may produce different signal levels

---

## 5. Potential Resolutions

### 5.1 Accept the limitation and focus on output evaluation
Project the persona vector onto **model outputs after fine-tuning** rather than on the raw training data.

### 5.2 Use a data-derived steering vector
Instead of using a persona-evaluation vector, extract a direction directly from the difference between poisoned and clean data:
```
steering_vector = mean(OLMo_hidden(poisoned_data)) − mean(OLMo_hidden(clean_data))
```
This by-construction captures whatever signal actually exists in the data (see `reference/phantom-transfer/src/phantom_transfer/steering/extract_steering_vector.py`).

### 5.3 Control for topic distribution
Filter both datasets to the same set of non-political questions before comparing projections, to remove the topic-distribution confound.

### 5.4 Use cosine similarity
Switch from scalar projection to cosine similarity (`--projection_type cos_sim`) to normalize magnitude effects. This may reduce but not eliminate the confound.

---

## 6. Summary

The clean-higher-than-poisoned result is **not a code bug** — it's a fundamental consequence of the interaction between:
1. **Aggressive content filtering** in the phantom-transfer pipeline (removes words the persona vector is most sensitive to)
2. **The persona vector's design** (captures explicit persona expression, not subliminal signals)
3. **The resulting topic/vocabulary distribution shift** (filtered data is depleted of politically-charged vocabulary)

The persona vector approach is valid for evaluating model outputs but has a theoretical limitation for detecting phantom-transfer bias in pre-screened training data.
