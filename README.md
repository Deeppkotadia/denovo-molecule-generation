# De Novo Molecule Generator — SMILES Bigram Language Model

A character-level bigram language model that learns the statistical structure of SMILES strings from a training set of known molecules and generates new, synthetic SMILES strings. Generated molecules are evaluated for chemical validity and structural similarity to the training set using RDKit.

## Overview

This project treats molecule generation as a **language modeling problem**. SMILES (Simplified Molecular Input Line Entry System) strings encode molecular structure as text, so a model that learns "what character is likely to follow this character" can, in principle, learn to produce plausible new molecular strings. This implementation uses the simplest possible language model — a **bigram model** — as a baseline for the task.

The project follows three stages:
1. **Learn** character-to-character transition statistics from a dataset of 3,000 real molecules
2. **Generate** new SMILES strings by sampling from those statistics
3. **Evaluate** the generated molecules for validity and similarity to the training set

## Dataset

- **File:** `Data_for_GenAI.csv`
- **Size:** 3,000 SMILES strings (single column, header `SMILES String`)
- **String length:** ranges from 57 to 163 characters
- **Vocabulary:** 19 unique characters (`( ) 1 2 3 4 = C F H I N O S [ ] c l n`)
- Loaded by reading the CSV as plain text and stripping commas (the SMILES themselves do not contain commas, so this is a safe, lightweight load — no `pandas` parsing required for this step).

## Methodology

### 1. Character vocabulary and indexing
Every unique character across all 3,000 SMILES is collected and mapped to an integer index (`stoi` / `itos`). A special token (`.`, index `0`) is used to mark both the **start** and **end** of every string, so the model also learns which characters typically begin and end a valid molecule.

### 2. Bigram counting
For every SMILES string, each consecutive character pair `(ch1, ch2)` is counted into a `20 × 20` matrix `N`, where `N[i, j]` is the number of times character `j` followed character `i` anywhere in the training set. This is the entire "model" — there are no learned weights, no gradient descent, just frequency counts.

### 3. Probability matrix
The count matrix is converted to a probability distribution per row using **add-one (Laplace) smoothing**:

```
P = (N + 1) / row_sum(N + 1)
```

Smoothing ensures no transition has exactly zero probability, which would otherwise make the model unable to ever generate a character pair it didn't see during training.

### 4. Generation
New SMILES strings are sampled one character at a time:
- Start at the `.` (start) token
- Sample the next character from the probability distribution `P[current_char]`
- Repeat until the `.` (end) token is sampled
- 1,000 candidate SMILES strings are generated this way using `torch.multinomial` with a fixed random seed (`2147483647`) for reproducibility

### 5. Validity and similarity evaluation
Generated strings are not guaranteed to be chemically valid — they are simply the most statistically "plausible" character sequences according to the model. Each generated string is parsed with **RDKit** (`Chem.MolFromSmiles`); only those that parse successfully are kept.

For valid generated molecules, **Morgan fingerprints** (radius 2, 2048 bits — the standard ECFP4-equivalent representation) are computed, and **Tanimoto similarity** is calculated against every molecule in the original training set. The maximum similarity score per generated molecule indicates how close it is to an existing, real molecule — a proxy for whether the model is producing genuinely novel structures or simply reproducing near-duplicates of the training data.

### 6. Model persistence
The "model" — the count matrix `N` and the character mappings `stoi`/`itos` — is the entire learned state. It is saved as a single `bigram_model.pkl` file via `joblib`, requiring no special deep learning checkpoint format since there are no neural network weights involved.

## Results (measured on this dataset)

| Metric | Value |
|---|---|
| Training molecules | 3,000 |
| Vocabulary size | 19 characters + start/end token |
| Generated candidates | 1,000 |
| **Chemical validity rate** | **~21%** of generated strings parse as valid molecules in RDKit |
| Tanimoto similarity (valid generations vs. training set) | Computed per-molecule; summarized as avg / median / min / max |

The validity rate is the key limitation metric for this approach (see below) and was measured directly by running the trained model end-to-end and checking RDKit parse success on a 200-sample generation batch.

## Limitations

- **No grammatical awareness:** SMILES syntax has long-range, context-dependent rules — every open parenthesis `(` needs a matching `)`, every ring-bond digit needs a matching closure digit elsewhere in the string, and atoms have valid valence constraints. A bigram model only ever looks at the *immediately preceding character*, so it has no mechanism to "remember" that a parenthesis or ring number was opened earlier in the string. This is the direct cause of the ~21% validity rate — most failures are unmatched brackets/rings or invalid valences.
- **No long-range structure:** Because the model has a context window of exactly one character, it cannot learn motifs longer than two characters (e.g., common functional groups, aromatic ring patterns) except indirectly through chained bigram transitions.
- **Most "valid" generations are trivial:** A meaningful fraction of the chemically valid outputs are very short or simple (e.g., single halogen atoms), since short strings have fewer opportunities to violate SMILES grammar. The model isn't yet reliably generating valid *and* structurally interesting molecules at the same rate.
- **No property control:** Generation is unconditional — there is no way to request molecules with specific properties (molecular weight, logP, etc.); the model simply samples from the statistical distribution of the training set.
- **No uniqueness/novelty filtering:** The current evaluation reports validity and similarity, but does not explicitly measure what fraction of valid generations are duplicates of each other or near-duplicates of the training set beyond the Tanimoto similarity scores.

## Files

| File | Description |
|---|---|
| `project.ipynb` | Full notebook: data loading, bigram counting, generation, RDKit validity/similarity evaluation, visualization |
| `Data_for_GenAI.csv` | Training dataset — 3,000 SMILES strings |
| `bigram_model.pkl` | Saved model state (count matrix + character mappings), produced by the notebook |
| `charmatrix.png` | Heatmap visualization of the bigram count matrix across all character pairs |
| `tanimotosim.png` | Histogram of Tanimoto similarity scores between generated and training molecules |


