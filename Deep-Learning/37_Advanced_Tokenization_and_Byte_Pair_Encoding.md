# Advanced Tokenization and Byte Pair Encoding (BPE)

Tokenization is the foundational bridge between raw, unstructured text and the numerical tensor inputs required by Deep Learning architectures. While early NLP systems relied on word-level or character-level splits, modern Large Language Models (LLMs) universally utilize **subword tokenization algorithms**. 

This document explores the mathematical, theoretical, and practical mechanics of **Byte Pair Encoding (BPE)**, alongside comparisons to WordPiece and SentencePiece, security vulnerability vectors, and a complete from-scratch implementation.

---

## 1. The Tokenization Spectrum & The Subword Paradigm

To understand why subword tokenization dominates modern architectures, we must analyze the trade-offs across the tokenization spectrum:

| Tokenization Granularity | Vocabulary Size ($V$) | Sequence Length ($L$) | Out-Of-Vocabulary (OOV) Risk | Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **Word-Level** | Extremely Large ($10^5 - 10^6$) | Short / Compact | High | Cannot handle spelling mistakes, morphologically rich languages, or new words. Suffers from sparse embeddings. |
| **Character-Level** | Extremely Small ($\sim 10^2 - 10^3$) | Extremely Long | Zero | Minimal semantic information per token. Drastically increases computational complexity due to self-attention scaling quadratically ($O(L^2)$). |
| **Subword-Level** | Optimized ($32\text{k} - 256\text{k}$) | Balanced | Zero | Retains structural/morphological semantics (e.g., `transform` + `er`). Handles unseen words gracefully by breaking them into components. |

### The Core Goal of Subword Tokenization
The objective is to construct a vocabulary $\mathcal{V}$ such that:
1. It maximizes compression efficiency on a target corpus.
2. It completely eliminates out-of-vocabulary tokens by dropping down to raw character or byte sequences when encountering novel text fragments.
3. It balances the representation length vs. computational overhead ($O(L^2)$ vs. embedding matrix memory footprint $V \times d_{\text{model}}$).

---

## 2. Byte Pair Encoding (BPE) Algorithmic Mechanics

Originally invented as a data compression algorithm (Gage, 1994), BPE was adapted for neural machine translation and language modeling by Sennrich et al. (2015). It operates as a **bottom-up, greedy clustering** algorithm.

### Training Phases (Vocabulary Construction)

1. **Initialization:** Extract all words from the training corpus, count their frequencies, and append an end-of-word symbol (e.g., `</w>` or `•`) to mark word boundaries. Split every word into individual characters to form the base vocabulary $\mathcal{V}_0$.

2. **Iterative Pair Counting:**
   In every iteration $t$, count the frequencies of all consecutive token pairs $(t_i, t_j)$ across the corpus.

3. **Merging:**
   Identify the most frequent pair $(t_A, t_B)$ and merge them into a new vocabulary token $t_{AB}$. Update the vocabulary: 
   $$\mathcal{V}_{t+1} = \mathcal{V}_t \cup \{t_{AB}\}$$
   Replace all occurrences of "$t_A \quad t_B$" in the corpus with "$t_{AB}$".

4. **Termination:**
   Repeat steps 2 and 3 until the target vocabulary size $|\mathcal{V}|$ is achieved or the maximum frequency of any pair drops below a defined threshold.

---

## 3. Mathematical Framework

Let a word string $W$ be defined as a sequence of symbols $s_1, s_2, \dots, s_n$. Let $C$ be the corpus dictionary mapping words to their absolute frequencies: $C = \{(W_k, f_k)\}_{k=1}^K$.

The frequency of a consecutive symbol pair $(s_i, s_j)$ within the corpus is calculated using an indicator function:

$$\text{Freq}(s_i, s_j) = \sum_{k=1}^{K} f_k \cdot \sum_{m=1}^{|W_k|-1} \mathbb{I}\left(W_{k, m} = s_i \land W_{k, m+1} = s_j\right)$$

At each optimization step, the BPE algorithm maximizes this frequency criteria:

$$(s_A, s_B) = \arg\max_{(s_i, s_j) \in \mathcal{V} \times \mathcal{V}} \text{Freq}(s_i, s_j)$$

### Byte-Level BPE (BBPE)
Standard BPE requires all base characters to be present in $\mathcal{V}_0$. For multilingual corpora or text with extensive emoji usage, the base character set can still scale to thousands of tokens, leaving less room for highly compressed subwords. 

Introduced in GPT-2 (Radford et al.), **Byte-Level BPE** resolves this by defining $\mathcal{V}_0$ strictly over raw bytes. Since there are exactly 256 possible byte values ($2^8$), the base vocabulary size is constrained to 256, guaranteeing that *any* string (regardless of language or encoding) can be fully decomposed without OOV tokens.

---

## 4. Alternative Paradigms: WordPiece vs. Unigram

Modern frameworks utilize different selection optimization criteria for subword induction:

### WordPiece (BERT)
Unlike BPE, which chooses the absolute most frequent pair, WordPiece calculates a scoring metric based on maximum likelihood. It prioritizes pairs that appear together far more frequently than expected based on individual token probabilities.

$$\text{Score}(s_i, s_j) = \frac{\text{Count}(s_i, s_j)}{\text{Count}(s_i) \times \text{Count}(s_j)}$$

* **Intuition:** This prevents highly frequent standalone characters (like spaces or common vowels) from dominating merges, ensuring highly specific semantic combinations are prioritized.

### Unigram Language Model (T5, SentencePiece)
Unigram flips the BPE approach. It starts with an **extremely large** initial vocabulary (e.g., all characters and highly frequent words/substrings) and iteratively **prunes** the least useful tokens. 

At each step, it optimizes a loss function based on the drop in corpus log-likelihood if a token $x$ were removed from $\mathcal{V}$:

$$\mathcal{L} = \sum_{k=1}^{K} \log P(W_k)$$

Where $P(W_k)$ is the maximized probability of generating word $W_k$ under the current vocabulary configuration using a hidden Markov parse.

---

## 5. Security Vulnerabilities & Engineering Edge Cases

### I. Token Smuggling and Injection
Because LLM safety alignments and system prompts rely heavily on explicit structural delimiters (e.g., `<|im_start|>`, `[INST]`), malicious users can attempt to bypass guardrails by feeding raw text strings that mirror these tokens. If tokenizers fail to treat user-input control tokens as raw text strings rather than system control flags, it can alter execution pathways.

### II. Grapheme Clusters & Byte-Splitting Glitches
In Byte-Level BPE, multi-byte characters (like complex non-Latin scripts or combined emojis) can be split down the middle across a subword boundary. 
* If a model truncates a generation exactly at a byte-boundary split, it outputs an invalid UTF-8 sequence, causing downstream application-level rendering failure crashes.

---

## 6. Complete Implementation: From-Scratch BPE

Below is a pure, production-ready Python implementation demonstrating the exact tokenization workflow: training a vocabulary from a small corpus, encoding raw text, and decoding back to strings.

```python
import re
from typing import Dict, List, Tuple

class BytePairEncoder:
    def __init__(self, target_vocab_size: int):
        self.target_vocab_size = target_vocab_size
        self.vocab: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[bytes, bytes], bytes] = {}
        
    def _get_stats(self, ids_corpus: List[List[bytes]]) -> Dict[Tuple[bytes, bytes], int]:
        """Counts frequencies of consecutive token pairs in the tokenized corpus."""
        counts: Dict[Tuple[bytes, bytes], int] = {}
        for row in ids_corpus:
            for pair in zip(row[:-1], row[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_corpus(self, ids_corpus: List[List[bytes]], pair: Tuple[bytes, bytes], idx_bytes: bytes) -> List[List[bytes]]:
        """Replaces all occurrences of the chosen target pair with the new merged byte string."""
        new_corpus = []
        for row in ids_corpus:
            new_row = []
            i = 0
            while i < len(row):
                if i < len(row) - 1 and (row[i], row[i+1]) == pair:
                    new_row.append(idx_bytes)
                    i += 2
                else:
                    new_row.append(row[i])
                    i += 1
            new_corpus.append(new_row)
        return new_corpus

    def fit(self, text_corpus: List[str]):
        """Trains the tokenizer vocabulary using BPE logic."""
        # 1. Initialize Base Vocabulary with raw byte configurations (0-255)
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # 2. Deconstruct corpus down into base individual byte representation arrays
        # Adding an explicit conceptual end-of-word space indicator mimicking real tokenizers
        processed_corpus = [w + " " for text in text_corpus for w in text.split()]
        ids_corpus: List[List[bytes]] = [[bytes([b]) for b in word.encode("utf-8")] for word in processed_corpus]
        
        # 3. Iterative calculation loop
        num_merges = self.target_vocab_size - 256
        for i in range(num_merges):
            stats = self._get_stats(ids_corpus)
            if not stats:
                break
                
            # Find the absolute highest frequency pair
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 1:
                break # Terminate if no repeated occurrences exist
                
            # Create the combined token
            merged_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged_token
            self.vocab[256 + i] = merged_token
            
            # Compress corpus sequences
            ids_corpus = self._merge_corpus(ids_corpus, best_pair, merged_token)
            
        print(f"Training Complete. Induced {len(self.merges)} structural merge patterns.")

    def tokenize(self, text: str) -> List[bytes]:
        """Encodes an incoming raw string sequence into subword byte components."""
        # Split text into structural fragments
        words = [w + " " for w in text.split()]
        tokens: List[bytes] = []
        
        for word in words:
            # Breakdown to base bytes initial state
            word_bytes = [bytes([b]) for b in word.encode("utf-8")]
            
            # Greedy replacement tracking iterative rules matching trained merges
            while len(word_bytes) > 1:
                pairs = list(zip(word_bytes[:-1], word_bytes[1:]))
                # Find which of the available sequence pairs matches our merge rules first
                eligible_merges = {pair: self.merges[pair] for pair in pairs if pair in self.merges}
                
                if not eligible_merges:
                    break # No remaining known merge groups to collapse
                    
                # Prioritize based on original generation order rule rank
                best_pair = min(eligible_merges.keys(), key=lambda p: list(self.merges.keys()).index(p))
                
                # Perform the merge transformation locally for this word string
                new_word_bytes = []
                idx = 0
                while idx < len(word_bytes):
                    if idx < len(word_bytes) - 1 and (word_bytes[idx], word_bytes[idx+1]) == best_pair:
                        new_word_bytes.append(eligible_merges[best_pair])
                        idx += 2
                    else:
                        new_word_bytes.append(word_bytes[idx])
                        idx += 1
                word_bytes = new_word_bytes
                
            tokens.extend(word_bytes)
        return tokens

    def decode(self, tokens: List[bytes]) -> str:
        """Assembles sequence chunks back cleanly into standard native text."""
        # Concatenate raw byte sequences and decode safely while catching edge formatting errors
        combined_bytes = b"".join(tokens)
        return combined_bytes.decode("utf-8", errors="replace")


# --- Execution Sandbox Verification ---
if __name__ == "__main__":
    corpus = [
        "hug monster hugger hugest hugging",
        "pug pumpkin pugger pugged",
        "unhugged tokenization subword deployment optimization architectures"
    ]
    
    # Initialize with a targeted experimental size limit
    encoder = BytePairEncoder(target_vocab_size=280)
    encoder.fit(corpus)
    
    # Run test scenarios
    sample_phrase = "hugging the pugger monster optimization"
    tokenized_chunks = encoder.tokenize(sample_phrase)
    decoded_string = encoder.decode(tokenized_chunks)
    
    print(f"\nSource Text:   '{sample_phrase}'")
    print(f"Tokenized Sequence Chunks: {tokenized_chunks}")
    print(f"Decoded Output Check:       '{decoded_string}'")
```
