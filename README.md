# 201 Project 4

This project is about Machine Translation from persian to english and to do so we use various algorithms and we assess their performance using various metrics.

## Dataset

The primary source of training data is the `shenasa/English-Persian-Parallel-Dataset`.
- **Custom Models (BiLSTM, GRU):** Trained on subsets ranging from 24,000 to 50,000 sentence pairs.
- **Pre-trained Models:** Evaluated on specific test splits.
- **NLLB:** Uses a separate evaluation set (`persian_english_french_3000_v3.csv`).

## Models & Approaches

We implemented and compared six different approaches, summarized below:

| Model | Notebook File | Type | Framework | Description |
|-------|--------------|------|-----------|-------------|
| **BiLSTM** | `Bilstm_translation.ipynb` | RNN | TensorFlow | Seq2Seq model with Bidirectional LSTM encoder, LSTM decoder, and Additive Attention. |
| **GRU** | `GRU.ipynb` | RNN | PyTorch | Custom Encoder-Decoder using GRU cells and Bahdanau Attention mechanism. Uses SentencePiece tokenization. |
| **mBART-50** | `Machine_translation_mbart50.ipynb` | Transformer | Hugging Face | Uses `facebook/mbart-large-50-many-to-many-mmt`. Multilingual translation model fine-tuned/verified on the dataset. |
| **NLLB** | `NLLB.ipynb` | Transformer | Hugging Face | Uses `facebook/nllb-200-distilled-600M`. Tested for direct translation capabilities. |
| **Llama 3.2** | `Llama_3_2.ipynb` | LLM | Hugging Face | Uses `Sheikhaei/llama-3.2-1b-en-fa-translator`. Generative translation using prompt engineering (`### English: ... ### Persian:`). |
| **SMT** | `SMT.ipynb` | Statistical | Python | Data cleaning, filtering, and formatting pipeline to prepare files for Statistical MT tools (e.g., Moses). |

## Methodological Details

### Preprocessing Pipeline
To ensure data quality, we applied several cleaning steps before training custom models:
1.  **Normalization:** Removal of URLs, emails, and English/Persian punctuation standardization.
2.  **Filtering:** 
    -   Removed sentences with excessive numbers or identical source/target text.
    -   Filtered by length (typically 4-50 tokens).
    -   Language verification checking (e.g., Persian text must contain valid Persian characters).
3.  **Tokenization:**
    -   **Deep Learning:** Custom BPE (SentencePiece) or Keras TextVectorization.
    -   **Transformers:** Subword tokenizers matching the specific pre-trained model.

### Evaluation Metrics
Models are evaluated using standard translation metrics:
-   **BLEU:** Measures n-gram overlap with reference translations.
-   **chrF++:** Character-level F-score, useful for morphologically rich languages like Persian.
-   **BERTScore:** Computes semantic similarity using BERT embeddings, capturing meaning rather than just exact matches.

  | Model / Method | Setup | BLEU | CHRF++ | BERTScore |
|---------------|-------|------|--------|-----------|
| SMT | Direct | 5.05 | 23.73 | 0.74 |
| BiLSTM | Direct | 0.24 | — | — |
| GRU + Attention | Direct | 8.12 | 25.36 | 0.70 |
| mBART50 | Pretrained NMT | 29.60 | — | — |
| NLLB | Direct (Persian → French) | 38.03 | 65.65 | 0.96 |
| NLLB | Pivot (English as intermediate) | 40.03 | 66.97 | 0.9635 |
| LLaMA 3.2 | LLM-based | 0.0044 | 7.6 | 0.63 |

### Results Analysis

The pivot-based approach using **NLLB** outperformed all other models across BLEU, CHRF++, and BERTScore.

Using **English as a pivot language** led to:
- Improved fluency in the target language
- Better semantic alignment between source and target sentences
- Reduced ambiguity in low-resource Persian–French translation

However, it is important to note that **NLLB was evaluated under different conditions** compared to other models in this project:

- NLLB is a **large-scale multilingual model** trained on extensive parallel corpora across many languages.
- The evaluation data and training setup for NLLB **differ from the dataset used for SMT and neural models**.
- Therefore, the results of NLLB should be interpreted as a **reference upper bound**, rather than a strictly fair comparison.

Additionally, while pivot translation improves translation quality, it introduces:
- Extra computational cost due to an additional translation step
- Increased inference time and system complexity

