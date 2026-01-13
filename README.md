# English-Persian Machine Translation Project

This project explores various Machine Translation (MT) architectures for translating English text into Persian. It compares the performance of custom RNN-based models against state-of-the-art pre-trained Transformers and Large Language Models.

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