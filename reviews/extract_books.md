# Extracting Books from Production Language Models

#### **What is this paper about?**
This paper investigates whether **production large language models (LLMs)** with built-in safeguards can be induced to generate near-verbatim copyrighted training data. It addresses the **research gap** that previous studies focused on open-weight or non-instruction-tuned models, leaving the vulnerability of production-level systems like GPT-4 and Claude largely unexplored.

#### **What are the key contributions of the work?**
*   **Two-Phase Extraction Procedure:** A novel methodology using an initial probe/jailbreak (Phase 1) followed by iterative continuation prompts (Phase 2) to elicit long-form text.
*   **Near-Verbatim Recall (nv-recall):** A new, **conservative measurement framework** for quantifying long-form extraction by identifying and filtering contiguous word blocks.
*   **Production Safeguard Analysis:** A systematic evaluation of extraction risks across four major frontier models: **Claude 3.7 Sonnet, GPT-4.1, Gemini 2.5 Pro, and Grok 3**.

#### **What are the key findings of the work?**
*   **Safeguard Disparity:** While Claude 3.7 Sonnet and GPT-4.1 required **Best-of-N (BoN) jailbreaks** to comply, Gemini 2.5 Pro and Grok 3 directly outputted copyrighted text without adversarial prompting.
*   **High-Fidelity Leakage:** Under specific configurations, **Claude 3.7 Sonnet** extracted nearly **96% of *Harry Potter and the Sorcerer’s Stone*** near-verbatim.
*   **Non-Deterministic Refusals:** GPT-4.1 often refused to continue after the first chapter, but these refusals were **non-deterministic**, allowing for further extraction through retries or chapter-by-chapter seeding.

#### **What are the main limitations or drawbacks of this work?**
*   **Monetary Cost:** Extracting a single book can be expensive, sometimes exceeding **$100 per run** for models with long-context charges.
*   **Restricted Scale:** The study tested only **14 specific books**, which may not represent general memorization patterns across entire training corpuses.
*   **API Instability:** Results are tied to specific API versions (mid-2025) and may be difficult to reproduce as providers update their **blackbox systems**.

#### **What are the key previous works that are evaluated in this paper and compared?**
*   **Carlini et al. (2021, 2023):** Established standard metrics for **short-form extraction** (approx. 50 tokens); this paper extends the scope to long-form documents.
*   **Cooper et al. (2025):** Demonstrated book extraction from **open-weight models** (e.g., Llama 3.1); this paper proves similar risks exist for **production models** with safety layers.
*   **Nasr et al. (2023, 2025):** Used jailbreaks to extract training data but did not focus on **contiguous, long-form copyrighted literature**.

#### **How to productionize this research result?**
*   **Enhanced Output Filtering:** Use findings to calibrate filters that detect not just exact matches, but **near-verbatim sequences** exceeding 100 words.
*   **Iterative Query Monitoring:** Implement rate-limiting or pattern-detection for users providing **sequential "continue" instructions** linked to known literature prefixes.
*   **Automated Red Teaming:** Integrate the **nv-recall metric** into pre-release safety pipelines to proactively identify highly memorized documents.
*   **Consistency Hardening:** Improve model alignment to ensure **refusals are deterministic**, preventing "retry-until-success" bypasses.

#### **What are some research gaps that still exist?**
*   **Non-Verbatim Copyright Analysis:** Investigating how models replicate **plot elements, character names, and themes** without verbatim copying.
*   **Advanced Adversarial Robustness:** Testing production models against more **sophisticated jailbreaks** beyond the simple Best-of-N approach used here.
*   **UI vs. API Vulnerabilities:** Initial evidence suggests **Chat UIs** may be more susceptible to extraction than APIs, requiring a dedicated comparative study.

#LLM #Copyright #AIAlignment #MachineLearning #DataLeakage
