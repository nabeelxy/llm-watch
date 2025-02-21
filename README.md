# Keeping up with LLMs

Since  the launch of ChatGPT in November 2022, LLMs (Large Language Models) and AI have been progressing so rapidly that it is often overwhelming to keeping up the developments in both the industry and the academia. This repo serves as an easy guide on the LLM/AI landscape for busy people.

## Primers
* LLMs: Beginners guide
* RAG: Beginner's guide
* LLM Agents: Beginner's guide


## Recent Important Publications

### Large Language Models
| Date | ID | Title | Venue | Materials | Tags | Short Summary | Summary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Feb 2025 | LLM-long-CoT-Structure | LLMs Can Easily Learn to Reason from Demonstrations; Structure, not content, is what matters! | arXiv | [Paper](https://arxiv.org/pdf/2502.07374) | long CoT, data-efficient SFT, LoRA, Qwen | | |
| Feb 2025 | A-MEM | A-MEM: Agentic Memory for LLM Agents | arXiv | [Paper](https://arxiv.org/abs/2502.12110v1) | agentic memory system, dynamic indexing, Zettelkasten | | |
| Feb 2025 | NSA-Deepseek | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention | arXiv | [Paper](https://arxiv.org/pdf/2502.11089) | sparse attention, long context modeling, instruction based reasoning | | |
| Feb 2025 | Scale-Test-Time-RL | Scaling Test-Time Compute Without Verification or RL is Suboptimal | arXiv | [Paper](https://arxiv.org/abs/2502.12118) | test time scaling, verification free methods, verification based methods | | |
| Feb 2025 | LLDM | Large Language Diffusion Models | arXiv | [Paper](https://arxiv.org/pdf/2502.09992) | LLaDA, SFT, ARM | | |
| Feb 2025 | s1 | s1: Simple test-time scaling | arXiv | [Paper](https://arxiv.org/pdf/2501.19393) | test-time scaling, reasoning, qwen2.5 | | |
| Feb 2025 | Self-Teach-LLM | LLMs Can Teach Themselves to Better Predict the Future | arXiv | [Paper](https://arxiv.org/pdf/2502.05253) | reasoning, direct preference optimization (dpo) | | |
| Feb 2025 | Distillation-LLM | Distillation Scaling Laws | arXiv | [Paper](https://arxiv.org/pdf/2502.08606) [Youtube](https://www.youtube.com/watch?v=TkX0RwG6cXk) | distillation | | |
| Feb 2025 | Comp-Programming | Competitive Programming with Large Reasoning Models | arXiv - OpenAI| [Paper](https://arxiv.org/pdf/2502.06807) | o1, o3, reasing models, ioi (international olympiad in informatics), reinforcement learning| | | 
| Jan 2025 | Zep | Zep: A Temporal Knowledge Graph Architecture for Agent Memory | arXiv | [Paper](https://arxiv.org/pdf/2501.13956) | agent memory, knowledge graph, MemGPT, DMR, RAG | | |
| Jan 2025 | Agent-R | Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training | arXiv | [Paper](https://arxiv.org/pdf/2501.11425) | agent, self-training | This paper by ByteDance proposes a framework for training LLMs to perform on-the-fly self-correction in interactive environments. Unlike traditional methods that reward or penalize actions solely based on correctness, their approach leverages Monte Carlo Tree Search (MCTS) to construct training samples that recover correct trajectories from erroneous ones. This helps to recover from mistakes and avoid unproductive loops, outperforming existing methods.| [Note](agent-r.md) |
| Jan 2025 | DeepSeek-R1 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | arXiv | [Report](https://arxiv.org/pdf/2501.12948) | llm, rl, cot, gpro, moe, mla | | [Note](deepseek-r1.md)|
| Dec 2024 | DeepSeek-V3 | DeepSeek-V3 Technical Report | arXiv | [Report](https://arxiv.org/pdf/2412.19437) | llm, rl, mlha, moe, mtp | | [Note](deepseek-v3.md)| 
| Sep 2024 | Adversarial-LLM | Assessing Adversarial Robustness of Large Language Models: An Empirical Study | arXiv | [Paper](https://arxiv.org/pdf/2405.02764) | llm, aml, robustness | | |

### RAG
| Date | ID | Title | Venue | Materials | Tags | Short Summary | Summary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Feb 2025 | KnowPath-RAG | KnowPath: Knowledge-enhanced Reasoning via LLM-generated Inference Paths over Knowledge Graphs | arXiv | [Paper](https://arxiv.org/abs/2502.12029) | RAG, knowledge graph, hullicination, reasoning paths | | |

### Benchmarks
| Date | ID | Title | Venue | Materials | Tags | Short Summary | Summary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jan 2025 | HALoGEN | HALOGEN : Fantastic LLM Hallucinations and Where to Find Them | arXiv | [Paper](https://arxiv.org/pdf/2501.08292) | benchmark, hallucination | Hallucination is a persistent issue in LLMs. HALoGEN benchmark shows SoTA models hallucinate up to 86% of the time. The benchmark consists of 10K prompts across nine domains. How does it detect haullucination? It breaks the response into atomic units and performs verifications against trusted knowledge sources. For example, imports in a Python code is verified against PyPI index. | [Note](halogen.md) |
| 2025 | HLE | Humanity's Last Exam |  | [Paper](https://static.scale.com/uploads/654197dc94d34f66c0f5184e/Publication%20Ready%20Humanity%27s%20Last%20Exam.pdf) | benchmark, LLM | Researcher from Scale AI, and Center for AI Safety came up with a benchmark consistings of 3000 question across dozens of subjects, including mathematics, humanities, and the natural science. DeepSeek-R1 is outperforming other state of the art models like O1 and Gemini 2 Flash Thinking. However, it should be noted that all models scored less than 10% accuracy on the benchmark raising the usefulness of the comparison. | [Note](hle.md) |

### Small Language Models
| Date | ID | Title | Venue | Materials | Tags | Short Summary | Summary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Feb 2025 | 1B-LLM | Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling | arXiv | [Paper]() | test-time scaling (TTS), math-500, aime24, process reward models (PRMs) | | |
| Dec 2024 | Phi-4 | Phi-4 Technical Report | arXiv | [Report](https://arxiv.org/pdf/2412.08905) | sml, distillation, synthetic data | | [Note](phi-4.md) |

### LLM/AI Engineering
| Date | ID | Title | Venue | Materials | Tags | Short Summary | Summary |
| --- | --- | --- | --- | --- | --- | --- | --- |
