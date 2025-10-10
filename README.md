# Keeping up with LLMs: Awesome LLM Guide

Since  the launch of ChatGPT in November 2022, LLMs (Large Language Models) and AI have been progressing so rapidly that it is often overwhelming to keeping up with the developments in both the industry and the academia. This repo serves as an easy guide on the LLM/AI landscape for busy people.

If you are interested in keeping up with LLM/Agent threat landscape, I encourage you to check my other repo on security for AI: [LLM Security](https://github.com/nabeelxy/ai-security-guide/tree/main/security_for_ai/llm_security) and [AI Red Teaming](https://github.com/nabeelxy/ai-security-guide/tree/main/security_for_ai/red_teaming), [Agent Security](https://github.com/nabeelxy/ai-security-guide/tree/main/security_for_ai/agents_security) and [RAG Security](https://github.com/nabeelxy/ai-security-guide/tree/main/security_for_ai/rag_security).

## Primers
* LLMs: Beginners guide
* RAG: Beginner's guide
* LLM Agents: Beginner's guide
* AI Design Patterns: [Guide](https://github.com/nabeelxy/llm-watch/blob/main/ai_design_patterns.md)
* Design Patterns against Prompt Injection: [Guide](https://github.com/nabeelxy/llm-watch/blob/main/design-pattern-prompt-injection.md)


## Recent Important Publications

### Large Language Models
| Date | Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| Oct 2025 | DEEPSEARCH: OVERCOME THE BOTTLENECK OF REINFORCEMENT LEARNING WITH VERIFIABLE REWARDS VIA MONTE CARLO TREE SEARCH | arXiv, Hugginface, Stanford | [Paper](https://arxiv.org/pdf/2509.25454) | RLVR, train-efficiency | | |
| Sep 2025 | Soft Tokens, Hard Truths | UA, Meta, NYU | [Paper](https://www.arxiv.org/pdf/2509.19170) [NotebookLM](https://notebooklm.google.com/notebook/727bc6ea-465d-4dcc-8048-b9f85b097631?authuser=1) [Podcast](https://notebooklm.google.com/notebook/727bc6ea-465d-4dcc-8048-b9f85b097631?artifactId=293a1f3b-1e0a-4af5-8c9d-db491c8612e2)| CoT, continuous CoT, RL | | [Note](https://github.com/nabeelxy/llm-watch/blob/main/reviews/soft_tokens_sep_2025.md) |
| Sep 2025 | MIXTURE OF THOUGHTS: LEARNING TO AGGREGATE WHAT EXPERTS THINK, NOT JUST WHAT THEY SAY | USC, arXiv | [Paper](https://arxiv.org/pdf/2509.21164) | routing, mixture models | | |
| Aug 2025 | OptimalThinkingBench: Evaluating Over and Underthinking in LLMs | Meta | [Paper](https://www.arxiv.org/pdf/2508.13141) | thinking, benchmark | | |
| Aug 2025 |  We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs | Usenix Security 2025 | [Paper](https://arxiv.org/pdf/2406.10279) [NotebookLM](https://notebooklm.google.com/notebook/6495317c-4a13-4909-815f-bcc4bd96f167?authuser=1) [Podcast](https://notebooklm.google.com/notebook/6495317c-4a13-4909-815f-bcc4bd96f167/audio)| llm, hallucination, packages, code generation | | |
| Aug 2025 | Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens | arXiv, ASU | [Paper](https://arxiv.org/pdf/2508.01191) | CoT, reasoning | | |
| Jul 2025 | Learning without training: The implicit dynamics of in-context learning | Google | [Paper](https://arxiv.org/pdf/2507.16003) [NotebookLM](https://notebooklm.google.com/notebook/f0f72407-5625-41f9-b2bc-7e0e37116eac?authuser=1)| in-context learning, implicit weight update | | |
| Jul 2025 | Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities | Google | [Paper](https://arxiv.org/pdf/2507.06261) [NotebookLM](https://notebooklm.google.com/notebook/16f5464a-bffb-4d3e-ae6d-4cedef0ca8a1?authuser=1) | Gemini 2.5, multi-modal, reasoning, agentic, context | | |
| Jul 2025 | Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs | ICML 2025 | [Paper](https://openreview.net/pdf?id=aOIJ2gVRWW) [NotebookLM](https://notebooklm.google.com/notebook/756d4c76-33b5-48bd-a696-d07ed805a9b9)| GPT-4o, insecure code, misalignment | | | 
| Jun 2025 | Chain-of-Thought Reasoning In The Wild Is Not Always Faithful | arXiv | [Paper](https://arxiv.org/pdf/2503.08679) [NotebookLM](https://notebooklm.google.com/notebook/cb6536b4-5434-480e-8d7e-425262c97324?authuser=1)| cot | | |
| Jun 2025 | Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities | Google DeepMind | [TR](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf) [NotebookLM](https://notebooklm.google.com/notebook/a84c8caf-8292-4ea0-89e4-d692ed53103f?authuser=1) [Podcast](https://notebooklm.google.com/notebook/a84c8caf-8292-4ea0-89e4-d692ed53103f/audio)| reasoining, long context, agents | | |
| Jun 2025 | Reinforcement Pre-Training | arXiv, Microsoft | [Paper](https://arxiv.org/pdf/2506.08007)  [NotebookLM](https://notebooklm.google.com/notebook/e960ffcd-1c6b-49e0-a2ff-fd25101ef5d4?authuser=1) [Podcast](https://notebooklm.google.com/notebook/e960ffcd-1c6b-49e0-a2ff-fd25101ef5d4/audio)| scaling paradigms, next token prediction, reinforcement learning | | |
| Jun 2025 | [Withdrawn] CoT is Not True Reasoning, It Is Just a Tight Constraint to Imitate: A Theory Perspective | arXiv | [Paper](https://www.arxiv.org/pdf/2506.02878) | CoT, reasoning, immitation learning, structructural constraints | | |
| Jun 2025 | KNOWLEDGE or REASONING ? A Close Look at How LLMs Think Across Domains | arXiv | [Paper](https://arxiv.org/pdf/2506.02126) [NotebookLM](https://notebooklm.google.com/notebook/a34e00f4-1d5c-4f24-a526-96bad91466cd?authuser=1)| Knoweledge Index, Infromation Gain, SFT, RL, Distillation | | |
| Jun 2025 | The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity | Apple | [Paper](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) [NotebookLM](https://notebooklm.google.com/notebook/8deb9d94-5638-43a8-8a3b-e55905d79422) [Podcast](https://notebooklm.google.com/notebook/8deb9d94-5638-43a8-8a3b-e55905d79422/audio)| LRM, accuracy collapse, problem complexity, reasoing | |[Note](https://github.com/nabeelxy/llm-watch/blob/main/reviews/illusion-of-thinking.md) |
| May 2025 | Can Large Reasoning Models Self-Train? | arXiv, CMU | [Paper](https://www.arxiv.org/pdf/2505.21444) [NotebookLM](https://notebooklm.google.com/notebook/5ded4714-7587-4692-afc3-939a52fbc9a0?authuser=1)| RL, self-training | | |
| May 2025 | ALPHAONE: Reasoning Models Thinking Slow and Fast at Test Time | arXiv, UIUC, Berkeley | [Paper](https://www.arxiv.org/pdf/2505.24863) [NotebookLM](https://notebooklm.google.com/notebook/76ff3ac6-0671-435c-a577-cb43d06b1b70?authuser=1)| large reasoning models, slow-to-fast reasoning | | |
| May 2025 | From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning | arXiv, Stanford, Meta | [Paper](https://arxiv.org/pdf/2505.17117) [NotebookLM](https://notebooklm.google.com/notebook/73fb9db1-7ab8-4dca-bff6-da33547144af?authuser=1)| thoughts, semantic compression compression vs. semnatic fidelity | | |
| May 2025 | How much do language models memorize? | Meta | [Paper](https://arxiv.org/pdf/2505.24832) | memorization, generalization, gpt | | |
| May 2025 | Learning to Reason without External Rewards | arXiv | [Paper](https://www.arxiv.org/pdf/2505.19590) | RLVR, RLIR, GRPO | | |
| May 2025 | KumoRFM: A Foundation Model for In-Context Learning on Relational Data | Kumo | [Paper](https://kumo.ai/research/kumo_relational_foundation_model.pdf) | relational, foundation model | | |
| May 2025 |  AlphaEvolve: A coding agent for scientific and algorithmic discovery | DeepMind | [Paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf) [Blog](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)| evolution, auto discovery, research | | |
| Mar 2025 | Reasoning to Learn from Latent Thoughts | arXiv | [Paper](https://arxiv.org/pdf/2503.18866) | synthetic data generation, latent thoughts | | |
| Mar 2025 | SuperBPE: Space Travel for Language Models | arXiv | [Paper](https://arxiv.org/pdf/2503.13423) | tokenization, multi-word tokens | | |
| Mar 2025 | Chain-of-Tools: Utilizing Massive Unseen Tools in the CoT Reasoning of Frozen Language Models | arXiv | [Paper](https://www.arxiv.org/pdf/2503.16779) | CoT, tools | | |
| Mar 2025 | Gemini Embedding: Generalizable Embeddings from Gemini | arXiv, Google | [Paper](https://arxiv.org/pdf/2503.07891v1) | MMTEB benchmark, Gemini, embedding | | |
| Mar 2025 | Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning | arXiv | [Paper](https://arxiv.org/pdf/2503.09516) | RL, RAG, Reasoning, DeepSeek, tool use | | |
| Feb 2025 | TabICL: A Tabular Foundation Model for In-Context Learning on Large Data | ICML 2025 | [Paper](https://arxiv.org/pdf/2502.05564) [GitHub](https://github.com/soda-inria/tabicl)| Tabular foundation model, talent benchmark | | |
| Feb 2025 | Large Language Diffusion Models | arXiv | [Paper](https://arxiv.org/pdf/2502.09992) | diffusion models, auto-regressive models | | |
| Feb 2025 | LLMs Can Easily Learn to Reason from Demonstrations; Structure, not content, is what matters! | arXiv | [Paper](https://arxiv.org/pdf/2502.07374) | long CoT, data-efficient SFT, LoRA, Qwen | | |
| Feb 2025 | A-MEM: Agentic Memory for LLM Agents | arXiv | [Paper](https://arxiv.org/abs/2502.12110v1) | agentic memory system, dynamic indexing, Zettelkasten | | |
| Feb 2025 | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention | arXiv | [Paper](https://arxiv.org/pdf/2502.11089) | sparse attention, long context modeling, instruction based reasoning | | |
| Feb 2025 | Scaling Test-Time Compute Without Verification or RL is Suboptimal | arXiv | [Paper](https://arxiv.org/abs/2502.12118) | test time scaling, verification free methods, verification based methods | | |
| Feb 2025 | Large Language Diffusion Models | arXiv | [Paper](https://arxiv.org/pdf/2502.09992) | LLaDA, SFT, ARM | | |
| Feb 2025 | s1: Simple test-time scaling | arXiv | [Paper](https://arxiv.org/pdf/2501.19393) | test-time scaling, reasoning, qwen2.5 | | |
| Feb 2025 | LLMs Can Teach Themselves to Better Predict the Future | arXiv | [Paper](https://arxiv.org/pdf/2502.05253) | reasoning, direct preference optimization (dpo) | | |
| Feb 2025 | Distillation Scaling Laws | arXiv | [Paper](https://arxiv.org/pdf/2502.08606) [Youtube](https://www.youtube.com/watch?v=TkX0RwG6cXk) | distillation | | |
| Feb 2025 | Competitive Programming with Large Reasoning Models | arXiv - OpenAI| [Paper](https://arxiv.org/pdf/2502.06807) | o1, o3, reasing models, ioi (international olympiad in informatics), reinforcement learning| | | 
| Feb 2025 | CoAT: Chain-of-Associated-Thoughts Framework for Enhancing Large Language Models Reasoning | arXiv | [Paper](https://arxiv.org/pdf/2502.02390v1) | monte carlo tree search, associative memory, system 2 thinking, reasoning | | |
| Jan 2025 | KIMI K1.5: SCALING REINFORCEMENT LEARNING WITH LLMS | Kimi | [Paper](https://arxiv.org/pdf/2501.12599) | long context scaling, reasoning, reward models | | |
| Jan 2025 | Zep: A Temporal Knowledge Graph Architecture for Agent Memory | arXiv | [Paper](https://arxiv.org/pdf/2501.13956) | agent memory, knowledge graph, MemGPT, DMR, RAG | | |
| Jan 2025 | Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training | arXiv | [Paper](https://arxiv.org/pdf/2501.11425) | agent, self-training | | [Note](reviews/agent-r.md) |
| Jan 2025 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | arXiv | [Report](https://arxiv.org/pdf/2501.12948) | llm, rl, cot, gpro, moe, mla | | [Note](reviews/deepseek-r1.md)|
| Dec 2024 | DeepSeek-V3 Technical Report | arXiv | [Report](https://arxiv.org/pdf/2412.19437) | llm, rl, mlha, moe, mtp | | [Note](reviews/deepseek-v3.md)| 
| Sep 2024 | Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries | Google | [Paper](https://arxiv.org/pdf/2409.12640) | long context, evaluation | | |
| Sep 2024 | RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback | ICML 2024 | [Paper](https://arxiv.org/pdf/2309.00267v3) | RFAIF, RLHF | ||
| Sep 2024 | Assessing Adversarial Robustness of Large Language Models: An Empirical Study | arXiv | [Paper](https://arxiv.org/pdf/2405.02764) | llm, aml, robustness | | |
| Jul 2024 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Stanford | [Paper](https://arxiv.org/pdf/2305.18290) | DPO, reward model, optimization | | |
| Jun 2024 | TinyLlama: An Open-Source Small Language Model | SUTD, arXiv | [Paper](https://arxiv.org/pdf/2401.02385) | 1.1B, FlashAttention, Lit-GPT | [GitHub](https://github.com/jzhang38/TinyLlama) | |
| May 2024 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | CMU, Princeton | [Paper](https://arxiv.org/pdf/2312.00752) | Mamba | | |
| Jul 2023 | Llama 2: Open Foundation and Fine-Tuned ChatModels | Meta | [Paper](https://arxiv.org/pdf/2307.09288) | open source FM, Llama2-Chat | | |
| Jan 2023 | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | Google Research | [Paper](https://arxiv.org/pdf/2201.11903) | CoT | | |
| Dec 2022 | Constitutional AI: Harmlessness from AI Feedback | Anthropic | [Paper](https://arxiv.org/pdf/2212.08073) | RLAIF, no human labels | | |
| Jun 2022 | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | arXiv | [Paper](https://arxiv.org/pdf/2101.03961) | transformers, moe | | |

### Agents
| Date |Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| Oct 2025 | Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models | arXiv, Stanford | [Paper](https://www.arxiv.org/pdf/2510.04618) [NotebookLM](https://notebooklm.google.com/notebook/732391f2-5938-4e61-b92a-d71c4c283265?authuser=1) | agentic LLM, context collapse, brevity bias, generator-reflector-curator | | |
| Sep 2025 | LIMI: Less is More for Agency | arXiv | [Paper](https://arxiv.org/pdf/2509.17567) | agency, anti-scaling-law | | |
| Aug 2025 | Memento: Fine-tuning LLM Agents without Fine-tuning LLMs | arXiv, AI Center | [Paper](https://www.arxiv.org/pdf/2508.16153) [NotebookLM](https://notebooklm.google.com/notebook/7a5267cd-7c6e-4d58-be34-347353b8e412?authuser=1) | fine-tuning, GAIA | [Github](https://github.com/Agent-on-the-Fly/Memento) | |
| Jul 2025 | Deep Researcher with Test-Time Diffusion | Google Cloud | [Paper](https://www.arxiv.org/pdf/2507.16075) | self-evolutionary algorithm, diffusion | | |
| Jul 2025 |  AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML | ICML 2025 | [Paper](https://openreview.net/pdf?id=p1UBWkOvZm) [NotebookLM](https://notebooklm.google.com/notebook/f5513c77-8e11-4ff2-a685-b2847500757e?pli=1&authuser=1&original_referer=https:%2F%2Fgds.google.com%23) | LLM, AutoLM | | |
| Jul 2025 | Establishing Best Practices for Building Rigorous Agentic Benchmarks | arXiv | [Paper](https://www.arxiv.org/pdf/2507.02825) | agentic benchmark, CVE-bench | | |
| Jun 2025 | DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents | arXiv | [Paper](https://arxiv.org/pdf/2506.11763) | deepresearch, benchmark | | |
| Jun 2025 | Small Language Models are the Future of Agentic AI | Nvidia | [Paper](https://arxiv.org/pdf/2506.02153) | SLM | [Github](research.nvidia.com/labs/lpr/slm-agents) | | 
| Jun 2025 | DEEP RESEARCH AGENTS: A SYSTEMATIC EXAMINATION AND ROADMAP | arXiv | [Paper](https://arxiv.org/pdf/2506.18096) [NotebookLM](https://notebooklm.google.com/notebook/2985c2ac-1c95-462c-a69a-54d74b317167) [Podcast](https://notebooklm.google.com/notebook/2985c2ac-1c95-462c-a69a-54d74b317167/audio)| deep research, mcp | | |
| May 2025 | Multi-Agent Collaboration via Evolving Orchestration | arXiv | [Paper](https://arxiv.org/pdf/2505.19591) | puppet master, RL, orchestrator, multi-agent system | | [Note](https://github.com/nabeelxy/llm-watch/blob/main/reviews/puppet-master.md)|
| May 2025 | Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents | arXiv | [Paper](https://arxiv.org/pdf/2505.09970) | Planning ahead, ReAct, planning, multi-agent | | [Note](https://github.com/nabeelxy/llm-watch/blob/main/reviews/preact.md)|
| Apr 2025 | Why Do Multi-Agent LLM Systems Fail? | arXiv | [Paper](https://arxiv.org/pdf/2503.13657) [NotebookLM](https://notebooklm.google.com/notebook/b5ba12e5-07e4-4629-a425-1b30f5c81e1c?authuser=1) | multi-agent, failures, systems | | |
| Apr 2025 | A Survey of AI Agent Protocols | arXiv | [Paper](https://arxiv.org/pdf/2504.16736) | protcols, survey | | |
| Feb 2025 | Understanding the planning of LLM agents: A survey | arXiv | [Paper](https://arxiv.org/pdf/2402.02716) | planning, survey | | |
| Jan 2025 | QUERYING DATABASES WITH FUNCTION CALLING | arXiv | [Paper](https://arxiv.org/pdf/2502.00032) | Agent, text-to-sql, function calling, json | | |
| Oct 2024 | Agent-as-a-Judge: Evaluate Agents with Agents | arXiv - Meta | [Paper](https://arxiv.org/pdf/2410.10934) | agent, judge | | |
| Mar 2023 | REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS | ICLR 2023 | [Paper](https://arxiv.org/pdf/2210.03629) | Agent, ReAct pattern | | |

### RAG
| Date | Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| May 2025 | REARANK: Reasoning Re-ranking Agent via Reinforcement Learning | arXiv | [Paper](https://arxiv.org/pdf/2505.20046v1) [NotebookLM](https://notebooklm.google.com/notebook/28753282-4d0e-4be4-9a24-3ef3322afe51)| reranking, qwen2.5-7b, BRIGHT benchmarks | | |
| Feb 2025 | KnowPath: Knowledge-enhanced Reasoning via LLM-generated Inference Paths over Knowledge Graphs | arXiv | [Paper](https://arxiv.org/abs/2502.12029) | RAG, knowledge graph, hullicination, reasoning paths | | |

### Benchmarks
| Date | Title | Venue | Materials | Tags | Code | Summary |
| --- |  --- | --- | --- | --- | --- | --- |
| May 2025 | General Scales Unlock AI Evaluation with Explanatory and Predictive Power | arXiv, Microsoft | [Paper](https://arxiv.org/pdf/2503.06378) | evals, benchmark | | |
| Apr 2025 | HealthBench: Evaluating Large Language Models Towards Improved Human Health | OpenAI | [Paper](https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf) [Blog](https://openai.com/index/healthbench/) | 5K conversaions, rubrics | [Github](https://kinds-of-intelligence-cfi.github.io/ADELE) | |
| Apr 2025 | The LeaderBoard Illusion | arXiv | [Paper](https://arxiv.org/pdf/2504.20879) | Chatbot Arean, Eval | | |
| Jan 2025 | HALOGEN : Fantastic LLM Hallucinations and Where to Find Them | arXiv | [Paper](https://arxiv.org/pdf/2501.08292) | benchmark, hallucination |  | [Note](reviews/halogen.md) |
| 2025 |  Humanity's Last Exam |  | [Paper](https://static.scale.com/uploads/654197dc94d34f66c0f5184e/Publication%20Ready%20Humanity%27s%20Last%20Exam.pdf) | benchmark, LLM |  | [Note](reviews/hle.md) |
| Mar 2024 | InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks | ICML 2024 | [Paper](https://arxiv.org/pdf/2401.05507) | LLM agents, analysis tasks, format prompting, close-ended tasks | [GitHub](https://github.com/InfiAgent/InfiAgent) | |

### Small Language Models
| Date | Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| Aug 2025 | Jet-Nemotron: Efficient Language Model with Post
Neural Architecture Search | Nvidia | [Paper](https://arxiv.org/abs/2508.15884) [NotebookLM](https://notebooklm.google.com/notebook/cc6ddfc7-5d54-42a3-9f38-ada185c14e2f?authuser=1) | NAS, Distillation, linear attention | [Github](https://github.com/NVlabs/Jet-Nemotron)| |
| Jul 2025 | rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking | ICML 2025 | [Paper](https://openreview.net/pdf?id=5zwF1GizFa) [NotebookLM](https://notebooklm.google.com/notebook/ea54dfcf-7cdf-4c94-9ce1-b8117746300c) | deep thinking through MCTS, PPM, AIME | [Github](https://github.com/microsoft/rStar) | |
| May 2025 | Llama-Nemotron: Efficient Reasoning Models | arXiv, Nvidia | [Paper](https://arxiv.org/pdf/2505.00949) | efficient reasoning, llama | | |
| Apr 2025 | Phi-4-Mini-Reasoning: Exploring the Limits of Small Reasoning Language Models in Math | arXiv | [Paper](https://arxiv.org/pdf/2504.21233) | reason, slm, distill, CoT | | |
| Feb 2025 | Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling | arXiv | [Paper]() | test-time scaling (TTS), math-500, aime24, process reward models (PRMs) | | |
| Dec 2024 | Phi-4 Technical Report | arXiv | [Report](https://arxiv.org/pdf/2412.08905) | sml, distillation, synthetic data | | [Note](reviews/phi-4.md) |

### LLM/AI Applciations/Engineering
| Date | Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| Jul 2025 | A Survey of Context Engineering for Large Language Models | arXiv | [Paper](https://arxiv.org/pdf/2507.13334) | survey, context engineering | | |
| Jun 2025 | A Practical Guide for Evaluating LLMs and LLM-Reliant Systems | Google | [Paper](https://arxiv.org/pdf/2506.13023) | evaluation framework, LLM-reliant systems | | |
| Mar 2025 | REALM: A Dataset of Real-World LLM Use Cases | arXiv | [Paper](https://arxiv.org/pdf/2503.18792v1) | LLM applications, use cases, news | | |

### Graph Foundation Models
| Date |Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| Jul 2025  | AutoGFM: Automated Graph Foundation Model with Adaptive Architecture Customization | ICML 2025 | [Paper](https://openreview.net/pdf?id=fCPB0qRJT2) | adaptive architecture | | |
|Mar 2025 | Graph Foundation Models: Concepts, Opportunities and Challenges | arXiv | [Paper](https://arxiv.org/pdf/2310.11829) | gnn, llm, gfm | | |

#### Useful Links
* Awesome Graph LLM references [repo](https://github.com/XiaoxinHe/Awesome-Graph-LLM)

### Influential Papers in NLP (Before ChatGPT in Nov 2022)
| Date | Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| 2022 | Training language models to follow instructions with human feedback | OpenAI | [Paper](https://arxiv.org/pdf/2203.02155) [Notebook](https://notebooklm.google.com/notebook/6ea736d5-b5ad-4ea2-b170-f15c82b52115?authuser=3)| InstructGPT, RLHF, LLM | | |
| 2022 | PaLM: Scaling Language Modeling with Pathways | Google Research | [Paper](https://arxiv.org/pdf/2204.02311) [NotebookLM](https://notebooklm.google.com/notebook/7c95a433-9319-4d63-b783-7b819272be0b?authuser=1) [Podcast](https://notebooklm.google.com/notebook/7c95a433-9319-4d63-b783-7b819272be0b/audio)| PaLM, LLM | | |
| 2022 | Training Compute-Optimal Large Language Models | Google DeepMind | [Paper](https://arxiv.org/pdf/2203.15556) [NotebookLM](https://notebooklm.google.com/notebook/2b0277f9-aa10-4206-8004-41d0d35730f6) [Podcast](https://notebooklm.google.com/notebook/2b0277f9-aa10-4206-8004-41d0d35730f6/audio)| chinchilla, LLM | | |
| 2021 | LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS | Microsoft | [Paper](https://arxiv.org/pdf/2106.09685) [NotebookLM](https://notebooklm.google.com/notebook/ee05a51e-7bb9-4e2a-bee6-f90353a3d7a9) [Podcast](https://notebooklm.google.com/notebook/ee05a51e-7bb9-4e2a-bee6-f90353a3d7a9/audio)| LoRA, fine-tuning | | |
| 2020 | Language Models are Few-Shot Learners | OpenAI | [Paper](https://arxiv.org/pdf/2005.14165) [NotebookLM](https://notebooklm.google.com/notebook/d0ef99d0-400e-4b46-80f5-4c363f6b4cbe) [Podcast](https://notebooklm.google.com/notebook/d0ef99d0-400e-4b46-80f5-4c363f6b4cbe/audio)| GPT-3 | | |
| 2020 | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | Google | [Paper](https://arxiv.org/pdf/1910.10683) [NotebookLM](https://notebooklm.google.com/notebook/637df39b-90b3-4244-ade1-5cfb265d184f?authuser=1) [Podcast](https://notebooklm.google.com/notebook/637df39b-90b3-4244-ade1-5cfb265d184f/audio)| T5 | [GitHub](https://github.com/google-research/text-to-text-transfer-transformer) | |
| 2020 | ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS | Google, Stanford, ICLR 2020 | [Paper](https://arxiv.org/pdf/2003.10555) [NotebookLM](https://notebooklm.google.com/notebook/eccd689f-99fd-4cdd-bb97-f9e16a4e56a1?authuser=1) [Podcast](https://notebooklm.google.com/notebook/eccd689f-99fd-4cdd-bb97-f9e16a4e56a1/audio)| replaced token detection, corrupted tokens, discriminative modeling | | |
| 2019 | DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter | HuggingFace | [Paper](https://arxiv.org/pdf/1910.01108) [NotebookLM](https://notebooklm.google.com/notebook/8f18a615-2862-46fa-9587-644a0d328aa7?authuser=1) [Podcast](https://notebooklm.google.com/notebook/8f18a615-2862-46fa-9587-644a0d328aa7/audio)| DistilBERT, distillation | | |
| 2018 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Google | [Paper](https://arxiv.org/pdf/1810.04805) [NotebookLM](https://notebooklm.google.com/notebook/6d726fb6-5c4a-4138-95e3-6f9223ab92bb) [Podcast](https://notebooklm.google.com/notebook/6d726fb6-5c4a-4138-95e3-6f9223ab92bb/audio)| encoder, embeddings | | |
|2017 | Attention is all you need | Google | [Paper](https://arxiv.org/pdf/1706.03762) [NotebookLM](https://notebooklm.google.com/notebook/2fff097e-444c-4d0a-892f-fa44ce7f3c8a) [Podcast](https://notebooklm.google.com/notebook/98ee4c7f-fbc6-4920-83dc-e67d586fd1ac/audio)| transformers, nlp | | |
| 2014 | Sequence to Sequence Learning with Neural Networks | Google | [Paper](https://arxiv.org/pdf/1409.3215) [NotebookLM](https://notebooklm.google.com/notebook/98ee4c7f-fbc6-4920-83dc-e67d586fd1ac) [Podcast](https://notebooklm.google.com/notebook/98ee4c7f-fbc6-4920-83dc-e67d586fd1ac/audio)| lstm | | |
| 2014 | GloVe: Global Vectors for Word Representation | Stanford | [Paper](https://nlp.stanford.edu/pubs/glove.pdf) [NotebookLM](https://notebooklm.google.com/notebook/38e03431-58e4-424c-96a0-9335ed2375b3) [Podcast](https://notebooklm.google.com/notebook/38e03431-58e4-424c-96a0-9335ed2375b3/audio) | GloVe | | |
| 2013 | Distributed Representations of Words and Phrases and their Compositionality | Google | [Paper](https://arxiv.org/pdf/1310.4546) [NotebookLM](https://notebooklm.google.com/notebook/677909a2-0260-40cf-ac8b-5ef96055fdee?authuser=1) [Podcast](https://notebooklm.google.com/notebook/677909a2-0260-40cf-ac8b-5ef96055fdee/audio)| word2vec | | |
| 1997 | Long Short-Term Memory | Neural Computation, MIT | [Paper](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf) [NotebookLM](https://notebooklm.google.com/notebook/659a2cee-c528-4849-b208-b75fadd72b4f?authuser=4) [Podcast](https://notebooklm.google.com/notebook/659a2cee-c528-4849-b208-b75fadd72b4f/audio)|  LSTM | | |

### Blog posts
* [2025-10-09] [Blog](https://www.anthropic.com/research/small-samples-poison) A small number of samples can poison LLMs of any size by Anthropic. #poisoning #aml #gibberish-text #fixed-poisoned-dataset ([Paper](https://arxiv.org/pdf/2510.07192))
* [2025-10-03] [Blog](https://www.anthropic.com/research/building-ai-cyber-defenders) Building AI for Cyber Defenders by Anthropic. #CTF #vulnerability-discovery #Cybench #CyberGym #patching #RedTeaming
* [2025-09-29] [Blog](http://thinkingmachines.ai/blog/lora/) LoRA with Regret by Thinking Machines. #post-training #PEFT #LoRA
* [2025-09-22] [Blog](https://huggingface.co/blog/gaia2) Gaia2 and ARE: Empowering the Community to Evaluate Agents by HuggingFace and Meta. #agent-eval #dataset #benchmark
* [2025-09-11] [Blog](https://www.anthropic.com/engineering/writing-tools-for-agents) Writing Effective Tools for Agents by Anthropic. #agents #tools #playbook
* [2025-09-10] [Blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) Defeating Nondeterminism in LLM Infernece by Thinking Machines. #floating-point #vLLM #FlexAttention
* [2025-07-24] [Case Study](https://www.anthropic.com/news/how-anthropic-teams-use-claude-code) How Anthropic teams use Claude code by Anthropic.
* [2025-06-26] [Blog](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html) How to fix your context by Drew Breuning #context-engineering #tips
* [2025-06-22] [Blog](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html) How long contexts fail by Drew Breuning #context-engineering #fail
* [2025-06-15] [Report](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) A practical guide to building agents by OpenAI. #agents #engineering
* [2025-06-12] [Blog](https://blog.langchain.dev/the-hidden-metric-that-determines-ai-product-success/) The Hidden Metric That Determines AI Product Success by LangChain. #product #engineering #confidence
* [2025-06-10] [Blog](https://blog.samaltman.com/the-gentle-singularity) The Gentle Singularity by Sam Altman. #superintelligence #agents #robots #productivity #datacenters #alignment #affordability
* [2025-06-10] [Blog](https://www.perplexity.ai/hub/blog/accelerating-sonar-through-speculation) Accelerating Sonar Through Speculation, by Perplexity AI. #speculative-decoding 
* [2025-06-10] [PR](https://mistral.ai/news/magistral) Announcing Magistral - the first reasoning model by Mistral AI. (Magistral Small is available in [Huggingface](https://huggingface.co/mistralai/Magistral-Small-2506)) #reasoning, #open-weight #moe 
* [2025-05-12] [Blog](https://www.microsoft.com/en-us/research/blog/predicting-and-explaining-ai-model-performance-a-new-approach-to-evaluation) Predicting and explaining AI model performance: A new approach to evaluation by Microsoft. #evals #benchmark
* [2025-05-01] [Blog](https://lilianweng.github.io/posts/2025-05-01-thinking/) Why We Think by Lilian Wang. #test-time-compute #cot #thinking 
* [2025-04-15] [Report](https://knightcolumbia.org/content/ai-as-normal-technology) AI as a normal technology by Arvind Narayanan and Sayash Kapoor.
* [2025-04-13] [Blog](https://www.newsletter.swirlai.com/p/mcp-vs-a2a-friends-or-foes) MCP vs. A2A: Friends or Foes?, by Aurimas Griciunas (SwirlAI Newsletter)
* [2025-04-11] [Report](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf) AI in the enterprise by Open AI.
* [2025-04-03] [Blog](https://eugeneyan.com/writing/recsys-llm/) Improving the recommendation systems and search in the age of LLMs, by Eugene Yan.
* [2025-03-20] [Blog](https://www.anthropic.com/engineering/claude-think-tool) The "think" tool: Enabling Claude to stop and think in complex tool use situations, by Anthropic. #thinking #toolinvovation #agents
* [2025-03-12] [Blog](https://blog.google/technology/developers/gemma-3/) Google's Gemma 3 release note.
* [2025-03-11] [Documentation](https://platform.openai.com/docs/guides/agents-sdk) OpenAI's Agent SDK guide.
* [2025-03-10] [Article](https://www.latent.space/p/why-mcp-won) Model Context Protocol, Summary of the AI Engineer summit talk by Anthropic.
* [2025-03-01] [Youtube](https://www.youtube.com/watch?v=kQmXtrmQ5Zg) MCP workshop by Mahesh Murag of Anthropic at AI Engineer summit.
* [2025-02-26] [Article](https://thomwolf.io/blog/scientific-ai.html) The Einstein AI model by Thomas Wolf, CSO, Huggingface. Thomas discusses the AI's current path of producing "obidient students" as opposed to "revolutionaries".
* [2025-02-06] [Article](https://cloudsecurityalliance.org/blog/2025/02/06/agentic-ai-threat-modeling-framework-maestro) Agentic AI threat modeling framework - MAESTRO.
* [2025-02-01] [White Paper](https://www.kaggle.com/whitepaper-agent-companion) Agents Companion by Google.
* [2024-11-25] [Article](https://www.anthropic.com/news/model-context-protocol) MCP (Model Context Protocol) by Anthropic.


### Notable Repos
* [12-factor agents](https://github.com/humanlayer/12-factor-agents) Principles of building LLM agents for prodcution
* [MetaGPT](https://github.com/FoundationAgents/MetaGPT) - Multi-Agent Framework for software development
