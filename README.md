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
| Aug 2025 |  We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs | Usenix Security 2025 | [Paper](https://arxiv.org/pdf/2406.10279) [NotebookLM](https://notebooklm.google.com/notebook/6495317c-4a13-4909-815f-bcc4bd96f167?authuser=1) [Podcast](https://notebooklm.google.com/notebook/6495317c-4a13-4909-815f-bcc4bd96f167/audio)| llm, hallucination, packages, code generation | | |
| Jun 2025 | Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities | Google DeepMind | [TR](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf) [NotebookLM](https://notebooklm.google.com/notebook/a84c8caf-8292-4ea0-89e4-d692ed53103f?authuser=1) [Podcast](https://notebooklm.google.com/notebook/a84c8caf-8292-4ea0-89e4-d692ed53103f/audio)| reasoining, long context, agents | | |
| Jun 2025 | Reinforcement Pre-Training | arXiv, Microsoft | [Paper](https://arxiv.org/pdf/2506.08007)  [NotebookLM](https://notebooklm.google.com/notebook/e960ffcd-1c6b-49e0-a2ff-fd25101ef5d4?authuser=1) [Podcast](https://notebooklm.google.com/notebook/e960ffcd-1c6b-49e0-a2ff-fd25101ef5d4/audio)| scaling paradigms, next token prediction, reinforcement learning | | |
| Jun 2025 | [Withdrawn] CoT is Not True Reasoning, It Is Just a Tight Constraint to Imitate: A Theory Perspective | arXiv | [Paper](https://www.arxiv.org/pdf/2506.02878) | CoT, reasoning, immitation learning, structructural constraints | | |
| Jun 2025 | KNOWLEDGE or REASONING ? A Close Look at How LLMs Think Across Domains | arXiv | [Paper](https://arxiv.org/pdf/2506.02126) | Knoweledge Index, Infromation Gain, SFT, RL, Distillation | | |
| Jun 2025 | The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity | Apple | [Paper](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) [NotebookLM](https://notebooklm.google.com/notebook/8deb9d94-5638-43a8-8a3b-e55905d79422) [Podcast](https://notebooklm.google.com/notebook/8deb9d94-5638-43a8-8a3b-e55905d79422/audio)| LRM, accuracy collapse, problem complexity, reasoing | |[Note](https://github.com/nabeelxy/llm-watch/blob/main/reviews/illusion-of-thinking.md) |
| May 2025 | Can Large Reasoning Models Self-Train? | arXiv, CMU | [Paper](https://www.arxiv.org/pdf/2505.21444) | RL, self-training | | |
| May 2025 | ALPHAONE: Reasoning Models Thinking Slow and Fast at Test Time | arXiv, UIUC, Berkeley | [Paper](https://www.arxiv.org/pdf/2505.24863) | large reasoning models, slow-to-fast reasoning | | |
| May 2025 | From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning | arXiv, Stanford, Meta | [Paper](https://arxiv.org/pdf/2505.17117) | thoughts, semantic compression compression vs. semnatic fidelity | | |
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
| Jan 2025 | Zep: A Temporal Knowledge Graph Architecture for Agent Memory | arXiv | [Paper](https://arxiv.org/pdf/2501.13956) | agent memory, knowledge graph, MemGPT, DMR, RAG | | |
| Jan 2025 | Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training | arXiv | [Paper](https://arxiv.org/pdf/2501.11425) | agent, self-training | | [Note](reviews/agent-r.md) |
| Jan 2025 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | arXiv | [Report](https://arxiv.org/pdf/2501.12948) | llm, rl, cot, gpro, moe, mla | | [Note](reviews/deepseek-r1.md)|
| Dec 2024 | DeepSeek-V3 Technical Report | arXiv | [Report](https://arxiv.org/pdf/2412.19437) | llm, rl, mlha, moe, mtp | | [Note](reviews/deepseek-v3.md)| 
| Sep 2024 | Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries | Google | [Paper](https://arxiv.org/pdf/2409.12640) | long context, evaluation | | |
| Sep 2024 | RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback | ICML 2024 | [Paper](https://arxiv.org/pdf/2309.00267v3) | RFAIF, RLHF | ||
| Sep 2024 | Assessing Adversarial Robustness of Large Language Models: An Empirical Study | arXiv | [Paper](https://arxiv.org/pdf/2405.02764) | llm, aml, robustness | | |
| Jun 2022 | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | arXiv | [Paper](https://arxiv.org/pdf/2101.03961) | transformers, moe | | |

### Agents
| Date |Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| Jun 2025 | DEEP RESEARCH AGENTS: A SYSTEMATIC EXAMINATION AND ROADMAP | arXiv | [Paper](https://arxiv.org/pdf/2506.18096) [NotebookLM](https://notebooklm.google.com/notebook/2985c2ac-1c95-462c-a69a-54d74b317167) [Podcast](https://notebooklm.google.com/notebook/2985c2ac-1c95-462c-a69a-54d74b317167/audio)| deep research, mcp | | |
| May 2025 | Multi-Agent Collaboration via Evolving Orchestration | arXiv | [Paper](https://arxiv.org/pdf/2505.19591) | puppet master, RL, orchestrator, multi-agent system | | [Note](https://github.com/nabeelxy/llm-watch/blob/main/reviews/puppet-master.md)|
| May 2025 | Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents | arXiv | [Paper](https://arxiv.org/pdf/2505.09970) | Planning ahead, ReAct, planning, multi-agent | | [Note](https://github.com/nabeelxy/llm-watch/blob/main/reviews/preact.md)|
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
| May 2025 | Llama-Nemotron: Efficient Reasoning Models | arXiv, Nvidia | [Paper](https://arxiv.org/pdf/2505.00949) | efficient reasoning, llama | | |
| Apr 2025 | Phi-4-Mini-Reasoning: Exploring the Limits of Small Reasoning Language Models in Math | arXiv | [Paper](https://arxiv.org/pdf/2504.21233) | reason, slm, distill, CoT | | |
| Feb 2025 | Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling | arXiv | [Paper]() | test-time scaling (TTS), math-500, aime24, process reward models (PRMs) | | |
| Dec 2024 | Phi-4 Technical Report | arXiv | [Report](https://arxiv.org/pdf/2412.08905) | sml, distillation, synthetic data | | [Note](reviews/phi-4.md) |

### LLM/AI Applciations/Engineering
| Date | Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| Mar 2025 | REALM: A Dataset of Real-World LLM Use Cases | arXiv | [Paper](https://arxiv.org/pdf/2503.18792v1) | LLM applications, use cases, news | | |

### Graph Foundation Models
| Date |Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
|Mar 2025 | Graph Foundation Models: Concepts, Opportunities and Challenges | arXiv | [Paper](https://arxiv.org/pdf/2310.11829) | gnn, llm, gfm | | |

#### Useful Links
* Awesome Graph LLM references [repo](https://github.com/XiaoxinHe/Awesome-Graph-LLM)

### Influential Papers in NLP
| Date | Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| 2022 | Training language models to follow instructions with human feedback | OpenAI | [Paper](https://arxiv.org/pdf/2203.02155) [Notebook](https://notebooklm.google.com/notebook/6ea736d5-b5ad-4ea2-b170-f15c82b52115?authuser=3)| InstructGPT, RLHF | | |
| 2021 | LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS | Microsoft | [Paper](https://arxiv.org/pdf/2106.09685) [NotebookLM](https://notebooklm.google.com/notebook/ee05a51e-7bb9-4e2a-bee6-f90353a3d7a9) [Podcast](https://notebooklm.google.com/notebook/ee05a51e-7bb9-4e2a-bee6-f90353a3d7a9/audio)| LoRA, fine-tuning | | |
| 2020 | Language Models are Few-Shot Learners | OpenAI | [Paper](https://arxiv.org/pdf/2005.14165) [NotebookLM](https://notebooklm.google.com/notebook/d0ef99d0-400e-4b46-80f5-4c363f6b4cbe)| GPT-3 | | |
| 2018 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Google | [Paper](https://arxiv.org/pdf/1810.04805) [NotebookLM](https://notebooklm.google.com/notebook/6d726fb6-5c4a-4138-95e3-6f9223ab92bb) [Podcast](https://notebooklm.google.com/notebook/6d726fb6-5c4a-4138-95e3-6f9223ab92bb/audio)| encoder, embeddings | | |
|2017 | Attention is all you need | Google | [Paper](https://arxiv.org/pdf/1706.03762) [NotebookLM](https://notebooklm.google.com/notebook/2fff097e-444c-4d0a-892f-fa44ce7f3c8a)| transformers, nlp | | |
| 2014 | Sequence to Sequence Learning with Neural Networks | Google | [Paper](https://arxiv.org/pdf/1409.3215) [NotebookLM](https://notebooklm.google.com/notebook/98ee4c7f-fbc6-4920-83dc-e67d586fd1ac) | lstm | | |
| 2014 | GloVe: Global Vectors for Word Representation | Stanford | [Paper](https://nlp.stanford.edu/pubs/glove.pdf) [NotebookLM](https://notebooklm.google.com/notebook/38e03431-58e4-424c-96a0-9335ed2375b3) | | |
| 2013 | Distributed Representations of Words and Phrases and their Compositionality | Google | [Paper](https://arxiv.org/pdf/1310.4546) [NotebookLM](https://notebooklm.google.com/notebook/677909a2-0260-40cf-ac8b-5ef96055fdee?authuser=1) [Podcast](https://notebooklm.google.com/notebook/677909a2-0260-40cf-ac8b-5ef96055fdee/audio)| word2vec | | |
| 1997 | Long Short-Term Memory | Neural Computation, MIT | [Paper](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf) [NotebookLM](https://notebooklm.google.com/notebook/659a2cee-c528-4849-b208-b75fadd72b4f?authuser=4) [Podcast](https://notebooklm.google.com/notebook/659a2cee-c528-4849-b208-b75fadd72b4f/audio)|  LSTM | | |
### Blog posts
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
* [MetaGPT](https://github.com/FoundationAgents/MetaGPT) - Multi-Agent Framework for software development
