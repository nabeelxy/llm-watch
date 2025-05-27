# Keeping up with LLMs: Awesome LLM Guide

Since  the launch of ChatGPT in November 2022, LLMs (Large Language Models) and AI have been progressing so rapidly that it is often overwhelming to keeping up with the developments in both the industry and the academia. This repo serves as an easy guide on the LLM/AI landscape for busy people.

If you are interested in keeping up with LLM/Agent threat landscape, I encourage you to check my other repo on security for AI: [LLM Security](https://github.com/nabeelxy/ai-security-guide/tree/main/security_for_ai/llm_security) and [AI Red Teaming](https://github.com/nabeelxy/ai-security-guide/tree/main/security_for_ai/red_teaming), [Agent Security](https://github.com/nabeelxy/ai-security-guide/tree/main/security_for_ai/agents_security) and [RAG Security](https://github.com/nabeelxy/ai-security-guide/tree/main/security_for_ai/rag_security).

## Primers
* LLMs: Beginners guide
* RAG: Beginner's guide
* LLM Agents: Beginner's guide
* AI Design Patterns: [Guide](https://github.com/nabeelxy/llm-watch/blob/main/ai_design_patterns.md)


## Recent Important Publications

### Large Language Models
| Date | Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| Aug 2025 |  We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs | Usenix Security 2025 | [Paper](https://arxiv.org/pdf/2406.10279) | llm, hallucination, packages, code generation | | |
| May 2025 | Learning to Reason without External Rewards | arXiv | [Paper](https://www.arxiv.org/pdf/2505.19590) | RLVR, RLIR, GRPO | | |
| May 2025 | KumoRFM: A Foundation Model for In-Context Learning on Relational Data | Kumo | [Paper](https://kumo.ai/research/kumo_relational_foundation_model.pdf) | relational, foundation model | | |
| May 2025 |  AlphaEvolve: A coding agent for scientific and algorithmic discovery | DeepMind | [Paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf) [Blog](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)| evolution, auto discovery, research | | |
| Mar 2025 | Reasoning to Learn from Latent Thoughts | arXiv | [Paper](https://arxiv.org/pdf/2503.18866) | synthetic data generation, latent thoughts | | |
| Mar 2025 | SuperBPE: Space Travel for Language Models | arXiv | [Paper](https://arxiv.org/pdf/2503.13423) | tokenization, multi-word tokens | | |
| Mar 2025 | Chain-of-Tools: Utilizing Massive Unseen Tools in the CoT Reasoning of Frozen Language Models | arXiv | [Paper](https://www.arxiv.org/pdf/2503.16779) | CoT, tools | | |
| Mar 2025 | Gemini Embedding: Generalizable Embeddings from Gemini | arXiv, Google | [Paper](https://arxiv.org/pdf/2503.07891v1) | MMTEB benchmark, Gemini, embedding | | |
| Mar 2025 | Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning | arXiv | [Paper](https://arxiv.org/pdf/2503.09516) | RL, RAG, Reasoning, DeepSeek, tool use | | |
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
| Sep 2024 | RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback | ICML 2024 | [Paper](https://arxiv.org/pdf/2309.00267v3) | RFAIF, RLHF | ||
| Sep 2024 | Assessing Adversarial Robustness of Large Language Models: An Empirical Study | arXiv | [Paper](https://arxiv.org/pdf/2405.02764) | llm, aml, robustness | | |
| Jun 2022 | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | arXiv | [Paper](https://arxiv.org/pdf/2101.03961) | transformers, moe | | |

### Agents
| Date |Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| Apr 2025 | A Survey of AI Agent Protocols | arXiv | [Paper](https://arxiv.org/pdf/2504.16736) | protcols, survey | | |
| Feb 2025 | Understanding the planning of LLM agents: A survey | arXiv | [Paper](https://arxiv.org/pdf/2402.02716) | planning, survey | | |
| Jan 2025 | QUERYING DATABASES WITH FUNCTION CALLING | arXiv | [Paper](https://arxiv.org/pdf/2502.00032) | Agent, text-to-sql, function calling, json | | |
| Oct 2024 | Agent-as-a-Judge: Evaluate Agents with Agents | arXiv - Meta | [Paper](https://arxiv.org/pdf/2410.10934) | agent, judge | | |
| Mar 2023 | REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS | ICLR 2023 | [Paper](https://arxiv.org/pdf/2210.03629) | Agent, ReAct pattern | | |

### RAG
| Date | Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| Feb 2025 | KnowPath: Knowledge-enhanced Reasoning via LLM-generated Inference Paths over Knowledge Graphs | arXiv | [Paper](https://arxiv.org/abs/2502.12029) | RAG, knowledge graph, hullicination, reasoning paths | | |

### Benchmarks
| Date | Title | Venue | Materials | Tags | Code | Summary |
| --- |  --- | --- | --- | --- | --- | --- |
| Apr 2025 | HealthBench: Evaluating Large Language Models Towards Improved Human Health | OpenAI | [Paper](https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf) [Blog](https://openai.com/index/healthbench/) | 5K conversaions, rubrics | | |
| Apr 2025 | The LeaderBoard Illusion | arXiv | [Paper](https://arxiv.org/pdf/2504.20879) | Chatbot Arean, Eval | | |
| Jan 2025 | HALOGEN : Fantastic LLM Hallucinations and Where to Find Them | arXiv | [Paper](https://arxiv.org/pdf/2501.08292) | benchmark, hallucination |  | [Note](reviews/halogen.md) |
| 2025 |  Humanity's Last Exam |  | [Paper](https://static.scale.com/uploads/654197dc94d34f66c0f5184e/Publication%20Ready%20Humanity%27s%20Last%20Exam.pdf) | benchmark, LLM |  | [Note](reviews/hle.md) |

### Small Language Models
| Date | Title | Venue | Materials | Tags | Code | Summary |
| --- | --- | --- | --- | --- | --- | --- |
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

### Blog posts
* [2025-05-01] [Blog](https://lilianweng.github.io/posts/2025-05-01-thinking/) Why We Think by Lilian Wang. #test-time-compute #cot #thinking 
* [2025-04-15] [Report](https://knightcolumbia.org/content/ai-as-normal-technology) AI as a normal technology by Arvind Narayanan and Sayash Kapoor.
* [2025-04-13] [Blog](https://www.newsletter.swirlai.com/p/mcp-vs-a2a-friends-or-foes) MCP vs. A2A: Friends or Foes?, by Aurimas Griciunas (SwirlAI Newsletter)
* [2025-04-11] [Report](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf) AI in the enterprise by Open AI.
* [2025-04-03] [Blog](https://eugeneyan.com/writing/recsys-llm/) Improving the recommendation systems and search in the age of LLMs, by Eugene Yan.
* [2025-03-12] [Blog](https://blog.google/technology/developers/gemma-3/) Google's Gemma 3 release note.
* [2025-03-11] [Documentation](https://platform.openai.com/docs/guides/agents-sdk) OpenAI's Agent SDK guide.
* [2025-03-10] [Article](https://www.latent.space/p/why-mcp-won) Model Context Protocol, Summary of the AI Engineer summit talk by Anthropic.
* [2025-03-01] [Youtube](https://www.youtube.com/watch?v=kQmXtrmQ5Zg) MCP workshop by Mahesh Murag of Anthropic at AI Engineer summit.
* [2025-02-26] [Article](https://thomwolf.io/blog/scientific-ai.html) The Einstein AI model by Thomas Wolf, CSO, Huggingface. Thomas discusses the AI's current path of producing "obidient students" as opposed to "revolutionaries".
* [2025-02-06] [Article](https://cloudsecurityalliance.org/blog/2025/02/06/agentic-ai-threat-modeling-framework-maestro) Agentic AI threat modeling framework - MAESTRO.
* [2025-02-01] [White Paper](https://www.kaggle.com/whitepaper-agent-companion) Agents Companion by Google.
* [2024-11-25] [Article](https://www.anthropic.com/news/model-context-protocol) MCP (Model Context Protocol) by Anthropic.
