# Keep up with LLMs

Since  the launch of ChatGPT in November 2022, LLMs (Large Language Models) and AI have been progressing so rapidly that it is often overwhelming to keeping up the developments in both the industry and the academia. This repo serves as an easy guide on the LLM/AI landscape for busy people.

## Primers
* LLMs: Beginners guide
* RAG: Beginner's guide
* LLM Agents: Beginner's guide


## Recent Important Publications

### Large Language Models
| Date | ID | Title | Venue | Materials | Tags | Short Summary | Summary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jan 2025 | Agent-R | Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training | arXiv | [Paper](https://arxiv.org/pdf/2501.11425) | agent, self-training | This paper by ByteDance proposes a framework for training LLMs to perform on-the-fly self-correction in interactive environments. Unlike traditional methods that reward or penalize actions solely based on correctness, their approach leverages Monte Carlo Tree Search (MCTS) to construct training samples that recover correct trajectories from erroneous ones. This helps to recover from mistakes and avoid unproductive loops, outperforming existing methods.| [Note](agent-r.md) |
| Jan 2025 | DeepSeek-R1 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | arXiv | [Paper](https://arxiv.org/pdf/2501.12948) | llm, rl, cot, gpro, moe, mla | | |
| Dec 2024 | DeepSeek-V3 | DeepSeek-V3 Technical Report | arXiv | [Paper](https://arxiv.org/pdf/2412.19437) | llm, rl, mlha, moe, mtp | | | 


### Benchmarks
| Date | ID | Title | Venue | Materials | Tags | Short Summary | Summary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | HLE | Humanity's Last Exam |  | [Paper](https://static.scale.com/uploads/654197dc94d34f66c0f5184e/Publication%20Ready%20Humanity%27s%20Last%20Exam.pdf) | benchmark, LLM | Researcher from Scale AI, and Center for AI Safety came up with a benchmark consistings of 3000 question across dozens of subjects, including mathematics, humanities, and the natural science. DeepSeek-R1 is outperforming other state of the art models like O1 and Gemini 2 Flash Thinking. However, it should be noted that all models scored less than 10% accuracy on the benchmark raising the usefulness of the comparison. | [Note](hle.md) |

### Small Language Models
| Date | ID | Title | Venue | Materials | Tags | Short Summary | Summary |
| --- | --- | --- | --- | --- | --- | --- | --- |

### LLM/AI Engineering
| Date | ID | Title | Venue | Materials | Tags | Short Summary | Summary |
| --- | --- | --- | --- | --- | --- | --- | --- |
