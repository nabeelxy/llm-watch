# The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity

## What is the problem addresses?
* Current evaluations primarily focuses on coding and math benchmarks where the final answer accuracy is measured. It suffers from several drawbacks.
  * It is often marred with data contamination.
  * It does not provide insight into  the structure and quality of the reasoning traces.

## How do they solve the problem?
* Use controllable puzzles that allows percise manipulation of compositional complexity while maintaining consistent logical structures.
* They specifically use 4 puzzles: Tower of Hanoi, Checker Jumping, River Crossing and Block World

## What are the key findings?
* For low-complexity tasks, non-thinking LLMs outperform LRMs.
* For medium-complexity tasks, LRMs show improvement over non-thinking models.
* For high-complexity tasks, all models collapse to near-zero accuracy.
* LRMs reduce their reasoning efforts as problem complexity increases beyond a threshold. (This suggests an internal scaling failure not caused by token limits but by intrinsic model behavior)
* 
