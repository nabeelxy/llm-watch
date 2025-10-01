# Soft Token, Hard Truths

This paper suggests we shoud think of LLM reasoning in different shades instead of black and white.

The chain of thought reason uses discrete tokens and words, but what if forcing LLMs to think in
terms of these rigid tokens and words limiting their potential? Can we reason in the continuous
concept space?

This work introduces "soft tokens", probability weighted mixture of tokens, to allow LLMs to maintain
multiple reasoning paths simultaneously during training. The were able to traing using RL by adding controlled
noise to continuous represenations without needing GT reasoning examples.

They show continuous CoTs (method proposed in this work) match existing discrete token CoTs  for pass@1 and surpass
in terms of diversity for pass@32 for Math reasoning benchmark on Llama/Qwen 8B models.
