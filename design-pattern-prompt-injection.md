# Design Patterns for Securing LLM Agents

The recent paper by authors from various organizations including IBM, Invariant Labs, ETH Zurich, Google and Microsoft proposes 6 design patterns to guard against prompt injection. This I believe a great extension to [the prior work](https://arxiv.org/pdf/2503.18813) by Google researchers. 

## The Action-Selector Pattern
<img src="media/action-selector-pattern.png" width=600 />
The red color represents untrusted data. The LLM acts as a translator between a natural language prompt and a series of predefined actions to be executed over untrusted data.

## The Plan-Then-Execute Pattern
<img src="media/plan-then-execute-pattern.png" width=600 />
Before processing any untrusted data, the LLM defines a plan consisting of a series of allowed tool calls. A prompt injection cannot force the LLM into executing a tool that is not part of the defined plan.

## The LLM Map-Reduce Pattern
<img src="media/map-reduce-pattern.png" width=600 />
Untrusted documents are processed independently to ensure that a malicious document cannot impact the processing of another document.

## The Dual LLM Pattern

## The Code-Then-Execute Pattern

## The Context-Minimization pattern
