# lemmata

[![Tests](https://github.com/abondrn/lemmata/workflows/Tests/badge.svg)](https://github.com/abondrn/lemmata/actions?query=workflow%3Aci)
[![pypi version](https://img.shields.io/pypi/v/lemmata.svg)](https://pypi.org/project/lemmata/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)

An opinionated, batteries-included LLM framework to jumpstart your next AI project

Builds tooling on top of Langchain so that creating, editing, using, and evaluating agents is as simple as possible. By extending langchain's primitives and staying modular, you are able to manage complexity while being able to swap and combine components that are most suitable for your domain.

Key features
 - Out-of-the-box CLI, Gradio web UI, and API
 - Recording of traces with feedback for result inspection, performance monitoring, model comparison, and finetuning
 - XML-based DSL for defining, sharing, and versioning Langchain-compatible, language-agnostic prompts
 - Goal tree persistance for long running and collaborative agents
 - Standardized utilities for defining large toolkits usable by agents, humans, and synthesized programs

## Requirements

lemmata requires Python >=3.9

## Installation

It is recommended to install with `pipx`, if `pipx` haven't been installed yet, refer to the [pipx's docs](https://github.com/pipxproject/pipx)

```bash
$ pipx install lemmata
```

Alternatively, install with `pip` to the user site:

```bash
$ python -m pip install --user lemmata
```

## Roadmap

 - Features needed before next minor release **marked in bold**
 - Features needed before next major release __underlined__

Deployment
   - [x] CLI entrypoint
   - [ ] **rich CLI exceptions and UI**
   - Docker container
   - gradio chat UI
     - [ ] **support ICE visualizer**
     - [ ] **regeneration, remove last message**
     - [ ] **file upload**
     - [ ] **markdown: images, tables, citations**
     - [ ] add examples
     - [ ] **show costs in real time**
     - [ ] __show which tools were invoked in real time__
     - [ ] model comparisons
     - [ ] __host on Huggingface Spaces__
     - [ ] human in the loop
     - [ ] retain memory when altering LLM parameters
   - publish package
     - [ ] **fix Github Actions failures to deploy to PIP**
     - [ ] **setup up gitflow**
     - [ ] __add screenshot__
     - [ ] __add categories__
     - [ ] __host documentation site__
   - github action
     - [ ] https://github.com/alstr/todo-to-issue-action
     - [ ] https://github.com/xpluscal/selfhealing-action-express

Traces
   - [x] langchain visualizer
   - [ ] **fix Callback**
   - [ ] __persistence and replay__
   - RLHF
   - [ ] __API cost tracking__

Memory
   - [ ] **explicit handling of context window**
   - [ ] __personalities__
   - [ ] **setup memory using transformers and faiss**
   - [ ] __document upload__
   - [ ] __https://python.langchain.com/en/latest/modules/models/llms/examples/llm_caching.html__
   - [ ] caching persistance
   - [ ] **https://python.langchain.com/en/latest/modules/agents/agents/custom_agent_with_tool_retrieval.html**
   - [ ] __filter search results with https://python.langchain.com/en/latest/modules/prompts/example_selectors.html#__
   - [ ] https://github.com/nomic-ai/semantic-search-app-template

Tools
   - gradio tools
     - [ ] add image loading: https://python.langchain.com/docs/use_cases/multi_modal/image_agent
     - [ ] add video loading
     - [ ] add audio loading
   - [ ] enable loading tools from paths
   - [ ] enable warning when required modules don't exist (and adding to dependencies interactively based on config)
   - **Brave Search API**
     - [ ] add setup instructions
   - [ ] snscrape
   - [ ] platypush
   - [ ] __backtrack when attempting to call the same API with the same arguments__
   - [X] **[confirmation](https://python.langchain.com/en/latest/modules/agents/tools/human_approval.html)**
   - chains as tools
     - [ ] https://python.langchain.com/en/latest/modules/chains/examples/flare.html
     - [ ] https://python.langchain.com/en/latest/modules/chains/examples/llm_bash.html
     - [ ] https://python.langchain.com/en/latest/modules/chains/examples/pal.html
   - .well-known/ai-plugin.json
  - https://python.langchain.com/en/latest/modules/chains/examples/openapi.html
    - [ ] extract all GET requests from a spec
    - [ ] exclude output paths
    - [X] https://python.langchain.com/en/latest/modules/agents/toolkits/examples/openapi.html
    - [ ] authentication
   - [ ] https://github.com/Significant-Gravitas/Auto-GPT-Plugins
   - [ ] https://python.langchain.com/en/latest/modules/agents/toolkits/examples/playwright.html
   - https://python.langchain.com/en/latest/modules/agents/tools/tools_as_openai_functions.html
     - [ ] fix bug where when running the visualizer, no functions are called
   - __todo__ goal tree persistance
     - [ ] checkvist
     - [ ] markdown checklists
     - [ ] todo comments
     - [ ] jira
     - [ ] github issues
     - [ ] delegation
   - [ ] stream cursors (paginated APIs)
   - [ ] https://index.discord.red/

Tech Debt
   - [X] better specification for API keys
   - [ ] **swap out argparse for a CLI parsing framework that supports environment variables and completions**
   - [ ] allow setting required arguments via prompt
   - [ ] __add streaming of final output__
   - [ ] better error messages for misspelt enum values
   - [ ] pydantic field groups
   - [ ] https://github.com/pydantic/pydantic/discussions/4281
   - [ ] **better handling of `KeyboardInterrupt`**
   - [ ] **remove all print statements**
   - [ ] **add more precommit checks, require all unit tests pass**
   - [ ] __add documentation for all major methods__
   - [ ] __typecheck all major methods__
   - [ ] **add utility to print initial agent prompt and exit**
   - lifecycle commands
     - [ ] **generate config file commented-out, populated with default values**
     - hot swapping
       - [ ] modify gradio reload to work with input arguments
