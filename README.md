# lemmata

[![Tests](https://github.com/abondrn/lemmata/workflows/Tests/badge.svg)](https://github.com/abondrn/lemmata/actions?query=workflow%3Aci)
[![pypi version](https://img.shields.io/pypi/v/lemmata.svg)](https://pypi.org/project/lemmata/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)

An opinionated, batteries-included LLM framework to jumpstart your next AI project

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

 - Features needed before next minor release marked in bold
 - Features needed before next major release underlined

Deployment
   - [x] CLI entrypoint
   - [ ] **rich CLI exceptions and UI**
   - gradio chat UI
     - [ ] **support ICE visualizer**
     - [ ] **regeneration, remove last message**
     - [ ] **file upload**
     - [ ] **markdown: images, tables, citations**
     - [ ] add examples
     - [ ] **show costs in real time**
     - [ ] __show which tools were invoked in real time__
     - [ ] __reload__
     - [ ] model comparisons
     - [ ] __host on Huggingface Spaces__
     - [ ] human in the loop
     - [ ] retain memory when altering LLM parameters
   - publish package
     - [ ] **fix Github Actions failures to deploy to PIP**
     - [ ] __add screenshot__
     - [ ] __add categories__
     - [ ] __host documentation site__
   - [ ] API: https://github.com/jina-ai/langchain-serve/blob/main/examples/websockets/hitl/README.md
   - __bot__
     - [ ] https://github.com/spankybot/spanky.py
     - [ ] https://github.com/paulpierre/RasaGPT
     - [ ] https://github.com/botfront/rasa-for-botfront
     - [ ] Discord: https://github.com/Haste171/langchain-chatbot
   - [ ] assistant: https://github.com/project-alice-assistant/ProjectAlice/tree/master
   - [ ] twitter bot?
   - [ ] STT: https://github.com/guillaumekln/faster-whisper
   - [ ] TTS: https://github.com/coqui-ai/TTS, https://github.com/neonbjb/tortoise-tts
   - github action
     - [ ] https://github.com/alstr/todo-to-issue-action
     - [ ] https://github.com/xpluscal/selfhealing-action-express

Traces
   - [x] langchain visualizer
   - [ ] **fix Callback**
   - [ ] __persistence and replay__
   - RLHF
   - [ ] __API cost tracking__
 - __XML DSL__
   - [ ] **add utility to print initial agent prompt and exit**
   - [ ] jinja syntactic sugar
   - [ ] regex generation
   - [ ] macros
   - [ ] guidance primitives
   - [ ] imports
   - [ ] guardrails extractor
 - [ ] FSM agent visualization

Memory
   - [ ] **explicit handling of context window**
   - [ ] __personalities__
   - [ ] **setup memory using transformers and faiss**
   - [ ] __document upload__
   - [ ] __https://python.langchain.com/en/latest/modules/models/llms/examples/llm_caching.html__
   - [ ] caching persistance
   - [ ] **https://python.langchain.com/en/latest/modules/agents/agents/custom_agent_with_tool_retrieval.html**
   - llama-index
     - [ ] https://llamahub.ai/l/papers-pubmed
     - [ ] https://llamahub.ai/l/papers-arxiv
     - [ ] https://llamahub.ai/l/remote_depth
     - [ ] https://llamahub.ai/l/snscrape_twitter
     - [ ] https://llamahub.ai/l/web-rss
     - [ ] https://llamahub.ai/l/file-ipynb
     - [ ] https://llamahub.ai/l/airtable
     - [ ] caching by URL
   - loaders
     - [ ] __filter search results with https://python.langchain.com/en/latest/modules/prompts/example_selectors.html#__
     - [ ] https://python.langchain.com/en/latest/modules/chains/generic/router.html
     - [ ] https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/youtube_transcript.html
     - [ ] https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/mediawikidump.html
     - [ ] https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/jupyter_notebook.html

Tools
   - [X] gradio tools
   - [ ] **Brave Search API**
   - [ ] snscrape
   - [ ] platypush
   - [ ] __backtrack when attempting to call the same API with the same arguments__
   - [ ] **[confirmation](https://python.langchain.com/en/latest/modules/agents/tools/human_approval.html)**
   - chains as tools
     - [ ] https://python.langchain.com/en/latest/modules/chains/generic/from_hub.html
     - [ ] https://python.langchain.com/en/latest/modules/chains/examples/flare.html
     - [ ] https://python.langchain.com/en/latest/modules/chains/examples/llm_bash.html
     - [ ] https://python.langchain.com/en/latest/modules/chains/examples/pal.html
   - [ ] __.well-known/ai-plugin.json__
   - [ ] https://github.com/Significant-Gravitas/Auto-GPT-Plugins
   - [ ] https://python.langchain.com/en/latest/modules/agents/toolkits/examples/playwright.html
   - [ ] https://python.langchain.com/en/latest/modules/agents/tools/tools_as_openai_functions.html
   - __todo__ goal tree persistance
     - [ ] checkvist
     - [ ] markdown checklists
     - [ ] todo comments
     - [ ] jira
     - [ ] github issues
     - [ ] delegation
   - __coding__
     - [ ] modifying files with diffs
     - [ ] https://github.com/irgolic/AutoPR
     - [ ] https://github.com/jina-ai/dev-gpt
     - [ ] code interpreter
     - [ ] Python docstring search
     - [ ] plugin generation
   - [ ] transformers agents
   - [ ] OpenStreetMap
   - [ ] stream cursors (paginated APIs)
   - [ ] __metaclass for toolkits__
 - tech debt
   - [X] better specification for API keys
   - [ ] **swap out argparse for a CLI parsing framework that supports environment variables and completions**
   - [ ] **generate config file commented-out, populated with default values**
   - [ ] allow setting required arguments via prompt
   - [ ] __add streaming of final output__
   - [ ] better error messages for misspelt enum values
   - [ ] pydantic field groups
   - [ ] https://github.com/pydantic/pydantic/discussions/4281
   - [ ] **better handling of `KeyboardInterrupt`**
   - [ ] **remove all print statements**
   - [ ] **add more precommit checks, require all unit tests pass**
