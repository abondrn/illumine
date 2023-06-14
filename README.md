# lemmata

[![Tests](https://github.com/abondrn/lemmata/workflows/Tests/badge.svg)](https://github.com/abondrn/lemmata/actions?query=workflow%3Aci)
[![pypi version](https://img.shields.io/pypi/v/lemmata.svg)](https://pypi.org/project/lemmata/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)

opinionated, batteries-included LLM framework to jumpstart your next project

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

 - deployment
   - [x] CLI entrypoint
   - [ ] rich CLI exceptions and UI
   - gradio chat UI
     - [ ] support ICE visualizer
     - [ ] regeneration, remove last message
     - [ ] file upload
     - [ ] markdown: images, tables, citations
     - [ ] add examples
     - [ ] show costs in real time
     - [ ] show which tools were invoked in real time
     - [ ] reload
     - [ ] model comparisons
     - [ ] host on Huggingface Spaces
     - [ ] human in the loop
   - publish package
     - [ ] fix Github Actions failures to deploy to PIP
     - [ ] add screenshot
     - [ ] add categories
     - [ ] host documentation site
   - [ ] API: https://github.com/jina-ai/langchain-serve/blob/main/examples/websockets/hitl/README.md
   - bot
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
 - traces
   - [x] langchain visualizer
   - [ ] persistence and replay
   - RLHF
   - [ ] API cost tracking
 - XML DSL
   - [ ] jinja syntactic sugar
   - [ ] regex generation
   - [ ] macros
   - [ ] guidance primitives
   - [ ] imports
   - [ ] guardrails extractor
 - [ ] FSM agent visualization
 - memory
   - [ ] personalities
   - [ ] setup memory using transformers and faiss
   - [ ] document upload
   - [ ] goal tree persistance
   - [ ] https://python.langchain.com/en/latest/modules/models/llms/examples/llm_caching.html
   - [ ] caching persistance
   - [ ] https://python.langchain.com/en/latest/modules/agents/agents/custom_agent_with_tool_retrieval.html
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
     - [ ] https://python.langchain.com/en/latest/modules/chains/generic/router.html
     - [ ] https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/youtube_transcript.html
     - [ ] https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/mediawikidump.html
     - [ ] https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/jupyter_notebook.html
     - [ ] https://python.langchain.com/en/latest/modules/prompts/example_selectors.html#
 - tools
   - [X] gradio tools
   - [ ] Brave Search API
   - [ ] snscrape
   - [ ] platypush
   - [ ] [confirmation](https://python.langchain.com/en/latest/modules/agents/tools/human_approval.html)
   - chains as tools
     - [ ] https://python.langchain.com/en/latest/modules/chains/generic/from_hub.html
     - [ ] https://python.langchain.com/en/latest/modules/chains/examples/flare.html
     - [ ] https://python.langchain.com/en/latest/modules/chains/examples/llm_bash.html
     - [ ] https://python.langchain.com/en/latest/modules/chains/examples/pal.html
   - [ ] .well-known/ai-plugin.json
   - [ ] https://github.com/Significant-Gravitas/Auto-GPT-Plugins
   - [ ] https://python.langchain.com/en/latest/modules/agents/toolkits/examples/playwright.html
   - [ ] https://python.langchain.com/en/latest/modules/agents/tools/tools_as_openai_functions.html
   - todo
     - [ ] checkvist
     - [ ] markdown checklists
     - [ ] todo comments
     - [ ] jira
     - [ ] github issues
     - [ ] delegation
   - coding
     - [ ] modifying files with diffs
     - [ ] https://github.com/irgolic/AutoPR
     - [ ] https://github.com/jina-ai/dev-gpt
     - [ ] code interpreter
     - [ ] Python docstring search
     - [ ] plugin generation
   - [ ] transformers agents
   - [ ] OpenStreetMap
   - [ ] stream cursors (paginated APIs)
   - [ ] metaclass for toolkits
 - tech debt
   - [ ] better specification for API keys
   - [ ] environment variables
   - [ ] generate config file commented-out, populated with default values
   - [ ] allow setting required arguments via prompt
   - [ ] add streaming of final output
   - [ ] better error messages for misspelt enum values
   - [ ] pydantic field groups
