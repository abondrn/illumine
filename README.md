# lemmata

[![Tests](https://github.com/abondrn/lemmata/workflows/Tests/badge.svg)](https://github.com/abondrn/lemmata/actions?query=workflow%3Aci)
[![pypi version](https://img.shields.io/pypi/v/lemmata.svg)](https://pypi.org/project/lemmata/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)

opinionated, batteries-included LLM framework to jumpstart your next project

 - deployment
   - [x] CLI entrypoint
   - [ ] publish to PIP
   - [ ] API
   - [ ] Discord
   - gradio chat UI
     - [ ] support reload
     - [ ] enable model comparisons
   - [ ] rich CLI exceptions and UI
 - traces
   - [x] langchain visualizer
   - [ ] persistence and replay
   - [ ] caching
   - [ ] RLHF
 - XML DSL
   - [ ] jinja syntactic sugar
   - [ ] regex generation
   - [ ] macros
   - [ ] guidance primitives
   - [ ] imports
   - [ ] guardrails
 - [ ] FSM agent visualization
 - memory
   - [ ] setup memory using transformers and faiss
   - [ ] goal tree persistance
 - tools
   - [X] Brave Search API
   - [ ] OpenStreetMap
   - [ ] Python docstring search
   - [ ] CLI invocation
   - [ ] llama-index
   - [ ] modifying files with diffs
   - [ ] confirmation
   - [ ] stream cursors

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
