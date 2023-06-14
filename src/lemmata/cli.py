import argparse
import os
import sys
from typing import Optional, get_origin, Literal, get_args

import pydantic
from pydantic import BaseModel, Field, ValidationError
import yaml
from langchain.callbacks import get_openai_callback

from lemmata import gradio, chain


def get_stdin():
    """
    This will get stdin piped input if it exists
    https://stackoverflow.com/a/76062874
    """

    with os.fdopen(sys.stdin.fileno(), "rb", buffering=0) as stdin:
        if not stdin.seekable():
            for line in stdin.readlines():
                yield line.decode("utf-8")


def description(field: pydantic.fields.ModelField) -> str:
    """Standardises argument description.

    Args:
        field (pydantic.fields.ModelField): Field to construct description for.

    Returns:
        str: Standardised description of the argument.
    """
    # Construct Default String
    default = f"(default: {field.get_default()})" if not field.required and field.get_default() is not None else None

    # Return Standardised Description String
    return " ".join(filter(None, [field.field_info.description, default]))


def generate_argparser(dataclass):
    parser = argparse.ArgumentParser()

    # Iterate over the fields of the Pydantic dataclass
    for field in dataclass.__fields__.values():
        # Get the field name
        field_name = field.alias
        extra = field.field_info.extra

        kwargs = {}
        if field.type_ is bool:
            default = field.default if hasattr(field, "default") else False
            if default:
                kwargs["action"] = "store_false"
            else:
                kwargs["action"] = "store_true"
        elif get_origin(field.annotation) is list:
            kwargs["action"] = argparse._StoreAction
            kwargs["nargs"] = argparse.ZERO_OR_MORE
            kwargs["default"] = ()
        elif get_origin(field.annotation) is Literal:
            kwargs["choices"] = get_args(field.annotation)
        elif "action" in extra:
            kwargs["action"] = extra["action"]
            if kwargs["action"] == "count":
                kwargs["default"] = 0
            else:
                print(field.field_info)
        else:
            kwargs["type"] = field.type_
            if field.type_ not in (str, int, float):
                print(field.type_, field.field_info)

        if "env" in extra:
            kwargs["env_var"] = field_name.upper() if extra["env"] is True else extra["env"]

        if kwargs.get("action") not in ("store_false", "store_true", "count"):
            kwargs["metavar"] = field.alias.upper()
        kwargs["help"] = description(field)

        default = getattr(field, "default", None)
        if default is not None:
            kwargs["default"] = default

        # Check if single-letter abbreviation is defined
        if "arg" in extra:
            parser.add_argument(field_name, nargs="?" if not field.required else 1, **kwargs)
        elif "letter" in extra:
            abbr = extra["letter"]
            if abbr is True:
                abbr = field_name[0]
            parser.add_argument(f"-{abbr}", f"--{field_name}", **kwargs)
        else:
            parser.add_argument(f"--{field_name}", **kwargs)

    return parser


class Config(BaseModel):
    prompt: Optional[str] = Field(arg=0)
    manual: bool = Field(description="Human in the Loop mode", letter="m")
    verbose: int = Field(default=0, action="count", letter="v")
    log: Optional[str] = Field(letter="l", description="Where to store the logs")
    cache: bool = Field(description="Whether to cache the responses of the LLM")
    temperature: float = Field(letter="t", default=0)
    model: Literal["gpt-3.5-turbo", "gpt-4", "claude-v1", "claude-instant-v1"] = Field(default="gpt-3.5-turbo")
    openai_api_key: Optional[str] = Field(letter="k")
    anthropic_api_key: Optional[str]
    cost_limit: Optional[float] = Field(description="The maximum allowable cost before aborting the chain")
    response_limit: Optional[int] = Field(description="Number of tokens to limit the response to")
    memory_limit: Optional[int] = Field(description="Number of tokens to limit history context to")
    tools: list[str] = Field(description="Names of tools to enable")
    config: Optional[str] = Field(letter="c", description="Configuration file")
    interactive: bool = Field(letter="i", description="Spawns an interactive session")
    visualize: bool = Field(description="Requires langchain-visualizer", letter="z")
    gradio: bool = Field(description="Requires gradio", letter="g")

    host: str = Field(default="0.0.0.0")
    debug: bool = Field(letter="d")
    port: Optional[int]

    class Config:
        arbitrary_types_allowed = True


argparser = generate_argparser(Config)


def main():
    args = argparser.parse_args()
    with open(args.config) as f:
        config_file = yaml.safe_load(f)
    kwargs = config_file
    for k, v in vars(args).items():
        if v is not None:
            kwargs[k] = v
    try:
        config = Config(**kwargs)
    except ValidationError as e:
        for error in e.errors():
            print(error.json())
        print(e.json())
        return

    chat = chain.ChatSession(config)

    if config.gradio:
        ui = gradio.build_gradio(chat)
        try:
            ui.launch(server_name=args.host, debug=args.debug, server_port=args.port, share=False)
        except KeyboardInterrupt:
            ui.close()
        except Exception as e:
            ui.close()
            raise e
        return ui
    elif config.interactive:
        agent_chain = chat.get_agent_chain()
        while True:
            with get_openai_callback() as cb:
                try:
                    prompt = input("> ")
                    chat.feed(prompt, agent_chain)
                    print(cb)
                except KeyboardInterrupt:
                    print(cb)
                    break
                except Exception as e:
                    print(e)
                    print(cb)
    else:
        agent_chain(get_stdin() + config.prompt)


if __name__ == "__main__":
    demo = main()
