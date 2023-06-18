import argparse
import functools
import os
import sys
from typing import Any, Literal, Tuple, Union, get_args, get_origin

from langchain.callbacks import get_openai_callback
from pydantic import ValidationError
import yaml

from lemmata import chain, gradio
from lemmata.config import BaseConfig, Config, description


def get_stdin():
    """
    This will get stdin piped input if it exists
    https://stackoverflow.com/a/76062874
    """

    with os.fdopen(sys.stdin.fileno(), "rb", buffering=0) as stdin:
        if not stdin.seekable():
            for line in stdin.readlines():
                yield line.decode("utf-8")


def metavar(type_: type, default: str = "ARG") -> Union[str, Tuple[str, ...]]:
    if get_origin(type_) is tuple:
        return tuple([metavar(t, default) for t in get_args(type_)])  # type: ignore[misc]
    elif get_origin(type_) is Literal:
        return "{" + ",".join(get_args(type_)) + "}"
    elif get_origin(type_) is list:
        return f"[{metavar(get_args(type_)[0], default)} ...]"
    else:
        return {
            int: "INT",
            float: "FLOAT",
        }.get(type_, default)


# TODO https://stackoverflow.com/questions/27146262/create-variable-key-value-pairs-with-argparse-python
# TODO singularize metavar
def generate_argparser(dataclass: BaseConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Iterate over the fields of the Pydantic dataclass
    for field in dataclass.__fields__.values():
        # Get the field name
        field_name = field.alias
        extra = field.field_info.extra

        kwargs: dict = {}
        if field.type_ is bool:
            default = field.default if hasattr(field, "default") else False
            if default:
                kwargs["action"] = "store_false"
            else:
                kwargs["action"] = "store_true"
        elif get_origin(field.annotation) is list:
            item_type = get_args(field.annotation)[0]
            if get_origin(item_type) is tuple:
                kwargs["nargs"] = 2
                kwargs["action"] = argparse._AppendAction
            else:
                kwargs["nargs"] = argparse.ZERO_OR_MORE
                kwargs["action"] = argparse._StoreAction
            kwargs["metavar"] = metavar(item_type)
        elif get_origin(field.annotation) is Literal:
            kwargs["choices"] = get_args(field.annotation)
        elif "action" in extra:
            kwargs["action"] = extra["action"]
            if kwargs["action"] == "count":
                kwargs["default"] = 0
        elif issubclass(field.type_, os.PathLike):
            kwargs["type"] = str
            kwargs["metavar"] = "FILE"
        else:
            if field.type_ not in (str, int, float):
                raise TypeError((field.type_, field.field_info))
            kwargs["type"] = field.type_
            kwargs["metavar"] = metavar(field.type_, field.alias.upper())

        if "env" in extra:
            kwargs["env_var"] = field_name.upper() if extra["env"] is True else extra["env"]

        kwargs["help"] = description(field)

        default = getattr(field, "default", None)
        if default is not None:
            kwargs["default"] = default

        kebab_case = field_name.replace("_", "-")
        # Check if single-letter abbreviation is defined
        if "arg" in extra:
            parser.add_argument(field_name, nargs="?" if not field.required else 1, **kwargs)
        elif "letter" in extra:
            abbr = extra["letter"]
            if abbr is True:
                abbr = field_name[0]
            parser.add_argument(f"-{abbr}", f"--{kebab_case}", **kwargs)
        else:
            parser.add_argument(f"--{kebab_case}", **kwargs)

    return parser


# prog - The name of the program (default: os.path.basename(sys.argv[0]))
# TODO description - Text to display before the argument help (by default, no text)
# epilog - Text to display after the argument help (by default, no text)
# formatter_class - A class for customizing the help output
# conflict_handler - The strategy for resolving conflicting optionals (usually unnecessary)
# allow_abbrev - Allows long options to be abbreviated if the abbreviation is unambiguous. (default: True)
def parse_args(cls: BaseConfig, *args: Any, **kwargs: Any) -> BaseConfig:
    argparser = generate_argparser(cls)
    args = argparser.parse_args(*args, **kwargs)
    if args.config:
        with open(args.config) as f:
            config_file = yaml.safe_load(f)
        kwargs = config_file
        for k, v in kwargs.items():
            if type(v) is dict and k in cls.__fields__ and get_origin(cls.__fields__[k].annotation) is list:
                kwargs[k] = list(v.items())
    else:
        kwargs = {}
    for k, v in vars(args).items():
        if v or v == 0:
            kwargs[k] = v
    try:
        return cls(**kwargs)
    except ValidationError as e:
        # TODO allow manual entry of missing fields
        print(e.errors())
        raise e


@functools.partial(yaml.add_representer, str)
def str_presenter(dumper, data):
    lines = data.splitlines()

    if len(data.splitlines()) > 2 and max(map(len, lines)) > 20:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    else:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def listen(chat):
    import speech_recognition as sr

    r = sr.Recognizer()
    agent_chain = chat.get_agent_chain()
    with sr.Microphone() as source:
        print("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=5)
        # optional parameters to adjust microphone sensitivity
        # r.energy_threshold = 200
        # r.pause_threshold=0.5
        while True:
            print("listening now...")
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=30)
                print("Recognizing...")
                # TODO: allow selection of which recognizer to use
                text = r.recognize_sphinx(
                    audio,
                    # model="medium.en",
                    # show_dict=True,
                )["text"]
            except Exception as e:
                unrecognized_speech_text = f"Sorry, I didn't catch that. Exception was: {e}s"
                text = unrecognized_speech_text
            print(text)

            response_text = chat.feed(text, agent_chain)
            print(response_text)


def repl(chat, input_func=input):
    import readline

    readline.parse_and_bind("tab: complete")
    agent_chain = chat.get_agent_chain()
    # TODO: add quit command
    # TODO: add history
    while True:
        with get_openai_callback() as cb:
            try:
                prompt = input_func("> ")
                chat.feed(prompt, agent_chain)
            except KeyboardInterrupt:
                print(cb)
                break
            except Exception as e:
                import traceback

                traceback.print_exc()

                if chat.config.debug:
                    import pdb

                    pdb.post_mortem(e.__traceback__)
            finally:
                print(cb)


def main():
    config = parse_args(Config)
    chat = chain.ChatSession(config)

    if config.dry_run:
        agent_chain = chat.get_agent_chain()
        try:
            print(yaml.dump(agent_chain.agent.llm_chain.prompt.dict()))
        except Exception as e:
            print("Prompt preview not available for this agent type: ", agent_chain)
            raise e
    elif config.gradio:
        ui = gradio.build_gradio(chat)
        try:
            ui.launch(server_name=config.host, debug=config.debug, server_port=config.port, share=False)
        except KeyboardInterrupt:
            ui.close()
        except Exception as e:
            ui.close()
            raise e
        return ui
    elif config.stt:
        listen(chat)
    elif config.interactive:
        repl(chat)
    else:
        agent_chain = chat.get_agent_chain()
        lines = list(get_stdin())
        if config.prompt:
            lines.append(config.prompt)
        agent_chain("\n".join(lines))


if __name__ == "__main__":
    demo = main()
