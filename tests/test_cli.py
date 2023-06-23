import pytest

from lemmata import cli
from lemmata.config import Config

# TODO test loading of config files
# TODO test interactive
# TODO mock out LLM https://python.langchain.com/en/latest/modules/models/llms/examples/fake_llm.html
# TODO test config file generation
# TODO test stdin


def test_help():
    with pytest.raises(SystemExit) as e:
        cli.parse_args(Config, ("-h",))
    assert e.value.code == 0


def test_parse_no_config_file():
    cli.parse_args(
        Config,
        (
            '"prompt" -t 1.0 -p 1.0 --model gpt-3.5-turbo --agent structured-react -k openai sk-KEY'
            " --cost-limit .03 --response-limit 100 --memory-limit 3000 -i -z -g --cache -vv"
            " -l file.log --host localhost -d --port 800 --tts --stt whisper_api"
        ).split(),
    )
