import argparse
import os
import sys
from typing import Optional, Dict, List, Any, Union, Collection, get_origin

import pydantic
from pydantic import BaseModel, Field
import yaml

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.callbacks import get_openai_callback
from langchain.schema import HumanMessage
from langchain.agents import load_tools, AgentType
from langchain.memory import ConversationTokenBufferMemory
from langchain.agents import initialize_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    LLMResult,
)
import langchain_visualizer

from lemmata.tools import tools


def get_stdin():
    '''
    This will get stdin piped input if it exists
    https://stackoverflow.com/a/76062874
    '''

    with os.fdopen(sys.stdin.fileno(), 'rb', buffering=0) as stdin:
        if not stdin.seekable():
            for line in stdin.readlines():
                yield _.decode('utf-8')


def description(field: pydantic.fields.ModelField) -> str:
    """Standardises argument description.

    Args:
        field (pydantic.fields.ModelField): Field to construct description for.

    Returns:
        str: Standardised description of the argument.
    """
    # Construct Default String
    default = None #f"(default: {field.get_default()})" if not field.required else None

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
            default = field.default if hasattr(field, 'default') else False
            if default:
                kwargs['action'] = 'store_false'
            else:
                kwargs['action'] = 'store_true'
        elif get_origin(field.annotation) is list:
            kwargs['action'] = argparse._StoreAction
            kwargs['nargs'] = argparse.ZERO_OR_MORE
            kwargs['default'] = ()
        elif 'action' in extra:
            kwargs['action'] = extra['action']
            if kwargs['action'] == 'count':
                kwargs['default'] = 0
        else:
            kwargs['type'] = field.type_

        if 'env' in extra:
            kwargs['env_var'] = field_name.upper() if extra['env'] is True else extra['env']

        if kwargs.get('action') not in ('store_false', 'store_true', 'count'):
            kwargs['metavar'] = field.alias.upper()
        kwargs['help'] = description(field)

        default = getattr(field, 'default', None)
        if default is not None:
            kwargs['default'] = default
        
        # Check if single-letter abbreviation is defined
        if 'arg' in extra:
            parser.add_argument(field_name, nargs='?' if not field.required else 1, **kwargs)
        elif 'letter' in extra:
            abbr = extra['letter']
            if abbr is True:
                abbr = field_name[0] 
            parser.add_argument(f'-{abbr}', f'--{field_name}', **kwargs)
        else:
            parser.add_argument(f'--{field_name}', **kwargs)

    return parser


class Config(BaseModel):
    prompt: Optional[str] = Field(arg=0)
    manual: bool = Field(description='Human in the Loop mode', letter='m')
    verbose: int = Field(default=0, action='count', letter='v')
    log: Optional[str] = Field(letter='l', description='Where to store the logs')
    cache: bool = Field(description='Whether to cache the responses of the LLM')
    temperature: float = Field(letter='t', default=0)
    model: str = Field(default='gpt-3.5-turbo')
    openai_api_key: str = Field(letter='k')
    cost_limit: Optional[float] = Field(description='The maximum allowable cost before aborting the chain')
    response_limit: Optional[int] = Field(description='Number of tokens to limit the response to')
    memory_limit: Optional[int] = Field(description='Number of tokens to limit history context to')
    tools: list[str] = Field(description='Names of tools to enable')
    config: Optional[str] = Field(letter='c', description='Configuration file')
    interactive: bool = Field(letter='i', description='Spawns an interactive session')
    visualize: bool = Field(description='Requires langchain-visualizer', letter='z')

    class Config:
        arbitrary_types_allowed = True


class CustomCallback(BaseCallbackHandler):
    """Base callback handler that can be used to handle callbacks from langchain."""

    def __init__(self):
        BaseCallbackHandler.__init__(self)
        self.events = []

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self.events.append(('on_llm_start', serialized, prompts, kwargs))

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.events.append(('on_llm_new_token', token, kwargs))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        self.events.append(('on_llm_end', response, kwargs))

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        self.events.append(('on_llm_error', error, kwargs))

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        self.events.append(('on_chain_start', serialized, inputs, kwargs))

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        self.events.append(('on_chain_end', outputs, kwargs))

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        self.events.append(('on_chain_error', error, kwargs))

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        self.events.append(('on_tool_start', serialized, input_str, kwargs))

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        self.events.append(('on_tool_end', output, kwargs))

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        self.events.append(('on_tool_error', error, kwargs))

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        self.events.append(('on_text', text, kwargs))

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self.events.append(('on_agent_action', action, kwargs))

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        self.events.append(('on_agent_finish', finish, kwargs))

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return False

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return False

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    @property
    def ignore_chat_model(self) -> bool:
        """Whether to ignore chat model callbacks."""
        return False


argparser = generate_argparser(Config)
callbacks = []#[CustomCallback()]


def main():
    args = argparser.parse_args()
    with open(args.config) as f:
        config_file = yaml.safe_load(f)
    kwargs = config_file
    for k, v in vars(args).items():
        if v is not None:
            kwargs[k] = v
    config = Config(**kwargs)

    llm = ChatOpenAI(
        callbacks=callbacks+[FinalStreamingStdOutCallbackHandler()],
        temperature=config.temperature,
        model_name=config.model,
        openai_api_key=config.openai_api_key,
    )
    all_tools = tools+load_tools(
        config.tools,
        llm=llm,
        callbacks=callbacks,
    )
    memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=config.memory_limit, memory_key='chat_history')
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        callbacks=callbacks,
        verbose=config.verbose,
    )
    while config.interactive:
        with get_openai_callback() as cb:
            try:
                
                prompt = input('> ')
                if config.visualize:
                    async def async_run():
                        return agent_chain.run(prompt)
                    langchain_visualizer.visualize(async_run)
                else:
                    agent_chain(prompt)
                print(cb)
            except KeyboardInterrupt as e:
                print(cb)
                print(callbacks[0].events)
                break
            except Exception as e:
                print(e)
                print(cb)


if __name__ == '__main__':
    main()