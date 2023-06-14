from typing import Optional, Dict, List, Any, Union, Tuple, Literal
import logging
import time

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.agents import load_tools, AgentType
from langchain.memory import ConversationTokenBufferMemory
from langchain.agents import initialize_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    AgentAction,
    AgentFinish,
    LLMResult,
)
from langchain.chains import ConversationChain
from gradio import Request

from lemmata.tools import tools
from lemmata.cli import Config


logger = logging.getLogger()


class CustomCallback(BaseCallbackHandler):
    """Base callback handler that can be used to handle callbacks from langchain."""

    events: list[tuple]

    def __init__(self):
        BaseCallbackHandler.__init__(self)
        self.events = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts running."""
        self.events.append(("on_llm_start", serialized, prompts, kwargs))

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.events.append(("on_llm_new_token", token, kwargs))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.events.append(("on_llm_end", response, kwargs))

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when LLM errors."""
        self.events.append(("on_llm_error", error, kwargs))

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain starts running."""
        self.events.append(("on_chain_start", serialized, inputs, kwargs))

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        self.events.append(("on_chain_end", outputs, kwargs))

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when chain errors."""
        self.events.append(("on_chain_error", error, kwargs))

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Run when tool starts running."""
        self.events.append(("on_tool_start", serialized, input_str, kwargs))

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        self.events.append(("on_tool_end", output, kwargs))

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when tool errors."""
        self.events.append(("on_tool_error", error, kwargs))

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""
        self.events.append(("on_text", text, kwargs))

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Run on agent action."""
        self.events.append(("on_agent_action", action, kwargs))

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        self.events.append(("on_agent_finish", finish, kwargs))

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


class ChatSession:
    history: list[Tuple[str, str]]
    feedback: list[dict]
    config: Config

    def __init__(self, config: Config, history: Optional[list[Tuple[str, str]]] = None):
        self.history = history or []
        self.feedback = []
        self.config = config

    def clear(self) -> None:
        self.history = []
        self.feedback = []

    def get_agent_chain(
        self,
        memory_limit: Optional[int] = None,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> ConversationChain:
        callbacks = [CustomCallback()]
        llm = ChatOpenAI(
            callbacks=[FinalStreamingStdOutCallbackHandler()],
            temperature=temperature or self.config.temperature,
            model_name=model or self.config.model,
            openai_api_key=openai_api_key or self.config.openai_api_key,
            model_kwargs={
                "top_p": top_p or self.config.top_p,
            },
        )
        memory = ConversationTokenBufferMemory(
            llm=llm,
            max_token_limit=memory_limit or self.config.memory_limit,
            memory_key="chat_history",
        )
        return initialize_agent(
            tools
            + load_tools(
                self.config.tools,
                llm=llm,
                callbacks=callbacks,
            ),
            llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            callbacks=callbacks,
            verbose=self.config.verbose,
            agent_executor_kwargs={
                "max_execution_time": None,
                "max_iterations": 15,
            },
        )

    def vote_last_response(self, vote_type: Literal["upvote", "downvote"], request: Optional[Request] = None) -> dict:
        feedback = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "history": self.history,
            "config": self.config.dict(),
            "ip": request.client.host if request else None,
        }
        logger.info(feedback)
        self.feedback.append(feedback)
        return feedback

    def feed(
        self,
        inp: str,
        agent_chain: ConversationChain,
    ) -> list[Tuple[str, str]]:
        """Execute the chat functionality."""
        # If chain is None, that is because no API key was provided.
        if self.config.openai_api_key is None:
            self.history.append((inp, "Please paste your OpenAI key to use"))
        # Run chain and append input.
        if self.config.visualize:
            import langchain_visualizer

            async def async_run() -> dict:
                output = agent_chain.run(input=inp)
                return output

            langchain_visualizer.visualize(async_run)
        else:
            output = agent_chain(inp)["output"]
            self.history.append((inp, output))
        return self.history
