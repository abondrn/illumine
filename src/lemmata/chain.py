from typing import Optional, Tuple, Literal
import logging
import time
from inspect import getmembers, isclass
from tempfile import TemporaryDirectory

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.agents import load_tools, AgentType, create_csv_agent
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain
from gradio import Request
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
import gradio_tools
from langchain.tools import BraveSearch, AIPluginTool, IFTTTWebhook
from langchain.agents.agent_toolkits import FileManagementToolkit, NLAToolkit
from langchain.utilities.graphql import GraphQLAPIWrapper
from langchain.tools.graphql.tool import BaseGraphQLTool
from langchain.chat_models.base import BaseChatModel

from lemmata.tools import tools
from lemmata.config import Config
from lemmata.langchain.human_callback import HumanApprovalCallbackHandler
from lemmata.langchain.initialize_agent import initialize_agent


logger = logging.getLogger()


# TODO turn into pydantic class
# TODO handle_parsing_errors: Union[bool, str, Callable[[OutputParserException], str]] = False
# TODO max_execution_time: Optional[float] = None
# TODO max_iterations: Optional[int] = 15
class ChatSession:
    history: list[Tuple[str, str]]
    feedback: list[dict]
    config: Config

    def __init__(self, config: Config, history: Optional[list[Tuple[str, str]]] = None):
        self.history = history or []
        self.feedback = []
        self.config = config
        self.scratchpad_dir = TemporaryDirectory()

    def clear(self) -> None:
        self.history = []
        self.feedback = []

    def get_tools(self, llm: BaseChatModel, callbacks: list) -> list:
        all_tools = (
            load_tools(
                self.config.tools,
                llm=llm,
                callbacks=callbacks,
            )
            + tools
        )
        file_toolkit = FileManagementToolkit(
            root_dir=str(self.scratchpad_dir.name),
            selected_tools=["read_file", "write_file", "list_directory", "file_search"],
        )
        all_tools.extend(file_toolkit.tools())
        if brave_key := self.config.get_api_key("brave"):
            all_tools.append(BraveSearch(api_key=brave_key, search_kwargs={"count": 3}))
        if not self.config.dry_run and False:
            for t in getmembers(gradio_tools, isclass):
                if t[0] not in ("GradioTool", "ImageCaptioningTool", "SAMImageSegmentationTool"):
                    try:
                        # TODO get rid of printing
                        all_tools.append(t[1]().langchain)
                    except Exception as e:
                        logging.warning("Exception occured when loading, skipping", t[0], e)
        # TODO: add searching for webhooks
        if ifttt_key := self.config.get_api_key("ifttt"):
            for trigger in self.config.ifttt_webhooks:
                all_tools.append(
                    IFTTTWebhook(
                        # name="Spotify",
                        # description="Add a song to spotify playlist",
                        url=f"https://maker.ifttt.com/trigger/{trigger}/json/with/key/{ifttt_key}"
                    )
                )
        elif len(self.config.ifttt_webhooks) != 0:
            logging.warning("IFTTT token is required to load webhooks, skipping")
        for url in self.config.graphql_endpoints:
            wrapper = GraphQLAPIWrapper(graphql_endpoint=url)
            all_tools.append(BaseGraphQLTool(graphql_wrapper=wrapper))
        for url in self.config.chatgpt_plugins:
            all_tools.append(AIPluginTool.from_plugin_url(f"{url}/.well-known/ai-plugin.json"))
        for url in self.config.openapi_specs:
            all_tools.extend(NLAToolkit.from_llm_and_url(llm, url).tools())
        for fn in self.config.include:
            if str(fn).endswith(".csv"):
                all_tools.append(
                    create_csv_agent(
                        llm,
                        fn,
                        verbose=self.config.verbose,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    )
                )
            else:
                logging.warning(f"Not able to load {fn} yet, skipping")
        return all_tools

    def get_agent_chain(
        self,
        memory_limit: Optional[int] = None,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> ConversationChain:
        for field in ("manual", "cache", "personality", "response_limit"):
            if getattr(self.config, field):
                raise NotImplementedError(field)
        model = model or self.config.model
        if model in ("claude-v1", "claude-instant-v1"):
            raise NotImplementedError(model)

        callbacks = [
            HumanApprovalCallbackHandler(),
            FinalStreamingStdOutCallbackHandler(),
        ]
        llm = ChatOpenAI(
            callbacks=callbacks,
            temperature=temperature or self.config.temperature,
            model_name=model + "-0613",
            openai_api_key=openai_api_key or self.config.openai_api_key,
            model_kwargs={
                "top_p": top_p or self.config.top_p,
            },
        )

        if self.config.agent in ("structured-react", "openai-functions"):
            if self.config.agent == "openai-functions" and not model.startswith("gpt-"):
                raise ValueError(f"Cannot use `openai-functions` agent with {model} model")
            memory = ConversationTokenBufferMemory(
                llm=llm,
                max_token_limit=memory_limit or self.config.memory_limit,
                memory_key="chat_history",
            )
            tools = self.get_tools(llm, callbacks)
            return initialize_agent(
                tools,
                llm,
                agent={
                    "openai-functions": AgentType.OPENAI_FUNCTIONS,
                    "structured-react": AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                }[self.config.agent],
                memory=memory,
                callbacks=callbacks,
                verbose=self.config.verbose,
                handle_parsing_errors=True,
                agent_executor_kwargs={
                    "max_execution_time": None,
                    "max_iterations": 15,
                },
            )
        else:
            planner = load_chat_planner(model)
            executor = load_agent_executor(model, tools, verbose=self.config.verbose)
            return PlanAndExecute(planner=planner, executor=executor, verbose=self.config.verbose)

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
