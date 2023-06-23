from inspect import getmembers, isclass
import logging
import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Literal, Optional, Tuple

from gradio import Request
import gradio_tools
from langchain.agents import AgentType, ZeroShotAgent, create_csv_agent, load_tools
from langchain.agents.agent_toolkits import FileManagementToolkit, NLAToolkit
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.memory import ConversationTokenBufferMemory
from langchain.tools import AIPluginTool, BraveSearch, IFTTTWebhook
from langchain.tools.base import BaseTool, Tool
from langchain.tools.graphql.tool import BaseGraphQLTool
from langchain.utilities.graphql import GraphQLAPIWrapper
from pydantic import BaseModel, Field

from lemmata.config import Config
from lemmata.langchain.agent_executor import AgentExecutor
from lemmata.langchain.human_callback import HumanApprovalCallbackHandler
from lemmata.langchain.initialize_agent import initialize_agent
from lemmata.toolkits import duckduckgo_search, wikibase_sparql

logger = logging.getLogger()


def agent_to_tool(agent: AgentExecutor, name: str) -> BaseTool:
    return Tool(
        name=name,
        func=lambda inp: agent.run(input=inp),
        description="",
        return_direct=False,
    )


# TODO handle_parsing_errors: Union[bool, str, Callable[[OutputParserException], str]] = False
# TODO max_execution_time: Optional[float] = None
# TODO max_iterations: Optional[int] = 15
class ChatSession(BaseModel):
    history: list[Tuple[str, str]] = Field(default_factory=list)
    feedback: list[dict] = Field(default_factory=list)
    config: Config
    scratchpad_dir: TemporaryDirectory[str] = Field(default_factory=TemporaryDirectory)

    class Config:
        arbitrary_types_allowed = True

    def clear(self) -> None:
        self.history = []
        self.feedback = []
        self.scratchpad_dir = TemporaryDirectory()

    def get_tools(self, llm: BaseChatModel, callbacks: list) -> list[BaseTool]:
        all_tools: list[BaseTool] = load_tools(
            self.config.tools,
            llm=llm,
            callbacks=callbacks,
        )

        try:
            all_tools.extend(duckduckgo_search.DuckDuckGoToolkit().get_tools())
        except ImportError as e:
            logger.warning(e)

        try:
            all_tools.append(
                agent_to_tool(
                    wikibase_sparql.SparqlToolkit(
                        wikidata_user_agent=self.config.get_api_key("wikidata_user_agent"),
                    ).create_agent(
                        llm=llm,
                        callbacks=callbacks,
                    ),
                    name="Wikibase",
                )
            )
        except ImportError as e:
            logger.warning(e)

        # TODO: save files from session
        file_toolkit = FileManagementToolkit(
            root_dir=str(self.scratchpad_dir.name),
            selected_tools=["read_file", "write_file", "list_directory", "file_search"],
        )
        all_tools.extend(file_toolkit.get_tools())

        if brave_key := self.config.get_api_key("brave"):
            # TODO: configure count
            all_tools.append(BraveSearch.from_api_key(api_key=brave_key, search_kwargs={"count": 5}))

        for t in getmembers(gradio_tools, isclass):
            if t[0] not in ("GradioTool", "ImageCaptioningTool", "SAMImageSegmentationTool"):
                try:
                    # TODO get rid of printing
                    all_tools.append(t[1]().langchain)
                except Exception as e:
                    logger.warning("Exception occured when loading, skipping", t[0], e, exc_info=True)

        # TODO: add searching for webhooks
        if ifttt_key := self.config.get_api_key("ifttt"):
            for trigger in self.config.ifttt_webhooks:
                # TODO: initialize from tool settings
                all_tools.append(
                    IFTTTWebhook(name=trigger.title(), url=f"https://maker.ifttt.com/trigger/{trigger}/json/with/key/{ifttt_key}")
                )
        elif len(self.config.ifttt_webhooks) != 0:
            logger.warning("IFTTT token is required to load webhooks, skipping")

        try:
            for url in self.config.graphql_endpoints:
                wrapper = GraphQLAPIWrapper(graphql_endpoint=url)
                all_tools.append(BaseGraphQLTool(graphql_wrapper=wrapper))
        except ImportError as e:
            logger.warning(e)

        for url in self.config.chatgpt_plugins:
            all_tools.append(AIPluginTool.from_plugin_url(f"{url}/.well-known/ai-plugin.json"))

        for url in self.config.openapi_specs:
            # TODO: create tool names that are compatible with OpenAI functions (no dash)
            all_tools.extend(NLAToolkit.from_llm_and_url(llm, url).get_tools())

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
                logger.warning(f"Not able to load {fn} yet, skipping")

        return all_tools

    def get_agent_chain(
        self,
        memory_limit: Optional[int] = None,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> ConversationChain:
        # TODO: use verbose to set log level
        for field in ("cache", "personality", "response_limit"):
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

        memory = ConversationTokenBufferMemory(
            llm=llm,
            max_token_limit=memory_limit or self.config.memory_limit,
            memory_key="chat_history",
        )

        tools = self.get_tools(llm, callbacks)
        if self.config.agent in ("structured-react", "openai-functions"):
            if self.config.agent == "openai-functions" and not model.startswith("gpt-"):
                raise ValueError(f"Cannot use `openai-functions` agent with {model} model")
            for t in tools:
                assert isinstance(t, BaseTool), t
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
        elif self.config.agent == "plan-and-execute":
            planner = load_chat_planner(model)
            # TODO: replace with own agent executor
            executor = load_agent_executor(model, tools, verbose=self.config.verbose)
            return PlanAndExecute(planner=planner, executor=executor, verbose=self.config.verbose)
        elif self.config.agent == "autogpt":
            from langchain.experimental import AutoGPT

            # TODO: allow configuration of name and role
            agent = AutoGPT.from_llm_and_tools(
                ai_name="Tom",
                ai_role="Assistant",
                tools=tools,
                llm=llm,
                memory=memory,
                # chat_history_memory=FileChatMessageHistory("chat_history.txt"),
            )
            # Set verbose to be true
            agent.chain.verbose = self.config.verbose
            return agent
        elif self.config.agent == "babyagi":
            # [BabyAGI](https://github.com/yoheinakajima/babyagi/tree/main) by [Yohei Nakajima](https://twitter.com/yoheinakajima)
            # is an AI agent that can generate and pretend to execute tasks based on a given objective.
            from langchain import LLMChain, PromptTemplate
            from langchain.embeddings import OpenAIEmbeddings
            from langchain.experimental import BabyAGI

            todo_prompt = PromptTemplate.from_template(
                "You are a planner who is an expert at coming up with a todo list for a given objective."
                " Come up with a todo list for this objective: {objective}"
            )
            todo_chain = LLMChain(llm=llm, prompt=todo_prompt)
            tools.append(
                Tool(
                    name="TODO",
                    func=todo_chain.run,
                    description=(
                        "useful for when you need to come up with todo lists."
                        " Input: an objective to create a todo list for."
                        " Output: a todo list for that objective."
                        " Please be very clear what the objective is!"
                    ),
                )
            )

            prefix = (
                "You are an AI who performs one task based on the following objective: {objective}."
                " Take into account these previously completed tasks: {context}."
            )
            suffix = "Question: {task}\n{agent_scratchpad}"
            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["objective", "task", "context", "agent_scratchpad"],
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=[t.name for t in tools])

            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=self.config.verbose)

            from langchain.docstore import InMemoryDocstore
            from langchain.vectorstores import FAISS

            # Define your embedding model
            embeddings_model = OpenAIEmbeddings()
            # Initialize the vectorstore as empty
            import faiss

            embedding_size = 1536
            index = faiss.IndexFlatL2(embedding_size)
            vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

            # If None, will keep on going forever
            max_iterations: Optional[int] = None
            baby_agi = BabyAGI.from_llm(
                llm=llm,
                vectorstore=vectorstore,
                task_execution_chain=agent_executor,
                verbose=self.config.verbose,
                max_iterations=max_iterations,
            )
            return baby_agi
            # baby_agi({"objective": OBJECTIVE})

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

    def speak(self, text: str) -> None:
        subprocess.call(["say", text])

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
            if self.config.tts:
                self.speak(output)
        return self.history
