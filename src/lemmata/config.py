from typing import Optional, Literal, Tuple
from pathlib import Path

import pydantic
from pydantic import BaseModel, Field, FilePath


Provider = Literal["openai", "anthropic", "brave"]


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
    return " ".join(filter(None, [field.field_info.description and field.field_info.description.replace("%", "%%"), default]))


class BaseConfig(BaseModel):
    verbose: int = Field(default=0, action="count", letter="v")
    log: Optional[Path] = Field(letter="l", description="Where to store the logs")
    host: str = Field(default="0.0.0.0")
    debug: bool = Field(letter="d", description="Run in debug mode")
    port: Optional[int]
    config: Optional[FilePath] = Field(letter="c", description="Allows you to specify the values of flags in a YAML file")

    class Config:
        arbitrary_types_allowed = True


# max_tokens "The maximum number of tokens to generate in the chat completion.
# The total length of input tokens and generated tokens is limited by the model's context length."
# TODO add validators
class Config(BaseConfig):
    temperature: float = Field(
        letter="t",
        default=0,
        min=0,
        max=2,
        description="""
        What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
        while lower values like 0.2 will make it more focused and deterministic.
        We generally recommend altering this or top_p but not both.""",
    )
    top_p: Optional[float] = Field(
        letter="p",
        default=1,
        min=0,
        max=1,
        title="Top Perplexity",
        description="""
        An alternative to sampling with temperature, called nucleus sampling,
        where the model considers the results of the tokens with top_p probability mass.
        So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or temperature but not both.
        """,
    )
    presence_penalty: float = Field(
        default=0,
        min=-2,
        max=2,
        description="""
        Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far,
        increasing the model's likelihood to talk about new topics.
        """,
    )
    frequency_penalty: float = Field(
        default=0,
        min=-2,
        max=2,
        description="""
        Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far,
        decreasing the model's likelihood to repeat the same line verbatim.
        """,
    )

    model: Literal["gpt-3.5-turbo", "gpt-4", "claude-v1", "claude-instant-v1"] = Field(
        default="gpt-3.5-turbo", description="Which LLM API to use"
    )
    agent: Literal["structured-react", "plan-and-execute"] = Field(default="structured-react")
    api_key: list[Tuple[Provider, str]] = Field(letter="k")
    cost_limit: Optional[float] = Field(description="The maximum allowable cost before aborting the chain")
    response_limit: Optional[int] = Field(description="Number of tokens to limit the response to")
    memory_limit: Optional[int] = Field(description="Number of tokens to limit history context to")
    tools: list[str] = Field(description="Names of tools to enable")
    # system: str
    personality: Optional[str]

    prompt: Optional[str] = Field(arg=0)
    interactive: bool = Field(letter="i", description="Spawns an interactive session")
    visualize: bool = Field(description="Requires langchain-visualizer", letter="z")
    gradio: bool = Field(description="Requires gradio", letter="g")
    manual: bool = Field(description="Human in the Loop mode", letter="m")
    cache: bool = Field(description="Whether to cache the responses of the LLM")

    @property
    def openai_api_key(self) -> Optional[str]:
        for p, k in self.api_key:
            if p == "openai":
                return k
        else:
            return None