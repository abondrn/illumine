from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import pydantic
from pydantic import AnyHttpUrl, BaseModel, Field, FilePath

Provider = Literal[
    "openai",
    "anthropic",
    "brave",
    "zapier_nla",
    "wolfram_alpha",
    "openweathermap",
    "bing",
    "metaphor",
    "scenexplain",
    "ifttt",
    "gplaces",
    "wikidata_user_agent",
]

STTEngine = Literal[
    "cmu_sphinx_offline",
    "gsr_api",
    "gcloud_api",
    "witai",
    "azure",
    "houndify_api",
    "ibm",
    "snowboy_hotword_offline",
    "tensorflow",
    "vosk_offline",
    "whisper_offline",
    "whisper_api",
]

Agent = Literal["structured-react", "plan-and-execute", "openai-functions", "autogpt", "babyagi"]


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


class RecognizerSettings(BaseModel):
    """https://github.com/Uberi/speech_recognition/blob/master/reference/library-reference.rst#recognizer---recognizer"""

    energy_threshold: float = Field(300, description="minimum audio energy to consider for recording")
    dynamic_energy_threshold: bool = Field(
        True,
        description=(
            "Represents whether the energy level threshold for sounds should be automatically adjusted"
            "based on the currently ambient noise level while listening."
            "Recommended for situations where the ambient noise level is unpredictable,"
            "which seems to be the majority of use cases."
            "If the ambient noise level is strictly controlled,"
            "better results might be achieved by setting this to False to turn it off."
        ),
    )
    dynamic_energy_adjustment_damping: float = Field(
        0.15,
        desciption=(
            "If the dynamic energy threshold setting is enabled, approximately the fraction of the current energy threshold"
            "that is retained after one second of dynamic threshold adjustment."
            "Can be changed (not recommended)."
            "Lower values allow for faster adjustment, but also make it more likely to miss certain phrases"
            "(especially those with slowly changing volume)."
            "As this value approaches 1, dynamic adjustment has less of an effect over time."
            "When this value is 1, dynamic adjustment has no effect."
        ),
        min=0,
        max=1,
    )
    dynamic_energy_ratio: float = Field(
        1.5,
        description=(
            "If the dynamic energy threshold setting is enabled, the minimum factor by which speech is louder than ambient noise."
            "Can be changed (not recommended)."
            "For example, the default value of 1.5 means that speech is at least 1.5 times louder than ambient noise."
            "Smaller values result in more false positives (but fewer false negatives)"
            "when ambient noise is loud compared to speech."
        ),
    )
    pause_threshold: float = Field(0.8, description="seconds of non-speaking audio before a phrase is considered complete")
    operation_timeout: Optional[float] = Field(
        None,
        description=(
            "seconds after an internal operation (e.g., an API request) starts before it times out, or ``None`` for no timeout",
        ),
    )

    phrase_threshold: float = Field(
        0.3,
        description="minimum seconds of speaking audio before we consider the speaking audio a phrase"
        "- values below this are ignored (for filtering out clicks and pops)",
    )
    non_speaking_duration: float = Field(0.5, description="seconds of non-speaking audio to keep on both sides of the recording")

    adjust_for_ambient_noise_duration: float = Field(
        1,
        description=(
            "seconds to wait while dynamically adjusting the energy threshold to account for ambient noise prior to listening,"
            "should be at least 0.5 in order to get a representative sample of the ambient noise"
        ),
    )


class ToolSettings(BaseModel):
    name: Optional[str]
    description: Optional[str]


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

    model: Literal["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k", "claude-v1", "claude-instant-v1"] = Field(
        default="gpt-3.5-turbo", description="Which LLM API to use"
    )
    agent: Agent = Field(default="structured-react")
    cost_limit: Optional[float] = Field(description="The maximum allowable cost before aborting the chain")
    response_limit: Optional[int] = Field(description="Number of tokens to limit the response to")
    memory_limit: Optional[int] = Field(description="Number of tokens to limit history context to")
    # system: str
    personality: Optional[str]

    tools: list[Union[str, ToolSettings]] = Field(description="Names of tools to enable", default_factory=list)
    api_key: list[Tuple[Provider, str]] = Field(letter="k")
    ifttt_webhooks: list[Union[str, ToolSettings]] = Field(default_factory=list)
    graphql_endpoints: list[Union[AnyHttpUrl, ToolSettings]] = Field(default_factory=list)
    chatgpt_plugins: list[Union[AnyHttpUrl, ToolSettings]] = Field(default_factory=list)
    openapi_specs: list[Union[AnyHttpUrl, ToolSettings]] = Field(default_factory=list)
    include: list[Union[Path, AnyHttpUrl]] = Field(default_factory=list)

    prompt: Optional[str] = Field(arg=0)
    interactive: bool = Field(letter="i", description="Spawns an interactive session")
    visualize: bool = Field(description="Requires langchain-visualizer", letter="z")
    gradio: bool = Field(description="Requires gradio", letter="g")
    cache: bool = Field(description="Whether to cache the responses of the LLM")
    dry_run: bool
    tts: bool = Field(description="Speaks the responses using the say command")
    stt: Optional[STTEngine] = Field(
        description="Runs a listen loop that listens to the microphone and transcribes it via sphinx"
    )
    recognizer: RecognizerSettings = Field(default_factory=RecognizerSettings)

    def get_api_key(self, service: Provider) -> Optional[str]:
        for p, k in self.api_key:
            if p == service:
                return k
        else:
            return None

    @property
    def openai_api_key(self) -> Optional[str]:
        return self.get_api_key("openai")
