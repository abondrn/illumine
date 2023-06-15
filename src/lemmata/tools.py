from typing import Optional, Dict, Literal, Any, Iterable, Callable
import json

import requests
from langchain.tools import tool
from langchain.tools.base import StructuredTool
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
import tiktoken

tool: Callable[[Callable], StructuredTool]


# safesearch: on, moderate, off. Defaults to "moderate".
# region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
DDG_KWARGS = {"region": "us-en", "safesearch": "moderate"}

ddgs = DDGS()

RESULT_LIMIT = 1000


QUERY = Field(
    description="Be sure to use keywords that would actually be present in the documents you are searching for",
    examples=[
        "cats dogs",
        '"cats and dogs"',
        "cats -dogs",
        "cats +dogs",
        "cats filetype:pdf",
        "dogs site:example.com",
        "cats -site:example.com",
        "intitle:dogs",
        "inurl:cats",
    ],
)


def count_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def limit_tokens(
    seq: Iterable,
    limit: int,
    stringifier: Optional[Callable[[Any], str]] = None,
    padding: int = 0,
    offset: int = 0,
) -> list:
    out = []
    total_len = 0
    for chunk in seq:
        chunk_len = count_tokens(stringifier(chunk)) if stringifier else count_tokens(chunk)
        if total_len + chunk_len + len(out) * padding + offset > limit:
            break
        else:
            out.append(chunk)
            total_len += chunk_len
    return out


def prune_json(doc):
    if isinstance(doc, list):
        return list(map(prune_json, doc))
    elif isinstance(doc, dict):
        return {k: prune_json(v) for k, v in doc.items() if v and k not in ("Result", "Height", "Width")}
    else:
        return doc


def drop_keys(doc, keys):
    return {k: v for k, v in doc.items() if k not in keys}


@tool
def ddg_lookup(query: str) -> Dict:
    """
    Looks up entities using DuckDuckGo's Quick Answers box, which aggregates structured metadata from around the web.
    Often returns nothing at all.
    """
    params = {"q": query, "format": "json", "region": "us-en"}
    result = prune_json(
        drop_keys(
            requests.get("https://api.duckduckgo.com", params=params).json(),
            ["ImageHeight", "ImageWidth", "meta", "Type"],
        )
    )
    if "Infobox" in result:
        result["Infobox"] = {
            **{c["label"]: c["value"] for c in result["Infobox"]["content"]},
            **{c["label"]: c["value"] for c in result["Infobox"]["meta"]},
            **drop_keys(result["Infobox"], ["content", "meta"]),
        }
    return result


class VideoInput(BaseModel):
    keywords: str = QUERY
    timelimit: Optional[Literal["d", "w", "m"]] = Field(description="Get results from the last day, week, or month")
    resolution: Optional[Literal["high", "standard"]]
    duration: Optional[Literal["short", "medium", "long"]]
    license_videos: Optional[Literal["creativeCommon", "youtube"]]


@tool(args_schema=VideoInput)
def ddg_videos(
    keywords: str,
    timelimit: Optional[str] = None,
    resolution: Optional[str] = None,
    duration: Optional[str] = None,
    license_videos: Optional[str] = None,
) -> list:
    """Search for videos. Returns a list of JSON objects that you should summarize into Markdown with citations."""
    return limit_tokens(
        (
            prune_json(drop_keys(vid, ["images", "embed_html", "embed_url", "image_token", "provider"]))
            for vid in ddgs.videos(keywords, **DDG_KWARGS)
        ),
        limit=RESULT_LIMIT,
        stringifier=json.dumps,
        padding=2,
        offset=2,
    )


class NewsInput(BaseModel):
    keywords: str = QUERY
    timelimit: Optional[Literal["d", "w", "m"]] = Field(description="Get results from the last day, week, or month")


@tool(args_schema=NewsInput)
def ddg_news(
    keywords: str,
    timelimit: Optional[str] = None,
) -> list:
    """Search for news. Returns a list of JSON objects that you should summarize into Markdown with citations."""
    return limit_tokens(
        (drop_keys(article, ["image"]) for article in ddgs.news(keywords, **DDG_KWARGS)),
        limit=500,
        stringifier=json.dumps,
        padding=2,
        offset=2,
    )


class MapsInput(BaseModel):
    keywords: str = QUERY
    place: Optional[str] = Field(description="if set, the other parameters are not used.")
    street: Optional[str] = Field(description="house number/street.")
    city: Optional[str]
    county: Optional[str]
    state: Optional[str]
    country: Optional[str]
    postalcode: Optional[str]
    latitude: Optional[int] = Field(description="geographic coordinate (north-south position)")
    longitude: Optional[int] = Field(
        description="""
        geographic coordinate (east-west position);
        if latitude and longitude are set, the other parameters are not used.
        """
    )
    radius: Optional[int] = Field(description="expand the search square by the distance in kilometers. Defaults to 0.")


@tool(args_schema=MapsInput)
def ddg_maps(
    keywords: str,
    place: Optional[str] = None,
    street: Optional[str] = None,
    city: Optional[str] = None,
    county: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
    postalcode: Optional[str] = None,
    latitude: Optional[str] = None,
    longitude: Optional[str] = None,
    radius: int = 0,
) -> list:
    """Geographic search for POIs and addresses."""
    return limit_tokens(ddgs.maps(keywords, **DDG_KWARGS), limit=RESULT_LIMIT, stringifier=json.dumps, padding=2, offset=2)


class TranslateInput(BaseModel):
    keywords: str = Field(description="string or a list of strings to translate")
    from_: Optional[str] = Field(description="translate from (defaults to autodetect)")
    to: Optional[str] = Field(default="en", examples=["de"], description='what language to translate. Defaults to "en"')


@tool(args_schema=TranslateInput)
def ddg_translate(
    keywords: str,
    from_: Optional[str] = None,
    to: str = "en",
) -> Dict:
    """Translate"""
    return ddgs.translate(keywords)


@tool
def ddg_suggestions(keywords: str) -> list:
    """Generate list of search completions; useful for refining short keyword search queries"""
    return ddgs.suggestions(keywords, region=DDG_KWARGS["region"])


tools = [
    ddg_lookup,
    ddg_videos,
    ddg_news,
    ddg_translate,
    ddg_suggestions,
]
