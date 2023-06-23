from lemmata import chain
from lemmata.config import Config


# TODO assert there are no warn logs
# TODO chatgpt_plugins
# TODO include
def test_chain():
    config = Config(
        temperature=1,
        top_p=0.5,
        presence_penalty=1,
        frequency_penalty=-1,
        model="gpt-3.5-turbo",
        agent="openai-functions",
        cost_limit=1,
        # response_limit=1000,
        memory_limit=1000,
        tools=[
            "requests_get",
            "python_repl",
            "terminal",
            "arxiv",
            "wikipedia",
            "pupmed",
            "human",
        ],
        ifttt_webhooks=["spotify"],
        graphql_endpoints=["https://swapi-graphql.netlify.app/.netlify/functions/index"],
        openapi_specs=[
            "https://api.speak.com/openapi.yaml",
            "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/",
        ],
        cache=False,
        debug=True,
        api_key=[
            ("openai", "sk-KEY"),
            ("wikidata_user_agent", "lemmata"),
        ],
        interactive=False,
        visualize=False,
        gradio=False,
        dry_run=False,
        tts=False,
    )
    chat = chain.ChatSession(config=config)
    chat.get_agent_chain()
