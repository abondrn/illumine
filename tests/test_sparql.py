from lemmata.toolkits import wikibase_sparql


def test_lookup():
    toolkit = wikibase_sparql.SparqlToolkit(wikidata_user_agent="lemmata")
    assert toolkit.vocab_lookup("Malin 1") == "Q4180017"
    assert toolkit.vocab_lookup("instance of", entity_type="property") == "P31"
    assert (
        toolkit.vocab_lookup("Ceci n'est pas un q-item")
        == "I couldn't find any item for 'Ceci n'est pas un q-item'. Please rephrase your request and try again"
    )


def test_query():
    toolkit = wikibase_sparql.SparqlToolkit(wikidata_user_agent="lemmata")
    assert toolkit.run_sparql("SELECT (COUNT(?children) as ?count) WHERE { wd:Q1339 wdt:P40 ?children . }")
