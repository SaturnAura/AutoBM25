"""A self-contained demo: import the package, hand it a corpus, ask it questions.

The corpus and the queries below are invented, so nothing here touches the
benchmark datasets and the whole thing runs in a few milliseconds::

    python -m autobm25 --demo           # after `pip install .`, via the CLI
    python -m autobm25.demo             # same thing, module entry point

The corpus is deliberately tiny and mixed-topic - a few brewing notes, a few
bicycle notes, two retrieval notes - so the ranking is easy to eyeball: an
on-topic query should pull its own notes to the top and leave the rest behind.
"""

import sys

# `id` + `text` is the document shape `AutoBM25.index` expects. `title` is not
# part of the API - it is only here so the demo output is readable.
DEMO_DOCS = [
    {
        "id": "c01",
        "title": "Cold brew basics",
        "text": "Cold brew is made by steeping coarse coffee grounds in cold water for "
                "twelve to eighteen hours, then filtering the concentrate and diluting "
                "it to taste.",
    },
    {
        "id": "c02",
        "title": "Espresso extraction",
        "text": "Espresso extraction depends on grind size, dose and water temperature. "
                "A finer grind raises resistance in the puck and shortens the shot time, "
                "which is why dialling in starts with the grinder.",
    },
    {
        "id": "c03",
        "title": "Pour over cones",
        "text": "A pour over cone brews a clean cup because the paper filter traps oils "
                "and fine particles that would otherwise end up in the drink.",
    },
    {
        "id": "c04",
        "title": "Chain maintenance",
        "text": "Bike chain maintenance: wipe the chain clean, apply one drop of "
                "lubricant per link, turn the cranks a few times, then wipe off the "
                "excess so grit does not stick to it.",
    },
    {
        "id": "c05",
        "title": "Tubeless tyres",
        "text": "Tubeless tyres need their sealant refreshed every few months; a slow "
                "puncture usually seals itself while you are still riding.",
    },
    {
        "id": "c06",
        "title": "Water hardness",
        "text": "Water hardness changes extraction: hard water over-extracts and tastes "
                "chalky, while very soft water tastes flat and thin.",
    },
    {
        "id": "c07",
        "title": "Roast level and density",
        "text": "Green coffee loses moisture during roasting. A darker roast is less "
                "dense and more soluble, so it usually wants a coarser grind than the "
                "same bean roasted light.",
    },
    {
        "id": "c08",
        "title": "BM25 scoring",
        "text": "BM25 ranks documents by saturated term frequency, inverse document "
                "frequency and document length normalisation.",
    },
    {
        "id": "c09",
        "title": "BM25 hyperparameters",
        "text": "The hyperparameters k1 and b control term frequency saturation and the "
                "strength of length normalisation; the usual defaults rarely suit every "
                "corpus.",
    },
    {
        "id": "c10",
        "title": "Bike computers",
        "text": "A bike computer with a barometer gives better elevation data than GPS "
                "alone, because GPS altitude drifts.",
    },
]

# `query` is required; `why` is demo commentary, not part of the API.
DEMO_QUERIES = [
    {"query": "how long should i steep cold brew", "why": "one clearly on-topic note"},
    {"query": "does grind size change espresso extraction", "why": "two notes, one on roast level"},
    {"query": "term frequency saturation and length normalisation", "why": "the two retrieval notes"},
    {"query": "bike chain lubrication", "why": "matches across the bicycle notes"},
]


def demo_documents():
    """The demo corpus as `[{"id", "text"}]` - exactly what `index()` wants."""
    return [{"id": d["id"], "text": d["text"]} for d in DEMO_DOCS]


def demo_queries():
    """The demo queries as plain strings."""
    return [q["query"] for q in DEMO_QUERIES]


def _print_params(params):
    return (
        f"k1={params['k1']} b={params['b']} k3={params['k3']} "
        f"delta={params['delta']} idf={params['idf_type']} ({params['model_variant']})"
    )


def _params_source(retriever, use_dictionary):
    """Which path produced the parameters - dictionary lookup or the fallback.

    Returns one or more lines, so the explanation can wrap.
    """
    if not use_dictionary:
        return ["heuristic rules only (parameter dictionary skipped)"]
    from .dictionary import cached_dictionary

    dictionary = cached_dictionary()
    hit = dictionary.lookup(retriever.features)
    if hit is None:
        return ["dictionary miss - a ten-document toy corpus is out of distribution",
                f"for the {len(dictionary.entries)} shipped entries, so the heuristic rules fill in",
                "(over the 16 BEIR datasets the dictionary answers 92.0% of the time)"]
    return [f"dictionary hit: {', '.join(hit['_dict_match'])}"]


def run_demo(top_k=3, use_dictionary=True, stream=None):
    """Run the demo and return the indexed retriever.

    `stream` defaults to stdout; pass any file-like object to capture the output.
    """
    from . import AutoBM25, backend_info

    out = stream if stream is not None else sys.stdout

    def say(line=""):
        print(line, file=out)

    docs = demo_documents()
    queries = demo_queries()
    titles = {d["id"]: d["title"] for d in DEMO_DOCS}
    words = sum(len(d["text"].split()) for d in DEMO_DOCS)

    say("AutoBM25 demo - no dataset directory, no labels, no tuning")
    say(f"corpus : {len(docs)} made-up documents / {words} terms")

    # Indexing is where the adaptation happens: the corpus statistics go in, the
    # BM25 hyperparameters come out. Passing the queries along adds query-side
    # statistics (query length, query/document term overlap) to the same lookup.
    retriever = AutoBM25()
    retriever.index(docs, queries, use_dictionary=use_dictionary)

    backend = backend_info()
    say(f"backend: {backend['backend']} ({backend.get('module') or 'pure python'})")
    say(f"params : {_print_params(retriever.params)}")
    for i, line in enumerate(_params_source(retriever, use_dictionary)):
        say(("source : " if i == 0 else "         ") + line)
    say()

    for i, item in enumerate(DEMO_QUERIES, 1):
        query = item["query"]
        say(f'query {i}/{len(DEMO_QUERIES)}: "{query}"   [{item["why"]}]')
        hits = retriever.search(query, top_k=top_k)
        if not hits:
            say("    (no match)")
        for rank, (doc_id, score) in enumerate(hits, 1):
            say(f"    #{rank}  {doc_id}  {score:8.4f}  {titles.get(doc_id, doc_id)}")
        say()

    # The batch form is what the CLI uses for a whole query file; here it shows
    # that you get one ranked list per query, in input order.
    batch = retriever.search_many(queries, top_k=1)
    say("search_many(queries, top_k=1) - one list per query:")
    for query, hits in zip(queries, batch):
        top = f"{hits[0][0]} ({hits[0][1]:.4f})" if hits else "(no match)"
        say(f"    {top:>14}  <- {query}")
    say()

    say(
        "Swap in your own data with: retriever = AutoBM25(); "
        'retriever.index(["doc one", "doc two"]); retriever.search("your query")'
    )
    return retriever


if __name__ == "__main__":
    run_demo()
