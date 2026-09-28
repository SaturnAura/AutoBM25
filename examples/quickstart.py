"""AutoBM25 as a library, on a corpus you invent yourself.

    python examples/quickstart.py

Four lines are the whole API:

    from autobm25 import AutoBM25

    retriever = AutoBM25()
    retriever.index(corpus)                 # list of strings, or [{"id", "text"}]
    retriever.search("your query", top_k=3) # -> [(doc_id, score), ...]
"""

from autobm25 import AutoBM25, backend_info

# A small corpus about a fictional support desk. Real text, real terms - the
# package only ever looks at the words, never at labels.
CORPUS = {
    "kb-100": "Reset a forgotten password from the account page: choose 'Forgot "
              "password', enter the email on the account, and follow the link that "
              "arrives within a few minutes. The link expires after one hour.",
    "kb-101": "Two-factor authentication codes drift out of sync when the phone "
              "clock is wrong. Enable automatic time on the device and the next "
              "code should be accepted.",
    "kb-102": "Invoices are issued on the first working day of the month and are "
              "available as PDF under Billing. Change the billing address there "
              "as well; it applies to the next invoice.",
    "kb-103": "Refunds for annual plans are prorated: cancel from Billing, and the "
              "unused months are credited to the original card within ten working "
              "days.",
    "kb-104": "The mobile app needs Android 9 or newer. On older devices the sync "
              "screen stays empty and no error is shown.",
    "kb-105": "Export all of your data as JSON from Settings > Export. Large "
              "accounts are prepared in the background and you get an email when "
              "the archive is ready.",
}

QUERIES = [
    "i forgot my password and the reset link has expired",
    "authenticator code is rejected",
    "where can i download my invoice as pdf",
    "cancel my yearly plan and get money back",
    "app does not sync on my old phone",
]


def main():
    info = backend_info()
    print(f"backend: {info['backend']} ({info.get('module') or 'pure python'})\n")

    # index() accepts a plain list of strings (ids become "0", "1", ...) or a list
    # of {"id", "text"} dicts when you want to keep your own ids, as here.
    corpus = [{"id": doc_id, "text": text} for doc_id, text in CORPUS.items()]

    retriever = AutoBM25()
    retriever.index(corpus, queries=QUERIES)  # queries are optional, they only sharpen the feature vector

    # Chosen for this corpus in one O(1) dictionary lookup, without any labels.
    print("adaptive params:", retriever.params)
    print()

    for query in QUERIES:
        print(f'query: "{query}"')
        for rank, (doc_id, score) in enumerate(retriever.search(query, top_k=2), 1):
            # score is BM25 relevance: comparable within one query, not across queries
            print(f"   {rank}. {doc_id}  {score:8.4f}  {CORPUS[doc_id][:58]}...")
        print()

    # search_many() runs a whole query list in one call; the Rust backend
    # parallelises it, the Python backend loops over the same code path.
    ranked = retriever.search_many(QUERIES, top_k=1)
    print("search_many(QUERIES, top_k=1) -> one ranked list per query:")
    for query, hits in zip(QUERIES, ranked):
        best = f"{hits[0][0]} {hits[0][1]:.3f}" if hits else "(no match)"
        print(f"   {best:>14}  <- {query}")


if __name__ == "__main__":
    main()
