## Observations on MMR algorithm for policy documents:
MMR evaluated against the fixed retrieval eval set (10 cases) at lambda_mult=0.5 and 0.8. Not adopted. On this corpus, chunks from the same source document are largely non-redundant (each covers a distinct policy section), so MMR's diversity penalty doesn't solve a real problem here — it actively displaces relevant same-document chunks in favor of unrelated documents to satisfy the diversity objective. Confirmed regression on case #9 ("what happens after 30 days"): Basic correctly retrieves the exception-handling chunk and answers correctly; MMR drops that chunk even at lambda_mult=0.8 and the chain reports sufficient_context=False on a question the corpus actually answers. Kept plain similarity search as the default retriever. MMR remains available (get_mmr_retriever) for future corpora where genuine chunk redundancy is a real problem (e.g. once the knowledge base grows to include overlapping historical reports), but is not the default.

### Explanation on why it did not work on these documents
When a query targets a specific policy or rule, standard Maximal Marginal Relevance (MMR) creates a severe **false-diversity trap**.

By penalizing embedding similarity among candidates in the selection pool, MMR treats semantic clustering as redundancy rather than depth. For policy and compliance corpora, this causes distinct failure modes:

* **Context Fragmentation:** A single policy often requires multiple interdependent passages (e.g., conditions, exceptions, definitions, and scope). MMR takes chunk 1 (the rule), penalizes chunk 2 (the exception) because its embedding is too similar to chunk 1, and instead selects an unrelated clause from a completely different policy.
* **Topic Bleed:** To maximize vector variance, MMR artificially forces chunks from unrelated policies into the top-$k$ context window, degrading the generator’s signal-to-noise ratio and increasing hallucination risks.

---

### Retrieval Strategies to Replace or Constrain MMR

If your corpus consists of distinct policies, replace unconstrained MMR with document-aware or relevance-first retrieval patterns:

**1. Hierarchical / Parent Document Retrieval**

* **Mechanism:** Index small chunks (128–256 tokens) for vector search, but link each chunk to its parent section or full document metadata.
* **Why it works:** Small chunks ensure high precision during the initial semantic search, but the system retrieves and injects the entire parent section/policy rather than individual sibling chunks.

**2. Two-Stage Retrieval with Cross-Encoder Reranking**

* **Stage 1 (Bi-Encoder Retrieval):** Pull a wide top-$k$ (e.g., 30–50 chunks) via standard cosine similarity or hybrid search (Dense + BM25).
* **Stage 2 (Cross-Encoder / Reranker):** Use a cross-encoder model (e.g., `bge-reranker-large`, Cohere Rerank) to score every chunk against the exact query without any intra-candidate diversity penalty. Relevant sibling chunks will naturally cluster at the top.

**3. Grouped / Document-Level MMR (Constrained Diversity)**

* **Mechanism:** Run MMR at the **document/policy level** rather than the chunk level:
1. Aggregate chunk scores to identify the top-$N$ most relevant distinct policy documents.
2. Within each selected policy, select the top-$k$ contiguous or highest-scoring chunks purely by relevance.


* **Why it works:** Guarantees document-level diversity across distinct policies while preserving dense, multi-chunk context within the target policy.

**4. Contextual Window Expansion (Chunk Stitching)**

* **Mechanism:** When a top chunk is retrieved, automatically pull its immediate neighboring chunks ($[n-1, n, n+1]$) based on document ordering, collapsing contiguous chunks into a unified block before passing them to the LLM.

---

## Observations on MultiQueryRetriever for policy documents

MultiQueryRetriever evaluated against the fixed retrieval eval set (10 cases),
under the pinned config (OpenAI `text-embedding-3-small` + `gpt-4o-mini`).
**Not adopted as default.**

Recall was 9/9, identical to plain similarity search — no case in the eval
set that Basic retrieval failed was fixed by query rewriting, including
case #3 ("what counts as an active customer?"), which was the case MQR was
specifically expected to help given the vocabulary gap between casual
phrasing and the document's formal definition language. Basic already
retrieves the correct chunk for this case without any rewriting.

**Root cause:** `text-embedding-3-small` already captures paraphrase-level
similarity between casual and formal phrasings well enough on this corpus
that generating alternate wordings surfaces nothing new. Query rewriting
solves a vocabulary-gap problem this corpus doesn't currently have.

**Cost observed:** one extra LLM call per query (variant generation) plus
25-75% more chunks returned per case (5-7 vs. a clean 4 under Basic), with
no corresponding recall improvement.

**Topic drift:** case #10 (negative control, customs fees -- not covered by
any document) showed mild drift toward `pricing_and_freight_policy.md`
under query rewriting (2 of 4 unique sources vs. Basic's 1 of 4). This did
not break the chain's honesty in practice -- `sufficient_context=False` was
returned correctly under both retrievers when tested end-to-end -- but it's
a real, measurable increase in irrelevant context volume worth watching if
the technique is revisited on a larger/noisier corpus.

**Kept available, not default:** `get_multi_query_retriever()` remains in
`vectorstore.py` for a future scenario with genuinely inconsistent internal
vocabulary (e.g. ingesting third-party documents that don't share this
project's own KPI/policy terminology).

**Emerging pattern across MMR + MQR:** both are correctly implemented
techniques that target failure modes (chunk redundancy, lexical mismatch)
this corpus doesn't actually have. Its real weak points appear to be
structural instead -- a fact buried mid-section (case #6) and the same
fact split across two documents for different purposes (case #8). This
points toward reranking and/or ParentDocumentRetriever as more likely to
show a real effect, since they target structural retrieval problems rather
than lexical or redundancy ones.
