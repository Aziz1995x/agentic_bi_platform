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

## Observations on Reranking (Cohere rerank-v3.5) for policy documents

Reranking evaluated against the fixed retrieval eval set, under the pinned
config, using ContextualCompressionRetriever with fetch_k=10 candidates
re-scored down to top_n=4 (then re-tested at top_n=2).

**At top_n=4:** recall matched Basic (9/9), but with one important side
effect. Reranking's re-scoring on case #8 ("how is a seller's region
determined?") demoted `pricing_and_freight_policy.md` -- which had passed
under Basic at rank 3 -- in favor of `data_dictionary.md`, which was not in
the original expected_sources. Inspection showed this was the reranker
correctly identifying a better match: `pricing_and_freight_policy.md`'s
rank-3 hit came from an unrelated section ("Regional Price Variation")
that merely mentions sellers/regions in a pricing context, while
`data_dictionary.md` contains an explicit "Known Data Quirks" note stating
the seller-region derivation directly. This was a genuine ground-truth gap
in the original test case, not a reranking failure -- corrected
expected_sources to {kpi_definitions.md, data_dictionary.md} and confirmed
both retrievers now score 9/9 under the corrected case.

**At top_n=2 (testing whether reranking enables a safe k reduction):**
Both Basic and Reranked scored 8/9, failing on the identical case (#4,
cross-document synthesis) for the identical reason -- `customer_segmentation.md`
occupies both top-2 slots under both retrievers, crowding out
`kpi_definitions.md` regardless of ranking quality. Case #8, originally
expected to be reranking's clearest win at reduced k, turned out not to
need reranking at all once its ground truth was corrected -- both correct
sources already rank 1-2 under plain Basic similarity search.

**Verdict: no case in this eval set demonstrates reranking succeeding at a
smaller k where Basic fails.** The corpus's actual constraint on k is
structural: multi-source questions need enough slots (k>=3-4) for every
required source to survive being crowded out by a single dominant source,
which is a "how many distinct sources fit in the window" problem, not a
"which chunk ranks highest within the window" problem -- and reranking only
solves the latter. `get_reranking_retriever()` kept available (both Cohere
and local BAAI/bge-reranker-base providers implemented) for future use if
the corpus grows to include genuinely noisy/low-precision candidate pools,
but not adopted as default. k=4 remains the recommended default.

---

## Phase 5 summary: MMR, MultiQueryRetriever, and Reranking

All three Advanced RAG techniques evaluated in this phase were correctly
implemented and produced no net improvement over plain similarity search
on this corpus, each for a distinct, now-documented reason:

- **MMR** targets chunk redundancy -- but this corpus's chunks are mostly
  non-redundant (distinct sections per document), so MMR's diversity
  penalty displaces genuinely relevant same-document chunks in favor of
  unrelated ones. Confirmed regression: case #9 produced a false
  "insufficient context" response even at lambda_mult=0.8.
- **MultiQueryRetriever** targets lexical/vocabulary mismatch -- but
  text-embedding-3-small already captures paraphrase-level similarity well
  enough on this corpus's fairly standard business English that query
  rewriting surfaced nothing new. Cost: one extra LLM call and 25-75% more
  chunks per query, no recall gain.
- **Reranking** targets ranking quality within a fixed candidate window --
  but this corpus's real bottleneck is how many *distinct sources* fit
  within k for multi-document questions, not which chunk from a single
  source ranks highest. Reranking optimizes the wrong axis for this
  specific constraint.

**Unifying conclusion:** this is a small (5-document), well-organized,
single-topic-per-section corpus written in standard business English. Its
one genuine retrieval constraint is that cross-document questions need a
generous k (>=4) so no single dominant source crowds out a second required
source -- a constraint no single-retriever ranking/rewriting technique
fixes, because it's a capacity problem, not a relevance problem. All three
techniques remain implemented and available (`get_mmr_retriever`,
`get_multi_query_retriever`, `get_reranking_retriever`) for future corpora
where their target failure modes are actually present -- e.g. a larger,
noisier corpus with genuine near-duplicate content, inconsistent
third-party vocabulary, or high-precision-required ranking within a large
candidate pool. Default retriever remains plain similarity search at k=4.
