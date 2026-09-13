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

## Observations on ParentDocumentRetriever for policy documents

ParentDocumentRetriever evaluated against the fixed retrieval eval set at
two matched k values (k=4 and k=2), under the pinned config, using
child_chunk_size=250 / parent_chunk_size=1200. **Not adopted.**

**At k=4:** recall matched Basic (9/9), with one encouraging-looking but
ultimately misleading signal -- case #4's `kpi_definitions.md` moved from
rank 4 (barely surviving) under Basic to rank 3 under Parent, at the same
nominal k. This looked like early evidence that larger returned chunks
free up room in the k-budget for a second required source.

**At k=2:** that signal did not hold up, and the technique measurably
underperformed Basic -- **7/9 vs. Basic's 8/9 at the same k.**

- **Case #4 failed identically to Basic.** The two nearest *child* chunks
  (250 chars) were both from `customer_segmentation.md`; mapping them up
  to their parent sections still produced only `customer_segmentation.md`
  content. `kpi_definitions.md` never appeared. This is the key mechanical
  finding: **parent-expansion happens only after the nearest children are
  already selected by embedding similarity.** If `customer_segmentation.md`
  is simply a stronger semantic match for this specific query than
  `kpi_definitions.md` -- which it evidently is -- no amount of returning
  larger parent chunks changes *which sources* get selected in the first
  place. Chunk size affects what's returned once a chunk is chosen; it
  does not affect the choice itself.

- **Case #8 newly failed, and did not fail under Basic at k=2.** Parent's
  two nearest children collapsed, after deduplication to their shared
  parent, into a **single** returned document (`kpi_definitions.md` only)
  -- `data_dictionary.md` was dropped entirely. This exposes the actual
  root cause of the original (wrong) hypothesis: **ParentDocumentRetriever's
  `k` parameter controls how many child chunks are fetched from the
  vectorstore, not how many distinct documents are ultimately returned.**
  When multiple selected children map to the same parent, deduplication
  silently reduces the effective number of distinct sources below the
  configured k. For a corpus whose one confirmed bottleneck is "does the
  window have enough slots for every source a multi-document question
  needs" (see MMR/MQR/Reranking verdicts above), this is the worst
  possible failure mode -- it shrinks effective k exactly when the corpus
  most needs it preserved.

**Verdict:** ParentDocumentRetriever is not merely neutral on this corpus
(as MMR, MultiQueryRetriever, and Reranking were) -- it is measurably
**worse** at reduced k, because its dedup behavior compounds rather than
relieves the source-slot capacity constraint that is this corpus's actual
weakness. `build_parent_document_retriever()` kept in the codebase for
reference and for a future corpus where parent/child splitting's intended
benefit (recovering full context around a precisely-matched small
fragment) is genuinely needed -- but not adopted as default, and not
recommended as a way to safely reduce k for this corpus.

**Fifth technique, fifth confirmation of the unifying finding:** no tested
retrieval strategy (MMR, MultiQueryRetriever, Reranking, ParentDocumentRetriever)
can substitute for simply keeping k>=4 on this corpus. The bottleneck is
structural -- how many distinct sources fit in the retrieval window for a
multi-document question -- not a relevance, redundancy, or lexical-matching
problem that ranking or rewriting techniques are designed to solve.

## Observations on BM25 and Hybrid (Dense + BM25) Retrieval for policy documents

BM25-only and Hybrid (dense + BM25 via EnsembleRetriever, RRF fusion,
dense_weight=0.5) evaluated against the fixed eval set at matched k=4 and
k=2, under the pinned config.

**At k=4:** all three retrievers (Basic, BM25-only, Hybrid) scored 9/9.
No differentiating signal at this k -- consistent with every prior
technique in this phase, since k=4 already has enough slack for this
corpus's multi-source questions.

**At k=2, BM25-only performed the worst of any technique tested this
phase: 5/9**, failing cases #2, #4, #5, and #8 -- all cases requiring
either an exact-but-differently-worded fact (case #2: "7 calendar days")
or a second required source getting crowded out. This is a useful,
somewhat counterintuitive finding on its own: BM25 was expected to excel
at exact term/number matching, but at k=2 it actually failed the
"how long to request a return" case, retrieving two `kpi_definitions.md`
chunks instead -- its term-frequency scoring did not weight "7 calendar
days" highly enough relative to more frequent terms elsewhere in the
corpus. BM25 alone is not a reliable retriever for this corpus at low k.

**At k=2, Hybrid scored 9/9 -- the first and only technique in this phase
to beat Basic's k=2 recall (8/9).** Mechanism, confirmed via rank
inspection: dense search and BM25 rank case #4's two candidate documents
in *opposite* order (dense: `customer_segmentation.md` rank 1,
`kpi_definitions.md` rank 4; BM25: `kpi_definitions.md` rank 1,
`customer_segmentation.md` rank 3). Reciprocal rank fusion, combining two
rankings that disagree, let `kpi_definitions.md` survive into the fused
top-2 by being ranked well by *either* method, displacing a second,
redundant `customer_segmentation.md` chunk that dense-only search would
have kept. This is hybrid retrieval's real mechanism working as intended:
it doesn't need BM25 to be individually strong (it isn't -- 5/9 alone) --
it needs BM25 and dense search to be *wrong in different, uncorrelated
ways*, so fusion cancels out dense search's specific blind spot (over-
concentrating on one dominant document).

**Chain-level verification, however, reveals this retrieval-level win did
not translate into a real answer-quality difference for this specific
case.** Running case #4's question through both retrievers at true,
matched k=2: Basic's single surviving `customer_segmentation.md` chunk
("Relationship to Active Customer Status") already states enough to
answer the question correctly on its own -- both Basic and Hybrid produced
essentially identical, correct answers, with `sufficient_context=True`
under both. The retrieval-level gap (missing `kpi_definitions.md`) was
real but did not cost anything at the chain level for this particular
question, because the one source that survived happened to be
self-sufficient.

**Verdict: Hybrid retrieval is the first technique in Phase 5 to show a
genuine, mechanistically-explained retrieval-level improvement over Basic
at reduced k -- but the chain-level check shows this specific improvement
was a safety margin, not a demonstrated fix for a broken answer.** The
distinction matters: it did not need to be tested purely on faith. This
is arguably still worth adopting at reduced k, precisely because it costs
little (no extra LLM call, cheap to compute) and provides a real
retrieval-level safety margin against a future question where the
surviving chunk is *not* self-sufficient -- unlike this case, where we
got lucky that redundant information across sources meant one working
chunk was enough. `build_hybrid_retriever()` is recommended if `k` is
ever reduced from the default 4 for cost/latency reasons; at k=4, it adds
no measurable benefit and is not necessary as the default.

## ColBERT (via RAGatouille) — not evaluated, dependency conflicts

Attempted to integrate ColBERT (late-interaction, token-level retrieval)
via the `ragatouille` library, intending a head-to-head comparison against
BM25 on the same exact-match weakness (both target precise term/number
matching, via very different mechanisms -- classical term frequency vs.
neural late-interaction/MaxSim).

**Not completed -- blocked by two independent, unrelated dependency
conflicts before any retrieval quality could be measured:**

1. Installing `ragatouille` upgraded `pyarrow` past version 21.0.0, which
   removed the `PyExtensionType` API that the `datasets` library (a
   transitive dependency of `sentence_transformers`, already installed
   for local embeddings support since Phase 4) still depends on. This
   broke an unrelated, already-working module (`reranker.py`) as
   collateral damage. Fixed by pinning `pyarrow<21.0`.
2. Separately, `ragatouille`'s own internal code imports from
   `langchain.retrievers.document_compressors.base`, a module path that
   no longer exists in the currently installed LangChain version (moved
   during a package restructuring to `langchain_core`/`langchain_classic`).
   This is a genuine compatibility gap between `ragatouille` and current
   LangChain, not fixable by a dependency version bump without risking
   breaking every other already-verified retriever in this project.

**Decision: skip ColBERT for this project.** Two independent, unrelated
import failures before any retrieval code even ran is itself informative:
this technique carries meaningfully higher integration risk and
maintenance burden than every other technique tested in this phase, none
of which required more than adding a single well-maintained package.
Combined with the corpus-size argument established across four other
techniques (MMR, MultiQuery, Reranking, ParentDocumentRetriever) --
this corpus's real constraint is source-slot capacity at low k, not
lexical/relevance precision, which is the specific problem ColBERT
solves -- the expected benefit was already predicted to be marginal even
before the dependency issues surfaced. Not worth the integration cost for
this corpus. Revisit only if the knowledge base grows into a domain where
BM25 has already been shown insufficient AND precise term-level matching
is a demonstrated, not merely theoretical, requirement.

## Observations on Contextual Retrieval for policy documents

Contextual Retrieval (Anthropic's technique: LLM-generated situating
description prepended to each chunk before embedding) evaluated against
the fixed eval set at k=4, with a full chain-level comparison across 6
cases. **Not adopted.**

**Retrieval-level result was inconsistent across repeated index builds --
itself the most important finding.** A first run at k=4 showed case #4
regressing to a hard FAIL (all 4 slots going to `customer_segmentation.md`,
`kpi_definitions.md` dropped entirely). A second rebuild, using the exact
same code and chunks, saw case #4 pass again (matching Basic, `kpi_definitions.md`
surviving at rank 4). This is a materially different property from every
other technique tested in this phase: MMR, Reranking, ParentDocumentRetriever,
and BM25/Hybrid are all deterministic given a fixed index -- Contextual
Retrieval's index itself depends on an LLM's non-deterministic generation
at build time, so a corpus already sitting at a razor-thin k-budget margin
(case #4's `kpi_definitions.md` has ranked exactly 4th-of-4 in nearly every
technique tested this phase) can flip pass/fail from rebuild to rebuild
for reasons unrelated to the query itself. This instability alone is a
meaningful production concern independent of any quality question --
a rebuilt index should not silently change which sources are considered
retrievable for a previously-passing question.

**One genuine, real ranking improvement was observed: case #9.** Every
prior technique's top-4 (Basic, MMR, MQR, Reranking, ParentDocumentRetriever,
BM25, Hybrid) consistently surfaced "Restocking Fee," "Non-Returnable
Categories," "Eligible Reasons," or the document header -- never the
"Eligibility Window" section, which is the part of the document
containing the actual 7-day/exception-process answer. Contextual
Retrieval's generated description for that chunk ("this chunk is from
the Eligibility Window section... handling of requests made after this
period as exceptions") correctly identified and surfaced it at rank 3 --
the first technique in this phase to visibly rank that specific section
in the top-4.

**This ranking improvement did not change the generated answer.** A
full chain-level comparison across 6 cases -- including case #9 --
showed functionally identical answers between Basic and Contextual
Retrieval in every case, word-for-word equivalent on the substantive
claims. Basic was already answering case #9 correctly using whatever
mix of chunks it retrieved, because the answer-bearing content
("Requests submitted after this window are handled as exceptions...")
was accessible regardless of which specific chunk ranked highest. The
retrieval-level win was real but invisible at the layer that actually
matters for the user.

**Verdict:** Contextual Retrieval demonstrated a genuine, mechanistically
real fix for one specific retrieval-ranking problem (case #9's buried
Eligibility Window section) that four prior techniques did not achieve --
but the fix produced no measurable improvement in final answer quality,
and introduced real costs: one LLM call per chunk at every index build
(44 calls for this corpus's current chunk count, scaling linearly with
corpus size and rebuild frequency) and non-deterministic retrieval
rankings that can flip a previously-passing case without any change to
the underlying documents or query. Not adopted as default. Worth
revisiting only if the corpus grows large enough that Basic's redundant
"the answer exists somewhere in several chunks anyway" safety net stops
holding -- i.e., if a future document's key fact exists in exactly one
chunk with no redundant restatement elsewhere, contextualization's
ranking improvement would likely matter at the chain level in a way it
didn't here.

## Observations on RAPTOR for policy documents

RAPTOR (recursive clustering + LLM summarization into a multi-level
retrieval tree) evaluated against the fixed eval set at k=4. **Not
adopted -- corpus too small to produce a meaningful hierarchy, confirming
the prediction made before testing.**

Clustering collapsed after a single level: 44 leaf chunks -> 2 cluster
summaries -> stopped (fewer than 4 nodes remained to cluster further).
This is not a hierarchy in any meaningful sense -- RAPTOR's actual value
proposition (multiple levels of increasingly abstract summaries, letting
a broad thematic question match a high-level node while a specific
question matches a leaf) never had the chance to operate, since this
corpus's 5 documents don't contain enough underlying volume or thematic
diversity to support more than one trivial round of clustering.

The resulting summary nodes were actively counterproductive rather than
neutral. With only 2 coarse clusters covering 5 distinct documents, each
cluster's LLM summary necessarily blended multiple unrelated topics
(e.g. one summary node's source metadata spanned
`customer_segmentation.md, kpi_definitions.md, pricing_and_freight_policy.md,
refund_and_returns_policy.md` -- four of five documents in one node).
This caused a real regression on case #4: the mega-summary node occupied
a k-slot that `kpi_definitions.md`'s specific relevant chunk would
otherwise have filled under Basic, dropping recall to 8/9. The summary
was too broad to add retrievable value and too generic to avoid
competing with a more specific leaf chunk for space.

**Verdict: RAPTOR requires a fundamentally larger and more thematically
diverse corpus to be worth its build cost** (clustering + one LLM
summarization call per cluster, repeated per level) **and its added
retrieval-time risk** (overly broad summary nodes crowding out specific
chunks in an already capacity-constrained retrieval window). Not adopted.
Revisit only if the knowledge base grows to a scale where multiple
genuine thematic groupings exist within a single topic area -- e.g. many
historical quarterly business reports, where a "revenue trends across
2024" summary node would capture something no single report chunk could.

## Self-Query Retrieval — not evaluated, dependency conflicts

Attempted to wire `SelfQueryRetriever` against enriched chunk metadata
(`document_type`, `section_title`, `mentions_segment` -- see
metadata_enrichment.py, which is independently valid and kept active)
to test whether LLM-generated metadata filters could resolve the
recurring capacity-constraint problem identified across MMR, MultiQuery,
Reranking, and ParentDocumentRetriever (case #4's `customer_segmentation.md`
crowding out `kpi_definitions.md` at low k).

**Not completed -- blocked by cascading import failures**, distinct from
but structurally similar to the ColBERT/RAGatouille issue earlier in this
phase:

1. `SelfQueryRetriever.from_llm()`'s internal translator-selection logic
   (`_get_builtin_translator`) performs one large, unconditional import of
   ~15-20 optional third-party vectorstore integration classes from
   `langchain_community.vectorstores` (Databricks, DeepLake, and others),
   solely to support `isinstance()` checks against backends never used in
   this project (FAISS only). Several of these classes have been
   deprecated/removed from the installed `langchain_community` version.
2. A self-healing compatibility shim was attempted (catch each missing
   name, patch a dummy placeholder, retry) but introduced its own
   distinct failure: the shim's probe-import behaved differently from
   the actual deferred runtime import inside `.from_llm()`, due to
   import-order/lazy-loading behavior in the current LangChain package
   split (`langchain` / `langchain_classic` / `langchain_community`)
   that wasn't worth reverse-engineering further.

**Decision: skip Self-Query Retrieval for this project.** Combined with
the ColBERT/RAGatouille experience earlier in this phase, this establishes
a consistent, real pattern: this specific corner of the current LangChain
package ecosystem (post `langchain_classic` split) has real, non-trivial
version-alignment fragility around less-common retriever features that
eagerly import many optional integrations. This is itself a legitimate
production-relevant finding, independent of retrieval quality -- a
technique's integration cost is part of its real cost, not just its
theoretical benefit. `metadata_enrichment.py`'s structured fields
(document_type, section_title, mentions_segment) remain valid and
correctly computed (validated via content inspection across cases #4, #6,
#7, #8, #9) and could support a hand-rolled metadata-filtering step (a
simple pre-filter on the vectorstore's `search_kwargs={"filter": ...}`,
bypassing SelfQueryRetriever's LLM-driven filter construction and its
fragile translator-selection layer entirely) if this capability is
revisited later without needing the full SelfQueryRetriever machinery.
