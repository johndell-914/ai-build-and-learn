# Graph RAG with Neo4j on Flyte 2

A complete Graph-RAG demo running on the DGX Spark Flyte 2 devbox: Neo4j 5
as a Flyte app (with native vector index), a pipeline that pulls papers
from Semantic Scholar by keyword query (cached) and loads them as a
graph, and a Gradio chat UI with three retrieval modes (pure vector,
vector + 1-hop graph expand, hybrid RRF) talking to Gemma 4 through vLLM.

```
┌────────────────────┐  HTTP /db/neo4j/tx/commit  ┌────────────────────┐
│   pipeline.py      │ ─────────────────────────▶ │   neo4j_app.py     │
│ S2 keyword fetch   │                            │  Flyte AppEnv      │
│  → bge-small       │   nodes, edges,            │  neo4j:5.26        │
│  → UNWIND MERGE    │   vector index             │  HTTP on 7474      │
└────────────────────┘                            └─────────┬──────────┘
                                                            │
                                                            │ Cypher
                                                            ▼
┌────────────────────┐  Bge-small encode + Cypher  ┌────────────────────┐
│  Gemma 4 26B vLLM  │ ◀────────────────────────── │   chat_app.py      │
│  (sibling project) │   chat completion stream    │  Gradio AppEnv     │
└────────────────────┘                             │  3 retrieval modes │
                                                   └────────────────────┘
```

## Files

| File | What it does |
|------|--------------|
| `config.py` | Flyte `TaskEnvironment` for the pipeline, shared Neo4j connection constants. |
| `Dockerfile.neo4j` | One-line wrapper around `neo4j:5.26-community`. Built via `flyte.Image.from_dockerfile` to skip the `USER flyte` footer that breaks the container. |
| `neo4j_app.py` | Flyte `AppEnvironment` running the neo4j image. HTTP on 7474, no persistence. |
| `pipeline.py` | Three tasks: `fetch_papers` (Semantic Scholar, cached) → `embed_papers` → `load_neo4j` (HTTP Cypher, batched via UNWIND), wrapped by `graphrag_pipeline`. |
| `chat_app.py` | Gradio `AppEnvironment` with three retrieval modes (vector, vector + expand, hybrid RRF). Streams from Gemma 4 vLLM, queries Neo4j over HTTP. |
| `snapshot.py` | Two Flyte tasks: `snapshot_neo4j` dumps the live graph to a `flyte.io.Dir` (JSONL, embeddings included); `restore_neo4j` replays it back. |
| `requirements.txt` | Local deps: `flyte[tui]`, `httpx`, `sentence-transformers`, `gradio`, `openai`, `kubernetes`. |

## Why this is shaped the way it is

Two things to know up front. They explain choices that look weird in the
code and that you will hit immediately if you try to swap things around.

**HTTP, not Bolt.** Flyte 2 deploys apps as Knative Serving services. The
queue-proxy sidecar that fronts every Knative pod only routes HTTP. Bolt
(TCP/7687) does not pass through. We use Neo4j's HTTP Cypher API on 7474
instead, which supports the full Cypher surface including the native
vector-index queries.

**`from_dockerfile`, not `from_base`.** The installed Flyte 2.2.3 image
builder appends `USER flyte` and `WORKDIR /home/flyte` to every image it
builds. The official neo4j image has no `flyte` user, so containerd fails
container creation with `no users found`. `from_dockerfile` skips that
footer entirely, which is why `Dockerfile.neo4j` is a one-liner: a bare
`FROM neo4j:5.26-community` is enough.

## Prereqs

Same Flyte 2 devbox as the vectorstore project, **started with `--gpu`** so the
sibling Gemma 4 vLLM app has a GPU to schedule onto:

```bash
flyte start devbox --gpu
docker exec flyte-devbox nvidia-smi -L           # verify GB10 visible
kubectl get nodes -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}'
# should print: 1
```

If the box was already started without `--gpu`, the only way to flip it on is
`flyte delete devbox` then `flyte start devbox --gpu` (a plain stop/start
just resumes the existing container without re-applying flags). See
`../../gemma4/gemma4-dgx-devbox/SPARK_SETUP.md` for the full GPU setup.

Set up the venv:

```bash
cd topics/graphs-neo4j/graphrag-neo4j-flyte
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Same Flyte CLI config (shared `flytesnacks/development`):

```bash
flyte create config \
    --endpoint localhost:30080 \
    --project flytesnacks \
    --domain development \
    --builder local \
    --insecure
```

## 1. Deploy Neo4j

```bash
python neo4j_app.py
```

Pulls `neo4j:5.26-community` (multi-arch; arm64 picked automatically on the
DGX Spark), deploys it as a Flyte app named `graphrag-neo4j` with
`replicas=(1,1)` (always-on), exposes HTTP on 7474.

After deploy:

```
Neo4j app deployed: http://graphrag-neo4j-flytesnacks-development.localhost:30081/
  HTTP Cypher API: .../db/neo4j/tx/commit
  Browser UI:      .../browser/
  User: neo4j
  Password: graphrag-demo
```

The Neo4j browser UI loads through Knative since 7474 is HTTP, but its
connect dialog defaults to Bolt on 7687, which Knative does not route.
To actually log in, port-forward the pod (not the Knative service: those
only expose 80/443) and connect over HTTP. `--address 0.0.0.0` makes the
forward reachable from another machine on the same network (Tailscale,
LAN), which is how the demo is usually run:

```bash
# revision number drifts, so look up the live pod
kubectl -n flyte get pods | grep graphrag-neo4j

kubectl -n flyte port-forward --address 0.0.0.0 \
    pod/<that-pod-name> 7474:7474
```

Open `http://<host>:7474` in a browser, where `<host>` is the Spark's
Tailscale/LAN IP (or `localhost` if you're on the Spark itself). In the
connect dialog, pick `http://` from the protocol dropdown (not `bolt://`)
and use the **same host** as the connect URL, since the browser JS runs
on the client:

```
Connect URL:  http://<host>:7474
Username:     neo4j
Password:     graphrag-demo
```

## 2. Run the pipeline

```bash
flyte run pipeline.py graphrag_pipeline
```

Defaults: query `"retrieval augmented generation language models"`,
`max_papers=400`. Both are CLI-overridable. `fetch_papers` is cached on
those two args, so re-runs with the same query skip Semantic Scholar.
`wipe_first=True` makes Neo4j reloads idempotent.

```bash
flyte run pipeline.py graphrag_pipeline \
    --query "graph neural networks" --max_papers 200
flyte run pipeline.py graphrag_pipeline --wipe_first false
```

The fetcher makes two S2 calls: `/paper/search/bulk` (one shot, up to
1000 results, sorted by `citationCount:desc`) for paper metadata, then
`/paper/batch` (up to 500 IDs per POST) for the citation edges. Bulk
doesn't accept `references.*` in its `fields` parameter so the second
call is necessary; bulk also avoids the brutal rate-limit behavior on
the relevance-paginated `/paper/search` endpoint. Citation-sorting
yields a stronger demo corpus: the most-cited papers matching the
query, which the foundational ones (RAG, BERT, GPT-3, Self-RAG) all
land in. Both calls retry on 429 / 5xx with exponential backoff. If
the references call fails after retries, the graph still loads with
no CITES edges (modes 1 and 3 still work; mode 2 falls back to
AUTHORED_BY / IN_CATEGORY neighbors). Set `S2_API_KEY` in the env to
skip the anonymous shared limit. The successful result is cached, so
this only matters on the first run for a given query/max_papers.

Typical counts on the default query, 400 papers (varies as S2's index updates):

```
papers: ~380       # some papers in S2 don't have abstracts; we drop those
authors: ~1500
categories: ~5     # S2 fieldsOfStudy is coarse: CS, Linguistics, …
cites_edges: ~600  # only edges where both endpoints are in the corpus
authored_edges: ~1500
```

## 3. Verify the vector index

Quickest check from a Python REPL inside the venv. The Knative ingress
serves Neo4j directly, so no port-forward is needed if you set the Host
header:

```python
import httpx
from sentence_transformers import SentenceTransformer

vec = SentenceTransformer("BAAI/bge-small-en-v1.5") \
        .encode(["retrieval augmented generation"], normalize_embeddings=True) \
        .tolist()[0]

c = httpx.Client(
    base_url="http://localhost:30081",
    headers={"Host": "graphrag-neo4j-flytesnacks-development.localhost"},
    auth=("neo4j", "graphrag-demo"),
    timeout=20.0,
)
r = c.post("/db/neo4j/tx/commit", json={"statements": [{
    "statement": (
        "CALL db.index.vector.queryNodes('paper_embedding_idx', 5, $vec) "
        "YIELD node, score RETURN node.title, score"
    ),
    "parameters": {"vec": vec},
}]})
for row in r.json()["results"][0]["data"]:
    print(f"  {row['row'][1]:.3f}  {row['row'][0]}")
```

Expected: the original RAG paper (Lewis et al., 2020) ranks at the top
with a cosine score around 0.92, followed by Self-RAG, Atlas, and other
RAG-family papers that S2 returned for the seed query.

## 4. Deploy the chat app

Prereq: the Gemma 4 vLLM server from the sibling project must be running.

```bash
cd ../../gemma4/gemma4-dgx-devbox
python prefetch_model.py                                  # one-time
GEMMA_PREFETCH_RUN=<run-name> python vllm_server.py       # deploy
cd -
```

Then deploy the chat UI:

```bash
python chat_app.py
```

URL is logged at the end:

```
Graph RAG chat UI deployed: http://graphrag-chat-ui-flytesnacks-development.localhost:30081/
```

### The three retrieval modes

The right-hand panel shows `📄 Retrieved papers` plus, in modes 2 and 3,
`🕸 Graph relations` so the audience can see exactly what graph context
the LLM got beyond raw vector hits. Each paper card has a `via …` source
label so you can tell why it surfaced.

1. **Vector** is pure `db.index.vector.queryNodes`. Baseline. Same shape
   as the `rag-chroma` chat app from the previous week.
2. **Vector + Expand** runs the vector query, then a single 1-hop
   traversal across `CITES`, `AUTHORED_BY`, `IN_CATEGORY` for every seed.
   Adds the neighbor titles into the LLM context as a `GRAPH RELATIONS`
   block.
3. **Hybrid (RRF)** runs the vector query *and* a Cypher pass for
   most-cited papers in the same `Category` as the vector hits, then
   fuses both lists with reciprocal rank. Surfaces papers that are
   authoritative in the topic but whose abstract isn't a great vector
   match.

Each retrieved paper card links to the actual paper (arXiv when
available, Semantic Scholar otherwise), so you can click through during
the demo to show the source.

### Demo prompts

Pick one and flip the mode radio to show what changes. Exact behavior
depends on what S2 returns for your query, but on the default RAG corpus
these consistently land:

- *"What's the relationship between RAG and Self-RAG?"*: mode 2 surfaces
  the explicit `CITES` edge between them when both papers appear in the
  result set.
- *"Compare dense passage retrieval and BM25."*: mode 1 finds DPR via
  abstracts; mode 2 pulls in citation neighbors that contrast the two.
- *"Who are the most influential authors in retrieval-augmented
  generation?"*: mode 3 promotes highly-cited papers via the graph that
  pure vector misses.

## 5. Snapshot / restore (optional)

The Neo4j pod has no persistent volume, so anything you type into the
browser between pipeline runs disappears when the pod cycles. `snapshot.py`
dumps the live graph (nodes, edges, **embeddings**) to a `flyte.io.Dir`
sitting in rustfs, which survives `flyte stop devbox` / `flyte start
devbox`. Restore replays it via HTTP MERGE.

```bash
# Take a snapshot of the current graph. Outputs a Dir (nodes.jsonl + edges.jsonl).
flyte run snapshot.py snapshot_neo4j
# → "Snapshot run: <run-name>"

# Restore that snapshot back into Neo4j. Pass the Dir from the snapshot run.
flyte run snapshot.py restore_neo4j \
    --snapshot=flyte://flytesnacks/development/<snapshot-run-name>/o0

# One-shot smoke test: snapshot, wipe, restore. Exit codes track success.
flyte run snapshot.py snapshot_then_restore
```

Notes worth knowing:

- The snapshot is **online**: pure Cypher over HTTP, no daemon stop. Works
  on Neo4j community edition (which has no online `neo4j-admin database
  dump`).
- Embeddings round-trip exactly. After a restore, querying any
  `Paper.embedding` against the index returns the same paper at score
  `1.000`.
- Snapshot is `wipe_first=True` on restore by default, so the target Neo4j
  ends up matching the snapshot exactly. Set `--wipe_first false` if you
  want to merge a snapshot on top of existing data.

## Known limitations

- **No persistence by default.** Pod restart wipes the graph. Re-run
  the pipeline to rebuild it; both `fetch_papers` (S2) and `embed_papers`
  are cached in rustfs, so re-runs with the same query are fast. For
  hand-edited graph state, take a snapshot first (see step 5).
- **HTTP API, not Bolt.** Functionally equivalent for our scale, but more
  verbose than the Bolt driver. We don't get the nice transaction objects
  or retry helpers; the loader compensates by batching with `UNWIND`.
- **Coarse categories.** S2's `fieldsOfStudy` are broad (Computer Science,
  Linguistics, …), so `IN_CATEGORY` doesn't discriminate as sharply as
  arXiv's `cs.IR` / `cs.CL`. Mode 3 (RRF) still works, just less
  selectively. Layering arXiv categories on top is a separate fetch.

## Next ideas

- **Text-to-Cypher mode.** A 4th chat mode where the LLM writes the
  Cypher itself from the question. Demoable with Gemma 4 + a tight
  prompt that includes the schema.
- **Persistence + larger corpus.** PVC-backed Neo4j and a 5–10k paper
  graph would make the demo feel weightier without changing any of the
  retrieval code.
- **Auto-snapshot on a timer.** Right now `snapshot.py` is on-demand. A
  scheduled Flyte task that snapshots every N minutes (or after each
  pipeline run) would mean any pod-cycle disaster only loses N minutes
  of browser edits.
- **Annotate-style entrypoint hook.** Today the snapshot is a separate
  task you run manually after edits. A shell entrypoint wrapping
  `/startup/docker-entrypoint.sh neo4j` could auto-restore from the
  latest snapshot on boot and best-effort save on SIGTERM. Keeps the
  current `from_dockerfile` path; the trade-off is that Knative's ~30s
  grace before SIGKILL means the shutdown save is best-effort only,
  so you'd still want the periodic timer for real durability.
