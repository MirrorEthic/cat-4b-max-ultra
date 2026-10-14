# Causal research graph

Why this project's architecture is what it is, with receipts. One TOML file per node under
`crg/nodes/`. Hologram compiles the graph into what every prompt sees (`hologram hook`), refuses
tool calls that break its invariants, and checks it (`hologram crg check`).

The problem it answers: an agent meets a design or a result without knowing what it replaced,
what failed before it, or whether it is still current, and rebuilds something already closed.

## A node

```toml
id = "proj.tiers.rank-based-working-set"
type = "mechanism"            # mechanism | invariant | result | question
status = "CURRENT"
created = "2026-10-06"
claim = "One sentence: what is asserted."
why = "One line: why it has this status."          # required for dead nodes
adjudicated_by = "your name 2026-10-06 (what you approved)"
supersedes = ["proj.tiers.absolute-thresholds"]
evidence = ["file:WORKLOG.md#2026-10-06", "commit:1a02a82", "test:tests/test_tiers.py"]
verify = ["test:tests/test_tiers.py"]
```

Edges, written on the node that asserts them: `supports`, `falsifies`, `corrects`, `supersedes`,
`depends_on`, `implements`, `requires` (holds only together with), `limits`, `replicates`,
`forked_from`, `blocks`.

Receipts: `file:PATH@COMMIT#anchor`, `test:PATH`, `commit:SHA`. `hologram crg pin` pins an
unpinned file receipt to HEAD so later edits cannot change the evidence.

## Statuses

| Status | Meaning | The check requires |
|---|---|---|
| CURRENT | the architecture now | a supporting result (mechanisms); nothing falsifying or superseding it |
| VALIDATED | established, deployed or not | a supporting result (mechanisms) |
| OPEN | unresolved | |
| EXPERIMENTAL | under test | |
| SUPERSEDED | was valid, replaced | a node that supersedes it, and a why |
| FALSIFIED | closed by a result | a result that falsifies it, and a why |
| RETIRED | abandoned on purpose, not proven wrong | a why |

Nothing is deleted. A node that turned out wrong gets a new status and a result that says why,
so a dead branch stays visible and is not rebuilt.

## Guards

An invariant may carry guards: tool calls Hologram refuses before they run.

```toml
[[guard]]
kind = "path"                 # path | command | graph-check
tools = ["Edit", "Write", "NotebookEdit"]
is = "crg/CURRENT.md"         # or: under = "some/directory"
on_violation = "deny"         # deny | ask (ask: waivable for a session with /guard allow)
fix = "Edit the nodes under crg/nodes, then run: hologram crg current --write"
```

A `command` guard lists regexes in `all`, every one of which a shell command must match; a
`graph-check` guard runs `hologram crg check` when a command matches `matches` and refuses it
while the graph fails.

## Building the graph

A session writes nodes; a person adopts them. `hologram crg init` makes `crg/drafts/`
(kept out of Hologram's routing corpus) beside `crg/nodes/`:

```
hologram crg bootstrap                  # a packet for the session: the form, the rules, the project's records
                                        # (worklogs, changelogs, decisions, results) to draft the first nodes from
hologram crg check --nodes crg/drafts   # check the drafts
hologram crg adopt <id> ... --by NAME   # move the drafts you stand behind into crg/nodes, stamped with your name
```

Every session start then says the same thing to the model: when this session settles,
falsifies or opens something, write it as a draft node; a person adopts it. Nothing is
CURRENT without that.

## Commands

```
hologram crg check                      # validate; non-zero exit on errors
hologram crg check --nodes crg/drafts   # validate drafts against the project's receipts
hologram crg adopt <id>... --by NAME    # drafts -> nodes, adjudicated
hologram crg bootstrap                  # the packet to draft a first graph from the project's records
hologram crg pin                        # pin unpinned file receipts to HEAD
hologram crg verify                     # run the verifiers, write crg/VERIFY.json
hologram crg current --write            # generate crg/CURRENT.md and crg/GUARDS.json
hologram crg brief                      # what a session start shows
hologram crg line --topic "..."         # what a prompt shows
hologram crg packet --topic "..."       # a bounded grounding packet for an agent
```

`crg/CURRENT.md` and `crg/GUARDS.json` are generated. Edit the nodes, then regenerate.

## This project

Verifiers import catforge, which needs `transformers`; system python3 lacks it and every such node shows CONTESTED.
Run them in the project venv (the graph module itself is stdlib-only):

```
PYTHONPATH=~/hologram-pro .venv/bin/python -m hologram.crg verify --root .
```

Run logs and state live on `/media/garret-sutherland/dev-data/cat4b/`, outside the repo; a result that rests on them
copies the lines into `crg/evidence/` and cites that file.
