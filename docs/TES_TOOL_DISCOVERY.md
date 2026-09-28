# TES Tool Discovery consumer contract

Dev Hub keeps documentation search and tool discovery as separate contracts.

```text
Documentation mode: repository Markdown → FAISS → RAG answers/citations
Tools mode:        TES Discovery API → structured tool results
```

Dev Hub does not parse TES YAML, load the TES generated catalog directly, or
place TES tool records in the FAISS documentation index. TES remains the
canonical tool-definition, discovery-catalog, schema-validation, and execution
authority.

## Configuration

Set `TES_DISCOVERY_URL` in the Dev Hub process environment to the approved TES
Discovery API base URL. `TES_DISCOVERY_TIMEOUT_SECONDS` defaults to 5 seconds
and is bounded by the adapter. No TES credentials are stored in Dev Hub; the
existing Dev Hub authentication dependency and browser auth-header behavior
remain in force.

The adapter is read-only and supports TES lookup, deterministic structured
search, data-type/category facets, and catalog version/status endpoints. It
preserves the canonical TES `tool_id` and normalized metadata.

## Failure isolation

If TES is unavailable, its catalog is unavailable, a request times out, or the
response is malformed, Tools mode shows a structured discovery-unavailable
state. It does not fall back to documentation similarity or claim input
compatibility from RAG. Documentation mode remains independently available.

## Status and execution boundary

Tool cards display `CONFIGURED`, `SERVING`, `REGISTERED`, `TESTED`, and
`OPERATIONALLY_VERIFIED` separately. Unknown/null runtime evidence is shown as
Unknown rather than false or executable. Server compatibility relationships
remain evidence-bearing static/dynamic data from TES.

This phase has no execution button and submits no TES jobs. A later execution
flow must pass the selected canonical `tool_id` back to TES for canonical
validation and delegated execution; Dev Hub is not an execution authority.
