# Dynamo Worker Identity in the MP Worker Adapter

Covers `lmcache/integration/vllm/vllm_multi_process_adapter.py`
(`DynamoIdentity`, `_resolve_dynamo_identity`). First step of publishing the
MP server's KV state to NVIDIA Dynamo's KV-aware router: the server will bind
one `KvStateAttachmentOwner` per registered engine worker, and each attachment
needs the worker's Dynamo identity — `namespace`, `component`, `endpoint`,
`worker_id`.

## 1. Where the identity comes from

`namespace`, `component` and `endpoint` are static deployment knobs. `worker_id`
is not: it is the discovery instance id Dynamo mints when its runtime starts,
after argv is parsed, and it is the id the frontend routes requests to. It
therefore cannot be typed into `--kv-transfer-config`; it must be read at
runtime inside the engine process.

Dynamo already exports exactly that id into the engine environment as
`DYN_FPM_WORKER_ID` (for its forward-pass-metrics scheduler) before the engine
starts, and the engine-core and TP worker processes inherit it. The adapter
reads it from there. Env is Dynamo's deliberate channel for runtime-only
identity into the engine (it was moved out of `additional_config` because that
dict is hashed into vLLM's compile cache), so this is precedent, not a hack.

## 2. Resolution and precedence

`_resolve_dynamo_identity(extra_config)` runs once in
`LMCacheMPWorkerAdapter.__init__` and stores the result as `dynamo_identity`:

| field | explicit key (wins) | fallback |
|---|---|---|
| `worker_id` | `lmcache.mp.dynamo.worker_id` | `$DYN_FPM_WORKER_ID` |
| `namespace` | `lmcache.mp.dynamo.namespace` | `$DYN_NAMESPACE`, else `dynamo` |
| `component` | `lmcache.mp.dynamo.component` | `backend` (aggregated mode) |
| `endpoint` | `lmcache.mp.dynamo.endpoint` | `generate` |

No worker id from either source means `dynamo_identity is None`: the engine is
not under Dynamo and the integration is entirely off — every later step keys
off that. A non-integer worker id raises `ValueError` at construction.

The explicit keys exist so a future Dynamo-side change can pass the identity
through `kv_connector_extra_config` (which vLLM does not hash, so it costs no
recompiles) with no adapter change; they are also how prefill-mode deployments
set `component=prefill`, and the only path for Ray-launched TP workers, which
inherit the serialized `VllmConfig` but not the parent's environment.

## 3. Invariant

Without a worker identity the adapter behaves byte-for-byte as before; the only
observable difference with one is the startup log line
`Dynamo identity resolved: DynamoIdentity(...)`. Forwarding the identity to the
server and everything after (attachments, events) are later steps and are
specified in the integration plan, not here.
