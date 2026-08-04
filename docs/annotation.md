# Automated Annotation

> **Planning status:** automated annotation providers, API clients, prompt
> templates, review services, and annotation runners are interface plans only.
> FluxMimir does not currently call an external annotation API or promise any
> model/provider integration.

## Design goals

Automated annotation should enrich canonical episodes without making opaque
changes to source data. Machine output is an auditable proposal associated with
a versioned schema, provider provenance, validation result, confidence, cost,
and review state. It is never silently promoted to ground truth.

Annotation profiles should be configuration and schema driven. Candidate
profiles include instruction normalization, temporal subtask segmentation,
instruction-video consistency, ego hand/object interaction descriptions, scene
and progress summaries, atomic next-action labels, and data-quality flags. The
canonical annotation schema remains authoritative; provider-native response
formats must be translated and validated at the boundary.

## Versioned records

A planned `AnnotationRequest` contains only what is needed for one logical
annotation:

- dataset, episode, frame/clip, camera-view, and instruction references;
- immutable content fingerprints and time ranges;
- requested annotation profile and schema name/version;
- prompt-template identifier/version and non-secret generation parameters;
- privacy classification, allowed provider class, and retention policy;
- a deterministic idempotency key.

A planned `AnnotationRecord` should contain:

- annotation and sample identifiers, schema name/version, and structured values;
- evidence references and time spans rather than copied media where practical;
- model confidence/uncertainty and schema-validation diagnostics;
- review status, reviewer decisions, and append-only revisions;
- provider/model/version, prompt hash, request identifier, generation parameters,
  timestamps, latency, token/media usage, and cost metadata;
- provenance links to the request and exact input fingerprints.

Unknown provider fields must not be inserted into canonical records. Raw
provider responses, if an approved deployment elects to retain them, belong in
a restricted artifact store with a retention deadline and a content hash; they
must not appear in normal logs or resolved configuration.

## Planned provider interface

Providers are adapters behind one contract, whether the future implementation
uses a local model or an approved remote multimodal API:

```python
class BaseAnnotationProvider:
    def setup(self, context): ...
    def capabilities(self) -> dict: ...
    def estimate_cost(self, request) -> CostEstimate: ...
    def annotate(self, request) -> ProviderResult: ...
    def annotate_batch(self, requests) -> list[ProviderResult]: ...
    def close(self): ...
```

`ProviderResult` should preserve the provider request ID, structured response,
usage, latency, retry count, and finish/error status. It must not contain the
credential used to obtain it. Capability checks should cover supported media,
context limits, structured-output support, data residency/retention, batching,
and model version. A provider that cannot satisfy the request's privacy or
schema requirements must fail before media upload.

## Annotation and review flow

A future runner should execute these boundaries explicitly:

1. Select eligible samples and validate source media/time alignment.
2. Apply privacy classification, minimization, redaction, and provider policy.
3. Resolve the schema and prompt template and compute the idempotency key.
4. Estimate cost and reserve budget before any billable request.
5. Invoke the provider with bounded retries for transient failures only.
6. Parse structured output and validate types, enums, ranges, time spans, and
   evidence references against the requested schema.
7. Run deterministic quality rules and optional independent model adjudication.
8. Route to auto-accept, human review, rejection, or abstention; write the record
   atomically with provenance and cost.

Recommended review states are `pending`, `auto_accepted`, `needs_review`,
`human_accepted`, `rejected`, and `abstained`. Provider output is append-only.
A human correction creates a revision containing the patch, reason, reviewer
identity/role, and timestamp; it does not overwrite the original proposal.
Thresholds and policies are versioned configuration. Low confidence, schema
repair, provider disagreement, safety-sensitive labels, or weak evidence should
route to review instead of being coerced into a label.

The Qwen-RobotManip curation recipe motivates a useful optional pattern:
temporally normalize long episodes into subtask clips, request a structured
analysis of manipulated objects, action semantics, temporal order, and
agent-environment interaction, then adjudicate ambiguous/non-aligned cases with
independent models. Cross-model voting records each vote and disagreement; it is
not a substitute for schema validation or human review.

Two optional research profiles from the report may be represented without
hard-coding them into a provider:

- Ego clips can be configured to span 1.5--3 seconds with four uniformly sampled
  frames, producing hand/arm motion, hand-object spatial interaction, and object
  state-change descriptions while abstaining on static clips.
- An embodied chain-of-thought profile can produce three structured fields:
  `scene_description`, `task_progress_assessment`, and `next_atomic_action`.
  Annotation-time history summaries, a future-frame preview, and coarse progress
  may be privileged evidence, but must be marked as such. Training examples must
  not accidentally include privileged future context, and generated text should
  be grounded in the intended current observations and instruction.

## Idempotency, retries, and concurrency

The idempotency key should hash a canonical representation of input content
fingerprints, relevant time spans/views, annotation profile and schema version,
prompt-template version, provider/model version, and generation parameters. It
must exclude credentials, transient signed URLs, timestamps of submission, and
provider request IDs.

The same key returns the existing terminal record instead of issuing another
billable call. Retries for timeouts, rate limits, and retryable server failures
remain one logical annotation and retain the same key. Invalid structured output
is a validation result, not an excuse for an unbounded retry loop. Persistent
storage needs an atomic claim/lease around each key so parallel workers cannot
submit duplicates; expired leases are recoverable and recorded. Partial batch
results are committed per request.

## Cost controls

Provider pricing must not be embedded as an timeless constant. Record the price
table/provider quote version and currency alongside measured prompt, output,
image, video, or compute usage. Where the provider returns authoritative billed
usage/cost, retain it separately from the preflight estimate.

The runner should support per-sample, per-run, per-provider, and daily budget
limits; dry-run cost estimation; warnings at configurable utilization levels;
and fail-closed behavior when a hard budget is exhausted. Cache hits and
idempotent replays cost zero new calls. Retries and review/adjudication calls are
separate cost line items linked to the same annotation lineage.

## Privacy and data minimization

Ego video can expose faces, voices, screens, homes, workplaces, bystanders,
precise locations, or other sensitive context. Before any remote request, a
future implementation must:

- verify collection rights, consent, dataset license, purpose, and allowed
  provider/data region;
- classify data (for example public, internal, restricted) and disallow remote
  providers that do not meet the class policy;
- upload only required frames/crops and metadata, strip EXIF/location and
  unrelated audio, and apply configured face/screen/identifier redaction;
- prefer references and derived crops over copying full episodes;
- enforce transport security, provider retention/training opt-out policy, and a
  local deletion schedule;
- keep raw media and provider responses out of logs, exception messages, test
  fixtures, and resolved configuration.

Provider credentials are supplied only through process environment variables
and are handled according to [security.md](security.md). No credential or secret
value is part of an annotation request, record, provenance manifest, cache key,
or cost report.

