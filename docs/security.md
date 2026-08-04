# Security and Secret Handling

> **Planning status:** these are mandatory design requirements for future
> FluxMimir backends and API integrations. No backend/API integration or secret
> manager is currently implemented, and this document must not be read as a
> claim that the controls are already enforced in code.

## Non-negotiable rules

1. Credentials enter a FluxMimir process only as environment variables injected
   by the caller, scheduler, or approved secret manager.
2. Secret values must never be committed to the repository, written to dataset
   artifacts, embedded in Python/MMEngine/YAML/JSON configuration, supplied as
   command-line arguments, or stored in shell scripts or `.env` files.
3. Secret values must never appear in resolved configuration, manifests, cache
   keys, logs, progress files, metrics, tracebacks, exceptions, telemetry, or
   provider request/response dumps.
4. A config may name the environment variable or credential alias to use, but
   resolution occurs at the last possible moment inside the provider boundary.
   The resolved value is not written back into the config object.
5. A missing credential fails closed with the name of the missing variable, not
   its value. There are no built-in development credentials or fallback keys.

Use descriptive, provider-scoped environment variable names such as
`FLUXMIMIR_<PROVIDER>_API_KEY`. Documentation and tests use placeholders only.
The value must not be echoed during installation, dry-run, health checks, or
exception handling.

## Sanitized configuration and provenance

FluxMimir should preserve reproducibility through a sanitized resolved config.
Credential references may be represented as an environment-variable name or a
placeholder such as `${ENV:VARIABLE_NAME}`. The secret value must never be
interpolated. Sanitization also applies to nested provider options, HTTP headers,
cookies, DSNs, query strings, and signed object-store URLs.

Provenance may safely retain provider/model identity, code and checkpoint
digests, prompt-template hash, non-secret parameters, input/output fingerprints,
provider request IDs, timings, retry counts, usage, costs, and outcome. It must
not retain authorization headers, bearer tokens, session cookies, client
secrets, webhook secrets, private keys, raw credential-bearing requests, or
short-lived signed URLs.

## Logging and error handling

Logging is allowlist based: emit only fields known to be safe. A generic
recursive key-name redactor is defense in depth, not the primary policy. At a
minimum it should recognize secret/token/key/password/authorization/cookie/DSN
fields, redact authorization-like header values, and remove sensitive URL query
parameters. Values known by the credential resolver should be registered with a
runtime value redactor without printing them.

Do not log whole configs, environment mappings, HTTP sessions, client objects,
request headers/bodies, provider responses, subprocess environments, or object
`repr` output. Exception wrappers should retain an internal reason code,
sanitized provider status, and request ID while discarding secret-bearing
payloads. HTTP debug logging and SDK wire traces are off by default and may not
be enabled in production runs.

Dry-run must validate registry/config shape without resolving credentials,
opening network connections, or loading private media. A provider health check
may report available/unavailable and capabilities, but not credential prefixes,
lengths, hashes, or account-private details.

## Least privilege and process boundaries

Each provider uses a separate credential scoped to the minimum API, project,
dataset, storage prefix, and lifetime required. Inputs should be mounted
read-only; outputs should be limited to a run-specific directory. Workers receive
only the credential needed for their backend, not the entire parent environment.
Network egress should be restricted to approved provider and artifact endpoints.

Local research tools and model checkpoints are also trust boundaries. Pin code
revisions and dependencies, verify available checksums/signatures, document
licenses, and isolate subprocesses with explicit arguments and a minimal
environment. Do not execute code fetched implicitly from a data file or model
repository.

## Data protection

Canonical datasets, ego videos, annotations, inpainted frames, and provider
responses can be sensitive even when they contain no API key. Apply dataset
classification and access policy independently of credential redaction:

- minimize uploaded media and metadata and use approved TLS endpoints;
- honor dataset licenses, consent, geography, provider retention, and
  provider-training restrictions;
- redact faces, screens, identifiers, location metadata, and unrelated audio
  when policy requires it;
- encrypt restricted artifacts at rest using infrastructure-managed controls;
- define retention and deletion periods for temporary media, remote jobs, raw
  responses, caches, and failed-run diagnostics;
- avoid public or cross-user cache keys/locations for restricted inputs.

Restricted data must not be sent to a remote backend merely because a valid API
credential is present. Authorization to call a provider and authorization to
share a particular sample are separate checks.

## Rotation and incident response

Credentials need documented owners, expiry/rotation schedules, and revocation
paths. If a credential may have appeared in source, chat, logs, process listings,
artifacts, or provider payloads, treat it as compromised:

1. stop affected jobs and revoke/rotate the credential;
2. audit provider access and billing using sanitized request IDs and timestamps;
3. locate and restrict or remove leaked copies from configs, logs, artifacts,
   caches, CI output, and history according to incident policy;
4. issue replacement credentials with minimum scope;
5. document impact and add a regression test or scanner rule without preserving
   the leaked value.

Never validate a suspected leaked key by making an unnecessary API call.

## Tests and release gates

Tests use fake values and synthetic media. CI should combine secret scanning with
behavioral tests that assert:

- resolved configs and manifests contain only credential references;
- logs and exceptions remain redacted for success, retry, and failure paths;
- cache/idempotency keys are independent of credentials and signed URLs;
- dry-run neither resolves a credential nor performs network I/O;
- subprocesses and workers receive only allowlisted environment variables;
- raw provider payloads and restricted media cannot enter normal artifacts.

Before any planned backend or API provider is declared supported, its threat
model, retention behavior, authentication path, redaction tests, privacy gates,
and incident owner must be documented and reviewed.
