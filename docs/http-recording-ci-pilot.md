# HTTP recording CI pilot

The sanity workflow separates four test selections:

- `unit` starts immediately on every supported OS and Python version.
- `local_transport` runs in each matrix environment with loopback as its only
  network dependency.
- `authenticated_probe` makes one unrecorded HTTPS request in each environment on
  protected `main` pushes and explicit workflow dispatches.
- `recorded_integration` runs live once on Ubuntu and Python 3.10 on those trusted
  events. The other environments replay only the sanitized artifact from the same
  commit, workflow run, and attempt.

Pull-request and merge-group revisions run only the non-secret unit, local transport,
and package jobs. GitHub does not expose API or cloud credentials to code controlled
by a pull request. Fresh deployed-server compatibility evidence is therefore produced
only after protected `main` accepts the revision or a maintainer starts a trusted
workflow dispatch.

The tag-triggered full workflow still owns slow opt-in dataset QA and ML runs,
including actual dataset loading through GCP, AWS S3, and authenticated Hugging Face
storage. Those workload launches are deliberately outside this pilot.

## What the recording proves

The first pilot targets the designated SaaS test deployment. Its manifest stores
the endpoint label, schema digest, recording window, SDK commit, workflow run and
attempt, tool versions, test selection, and cassette checksums. The platform
revision is `unknown` until a deployment workflow supplies independently verified
image or commit metadata.

A successful run means the SDK worked against the server reached during that
recording window. It does not prove which platform commit served the request, that
the deployment stayed unchanged throughout the recording, or that an unmerged
platform pull request is compatible. An OpenAPI digest identifies the fetched
schema, not the deployed implementation.

The bounded recording suite covers Git repository CRUD; Git, GCP, AWS S3, and
authenticated Hugging Face storage-backed dataset metadata CRUD; Dataset QA run
listing; LLM model CRUD and unlearning run listing; and LLM behavior evaluation run
listing. The live recording therefore checks that the backend accepts each supported
storage contract, while the tag-triggered full tests check that workers can load the
datasets. List responses retain only records containing the recording suite's
test-owned name prefix. The sanitizer removes unrelated organization records and
provider credentials before upload; identifiers alone are not trusted because they
can collide across resource namespaces.

Run launches, status streams, and result downloads remain in the opt-in full-backend
suite because they start Dataset QA, model unlearning, or evaluation work. Required
CI does not start inference or training. Local fixtures cover the SSE and ZIP
transport behavior without depending on a deployed workload.

Before recording, CI compares the canonical deployed OpenAPI document byte-for-byte
with `schemas/openapi_snapshot.json`. A schema mismatch stops the job, so generated
model validation and cassette provenance cannot refer to different contracts.

## Replay and transport limits

VCR.py buffers recorded bodies. Replay checks HTTP method, path and ordered query,
semantic JSON, authentication presence and scheme, and meaningful content and API
version headers. It does not exercise TLS negotiation, socket timing, incremental
SSE chunk delivery, disconnects, read timeouts, or retry delays. The
`local_transport` tests cover those transport behaviors with a bounded localhost
server. Binary and compressed response bodies are rejected by the cassette
sanitizer; the ZIP path stays in the localhost suite.

## Measurement status

The available parallelism pull-request run, GitHub Actions run `34338855427`,
completed its 15-job matrix in 30 seconds but intentionally skipped installation
and tests on pull requests. It made no useful test-duration or backend-request
baseline, so it cannot support a before/after claim.

Each recording manifest reports cassette interaction count, artifact size, and
recording duration. GitHub Actions supplies job and workflow duration and failure
history. After equivalent main or merge-queue runs exist on both sides of the
change, record these values before expanding the pilot:

| Measure | Before | Pilot | Current limitation |
| --- | ---: | ---: | --- |
| Critical-path duration | unavailable | pending CI | baseline run skipped tests |
| Live backend requests | unavailable | manifest interaction count | no prior request telemetry |
| 429/retry count | unavailable | pending CI | not emitted by the old workflow |
| Flake rate | unavailable | pending repeated runs | one run is not a rate |
| Sanitized artifact size | none | manifest artifact bytes | produced only in CI |

Do not claim a duration, traffic, or flake-rate improvement until comparable CI
runs populate this table or an external report derived from the same test selection
and target environment.
