# HTTP recording CI pilot

The sanity workflow separates four test selections:

- `unit` starts immediately on every supported OS and Python version.
- `local_transport` runs in each matrix environment with loopback as its only
  network dependency.
- `authenticated_probe` makes one unrecorded HTTPS request in each environment.
- `recorded_integration` runs live once on Ubuntu and Python 3.10. The other
  environments replay only the sanitized artifact from the same commit, workflow
  run, and attempt.

The tag-triggered full workflow still owns slow opt-in dataset QA and ML runs.
Those tests are deliberately outside this pilot.

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

The recorded lifecycle covers Git repository creation, listing, and deletion. It
does not call `GitRepo.get_by_id`: the designated deployment currently requires a
`git_repo_organization_id` query parameter that this SDK method cannot supply. That
contract drift remains visible as a pilot limitation instead of being encoded into
a supposedly portable cassette.

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
