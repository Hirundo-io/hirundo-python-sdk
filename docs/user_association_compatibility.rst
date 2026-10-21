User-association route compatibility
====================================

The Python SDK needs no code change for the single-association route fix in
`platform PR #588 <https://github.com/Hirundo-io/hirundo-platform/pull/588>`_.
It exposes no user-association read or list method and never calls either route.
This audit addresses
`SDK-148 <https://linear.app/hirundo/issue/SDK-148/check-python-sdk-compatibility-with-the-user-association-read-route>`_.

Scope checked
-------------

On 2026-10-21, the audit checked SDK main at ``08ece552`` and all 44 repository
tags, including the current ``v0.3.0``, ``v0.3.1``, ``v0.3.2``, and ``v0.3.3``
tags and the older on-premises tags. These are source inspections, not live
deployment or installed-release tests.

* Handwritten methods and request builders: no user-association operations in
  ``hirundo/``. Organization filters on datasets, storage configs, models, and
  runs target their own resource routes. They do not fetch user associations.
* Authentication and CLI: ``_env.py`` loads a configured API key, ``_headers.py``
  builds bearer-auth and API-version headers, and ``cli.py`` saves configuration.
  None discovers organizations through user-association requests.
* Generated methods and schemas: no OpenAPI/Swagger client, schema snapshot, or
  client-generation configuration in the repository. The SDK does not load the
  server's OpenAPI schema at runtime.
* Tests, documentation, and examples: no existing association read/list calls or
  association response models. There is no SDK method whose return type needs
  to change from a list to a single association.

The release-source search can be repeated from the repository root:

.. code-block:: bash

   git fetch origin main --tags
   for revision in origin/main $(git tag); do
       git grep -n -i -E \
           'user[-_ /]?association|UserAssociation|openapi|swagger|generated.client' \
           "$revision" -- hirundo tests docs source README.md DEV_README.md pyproject.toml
   done

The search returned no matches at the audited revisions. Later revisions include
this audit document and will therefore match it.

Server contract and migration
-----------------------------

The platform diff at ``d90771df`` changes the single-read registration from
``/`` to ``/{user_id}`` in ``server/api/user_association.py``:

* Read one association with ``GET /user-association/{user_id}``. The optional
  ``organization_id`` remains a query parameter. The response is one association
  object; omitting the organization uses the caller's default organization.
* List associations with ``GET /user-association/``. The response remains a JSON
  list. Passing ``user_id`` in its query does not turn it into a single read.

Direct HTTP consumers must move ``user_id`` into the path and keep any explicit
``organization_id`` in the query. Clients generated independently from platform
OpenAPI must regenerate against the fixed contract. The Python SDK has no such
generated code to update.

Before the platform fix, the list handler shadows the old single-read handler.
An older server therefore cannot provide this path-based single-read operation.
A direct or future generated client that uses it requires a server containing
the fix. Do not fall back to the old query URL and deserialize its list as one
association. PR #588 was open at audit time, so no released minimum server
version can be named from this evidence.

Version constraints
-------------------

SDK tags ``v0.3.0`` through ``v0.3.3`` and audited main send
``HIRUNDO-API-VERSION: 0.3``. Platform main at ``ea490347`` and PR #588 use
``CURRENT_HIRUNDO_API_VERSION = "0.3"``. The middleware in
``server/http_entrypoint.py`` rejects an explicit mismatched API version with
HTTP 400. The route fix does not change this mechanism.

There is no published SDK/server release support matrix in the SDK repository.
For this route change, released SDKs remain unaffected against the updated
server, and the SDK with this documentation update remains unaffected against
older servers using API version ``0.3``. No additional SDK pin, server minimum,
version-header change, or SDK release is required. This finding does not promise
that every SDK feature works on every server with the same API-version header,
or extend support to historical releases using other API versions.

Verification
------------

There is no affected SDK operation to exercise with an association fixture.
Adding an association client solely for this check would introduce new behavior.
Existing mocked HTTP tests check prepared requests, organization filters,
API-version headers, and single-object versus list response parsing on the
resources the SDK does expose:

.. code-block:: bash

   source .venv/bin/activate
   pytest -q tests/unlearning-llm/llm_unlearning_http_test.py \
       tests/test_llm_behavior_eval.py tests/test_external_eval.py

On Python 3.10.22, these tests passed: 75 passed. ``ruff check .``,
``ruff format --check .``, and ``basedpyright`` also passed. These are local
checks; the PR's CI results provide separate evidence for the committed change.

Platform PR #588 adds the association-specific HTTP regression tests in
``server/api/user_association_endpoints_test.py``. Source review confirmed tests
for a single object with and without an explicit organization, a list response,
forbidden reads, missing records, and invalid user IDs. Those database-backed
platform tests were not run in the SDK environment.

The full SDK ``pytest`` command cannot collect nine integration modules without
AWS, GCP, and Hugging Face credentials. Live cross-version server testing was not
performed.
