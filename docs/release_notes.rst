Release notes
=============

Unreleased
----------

* SDK-146: Custom Hugging Face forget, retain, and additional utility datasets
  now accept optional inline ``token`` or saved ``token_id`` credentials and
  preserve them in unlearning launch requests. Supplying both fails locally.
  Existing dataset configurations without credentials remain supported.

* SDK-147: A single CUSTOM behavior now accepts independent nonempty forget and
  retain dataset lists, including mixed single/list inputs, HF/CSV sources, and
  per-dataset credentials from SDK-146. Existing single-object payloads and run
  responses remain compatible. Empty lists and invalid members fail locally.
  The editable Security-source example includes additive utility configuration.
  List requests require a platform deployment containing HPLT-377; confirm
  deployment support before use. No released platform version is identified yet,
  and unsupported-server errors propagate without fallback training runs.
