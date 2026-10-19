Release notes
=============

Unreleased
----------

* SDK-146: Custom Hugging Face forget, retain, and additional utility datasets
  now accept optional inline ``token`` or saved ``token_id`` credentials and
  preserve them in unlearning launch requests. Supplying both fails locally.
  Existing dataset configurations without credentials remain supported.
