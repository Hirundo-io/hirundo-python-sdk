.. meta::
   :http-equiv=Content-Security-Policy: default-src 'self'; frame-ancestors 'none'

hirundo.unlearning_llm module
=============================

Client-side model paths
-----------------------

``LocalTransformersModel.local_path`` records the path used by the Hirundo server.
If the SDK client runs on a different filesystem, pass the equivalent client path
when loading an unlearning adapter:

.. code-block:: python

   pipeline = llm.get_hf_pipeline_for_run(
       run_id,
       base_model_path="/path/on/sdk-client/Qwen3-0.6B",
   )

Transformers loads the tokenizer, configuration, and base model from this path.
The SDK still loads the PEFT adapter from the downloaded run archive.

.. automodule:: hirundo.unlearning_llm
   :members:
   :undoc-members:
   :show-inheritance:
