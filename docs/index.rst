.. hirundo documentation master file, created by
   sphinx-quickstart on Sun Jul 21 10:18:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :http-equiv=Content-Security-Policy: default-src 'self', frame-ancestors 'none'

hirundo documentation
=====================

Welcome to the ``hirundo`` client library documentation.

This package is used to interface with Hirundo's platform.

Currently the only supported feature is a simple API to optimize your ML datasets.

Optimizing a dataset
--------------------

You do not need to share any of your code to optimize your dataset.

Currently ``hirundo`` supports the following dataset types:
   - (Multi-class) Classification datasets
   - Object Detection datasets

Support dataset storage configs include:
   - Google Cloud (GCP) Storage
   - Amazon Web Services (AWS) S3
   - Git LFS (Large File Storage) repositories (e.g. GitHub or HuggingFace)

Optimizing a classification dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently ``hirundo`` requires a CSV file with the following columns (all columns are required):
   - ``image_path``: The location of the image within the dataset ``data_root_url``
   - ``class_name``: The semantic label, i.e. the class name of the class that the image was annotated as belonging to

And outputs two Pandas DataFrames with the dataset columns as well as:

Suspect DataFrame (filename: `mislabel_suspects.csv`) columns:
   - ``suspect_score``: mislabel suspect score
   - ``suspect_level``: mislabel suspect level
   - ``suspect_rank``: mislabel suspect ranking
   - ``suggested_class_name``: suggested semantic label
   - ``suggested_class_conf``: suggested semantic label confidence

Errors and warnings DataFrame (filename: `invalid_data.csv`) columns:
   - ``status``: status message (one of ``NO_LABELS`` / ``MISSING_IMAGE`` / ``INVALID_IMAGE``)

Optimizing an object detection (OD) dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently ``hirundo`` requires a CSV file with the following columns (all columns are required):
   - ``image_path``: The location of the image within the dataset ``data_root_url``
   - ``object_id``: The ID of the bounding box within the dataset. Used to indicate object suspects
   - ``class_name``: Object semantic label, i.e. the class name of the object that was annotated
   - ``xmin``: leftmost horizontal pixel coordinate of the object's bounding box
   - ``ymin``: uppermost vertical pixel coordinate of the object's bounding box
   - ``xmax``: rightmost horizontal pixel coordinate of the object's bounding box
   - ``ymax``: lowermost vertical pixel coordinate of the object's bounding box

And outputs two Pandas DataFrames with the dataset columns as well as:

Suspect DataFrame (filename: `mislabel_suspects.csv`) columns:
   - ``suspect_score``: object mislabel suspect score
   - ``suspect_level``: object mislabel suspect level
   - ``suspect_rank``: object mislabel suspect ranking
   - ``suggested_class_name``: suggested object semantic label
   - ``suggested_class_conf``: suggested object semantic label confidence

Errors and warnings DataFrame (filename: `invalid_data.csv`) columns:
   - ``status``: status message (one of ``NO_LABELS`` / ``MISSING_IMAGE`` / ``INVALID_IMAGE`` / ``INVALID_BBOX`` / ``INVALID_BBOX_SIZE``)

Google Colab notebooks with more examples
-----------------------------------------
You can find more examples of how to use ``hirundo`` in
`Google Colab notebooks <https://github.com/Hirundo-io/hirundo-python-sdk/tree/main/notebooks>`_.

Package contents
----------------
.. toctree::
   :maxdepth: 4

   modules
