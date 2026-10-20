"""HTTP API for the web frontend (plan P6, minimal): job intake, a one-at-a-time queue that runs the
CLI per job, signed links for direct browser uploads and downloads, and the self-check.

Start it with ``mlredact serve``.  Everything it stores follows the CLI's rules: no document text,
no input filenames, no scope ids, and inputs deleted as soon as a job ends.  The one exception is a
job that asks to keep its original for side-by-side review: the document stays with the job until
``MLREDACT_API_ORIGINAL_RETENTION_DAYS`` have passed, or the job is deleted.
"""
