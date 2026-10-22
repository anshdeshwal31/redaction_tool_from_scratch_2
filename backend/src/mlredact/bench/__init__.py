"""Benchmark adapter (``mlredact bench``): for the external evaluator, never production.

It runs the normal pipeline with a trace and writes what the benchmark's adapter contract asks for:
the output PDF (also when the tool does not release it) and the stage files, which hold document text.
"""
