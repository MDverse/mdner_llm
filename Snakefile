
"""Sakefile to run gliner and LLM-based NER benchmarks."""

include: "workflow/rules/gliner.smk"
include: "workflow/rules/llm.smk"

rule all:
    input:
        rules.gliner_all.input,
        rules.llm_all.input,