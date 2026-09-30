# Sakefile to run gliner/LLM-based NER benchmarks and the mdverse annotation pipeline.

include: "workflow/rules/benchmark_gliner.smk"
include: "workflow/rules/benchmark_llm.smk"
include: "workflow/rules/mdverse_annotation.smk"

rule all:
    input:
        rules.run_benchmark_gliner.input,
        rules.run_benchmark_llm.input,
        rules.run_mdverse_annotation.input