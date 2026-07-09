"""Optional framework adapters. Nothing here is imported by `tenet` itself —
each submodule imports its target framework lazily, so `pip install tenet-memory`
stays free of e.g. `langgraph` unless you actually import that submodule
(`pip install tenet-memory[langgraph]`).
"""
