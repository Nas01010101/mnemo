"""Optional framework adapters. Nothing here is imported by `tenet` itself —
each submodule imports its target framework lazily, so `pip install git+https://github.com/Nas01010101/tenet.git`
stays free of e.g. `langgraph` unless you actually import that submodule
(`pip install "tenet-memory[langgraph] @ git+https://github.com/Nas01010101/tenet.git"`).
"""
