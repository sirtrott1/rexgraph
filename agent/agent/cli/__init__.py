"""
agent.cli - command-line entry points.

Console scripts (declared in pyproject.toml [project.scripts]):

    rexgraph-ocr     status | serve | stop | run | setup   (OCR + GPU server)
    rexgraph-run     run the full document->analysis pipeline
    rexgraph-serve   start | stop | status                 (GPU server)
    rexgraph-setup   install OCR / GPU dependencies
    rexgraph-config  show | platform                       (inspect config)
    rexgraph-auth    token / cert management
    rexgraph-test    run the agent test suite
    rcf-server       launch the FastAPI web app (uvicorn)
"""
