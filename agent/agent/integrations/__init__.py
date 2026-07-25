"""
RexGraph integrations - thin bridges to LangChain, HuggingFace, vLLM, TrustGraph,
and Unlimited-OCR.

Each integration is optional. Import only what you need:

    from agent.integrations.langchain_tools import RexConfidenceTool
    from agent.integrations.langgraph_rex import RexStateGraph
    from agent.integrations.huggingface_analyzer import analyze_transformer
    from agent.integrations.vllm_router import RexRouter
    from agent.integrations.trustgraph_adapter import TrustGraphAdapter
    from agent.integrations.unlimited_ocr import UnlimitedOCRClient
"""
