"""Orchestrator — Vietnamese Legal Agentic RAG system."""
from __future__ import annotations

from deepagents import create_deep_agent, SubAgent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from src.agents.ingestion_agent import INGESTION_AGENT_CONFIG
from src.agents.rag_agent import RAG_AGENT_CONFIG
from src.llm import get_llm

ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator of a Vietnamese Legal Agentic RAG system.

## Primary purpose — Answer legal questions

When the user asks a legal question, delegate to the `legal-rag` subagent.
It will classify the query, route to the right retrieval strategy, grade results,
and return a grounded answer with citations.

## Secondary — Build indexes

When the user asks to ingest or index documents, delegate to the `ingestion` subagent.
It will download the dataset, clean HTML, chunk text, and build Chroma + BM25 + graph indexes.
The ingestion only needs to run once (or when the dataset is updated).

## Session memory

Answers and retrieved documents are persisted in the session store so users can ask
follow-up questions within the same session without re-retrieving.

## Guidelines

- Always use write_todos to plan multi-step tasks before delegating.
- When delegating, provide complete and self-contained instructions — subagents are stateless.
- If the indexes are not ready, ask the user to run the ingestion step first.
"""

_store = InMemoryStore()
_checkpointer = MemorySaver()
_backend = CompositeBackend(
    default=StateBackend(),
    routes={"/results/": StoreBackend(store=_store)},
)

def build_agent():
    return create_deep_agent(
        name="vn-legal-rag",
        model=get_llm(),
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        subagents=[SubAgent(**INGESTION_AGENT_CONFIG), SubAgent(**RAG_AGENT_CONFIG)],
        backend=_backend,
        store=_store,
        checkpointer=_checkpointer,
        skills=["./skills/"],
    )


agent = build_agent()
