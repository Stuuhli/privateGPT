from __future__ import annotations

from dataclasses import dataclass

from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    SimilarityPostprocessor,
)

from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.settings.settings import Settings


@dataclass
class RagWorkflowInput:
    """Container for chat-engine creation parameters."""

    system_prompt: str | None
    use_context: bool
    context_filter: ContextFilter | None


class RagWorkflowFactory:
    """Factory that prepares the building blocks for a RAG workflow.

    The goal is to centralize how contextual chat engines are assembled so that the
    project can later plug in LlamaIndex Workflows/Agents such as the
    ``citation_query_engine`` example with minimal code churn.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        index: VectorStoreIndex,
    ) -> None:
        self._settings = settings
        self._llm_component = llm_component
        self._vector_store_component = vector_store_component
        self._index = index

    def build_chat_engine(self, workflow_input: RagWorkflowInput) -> BaseChatEngine:
        if not workflow_input.use_context:
            return SimpleChatEngine.from_defaults(
                system_prompt=workflow_input.system_prompt,
                llm=self._llm_component.llm,
            )

        retriever = self._vector_store_component.get_retriever(
            index=self._index,
            context_filter=workflow_input.context_filter,
            similarity_top_k=self._settings.rag.similarity_top_k,
        )
        node_postprocessors = [
            MetadataReplacementPostProcessor(target_metadata_key="window"),
        ]
        if self._settings.rag.similarity_value:
            node_postprocessors.append(
                SimilarityPostprocessor(
                    similarity_cutoff=self._settings.rag.similarity_value
                )
            )

        if self._settings.rag.rerank.enabled:
            node_postprocessors.append(
                SentenceTransformerRerank(
                    model=self._settings.rag.rerank.model,
                    top_n=self._settings.rag.rerank.top_n,
                )
            )

        return ContextChatEngine.from_defaults(
            system_prompt=workflow_input.system_prompt,
            retriever=retriever,
            llm=self._llm_component.llm,
            node_postprocessors=node_postprocessors,
        )
