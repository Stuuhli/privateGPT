from dataclasses import dataclass
from typing import TYPE_CHECKING

from injector import inject, singleton
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.storage import StorageContext
from llama_index.core.types import TokenGen
from pydantic import BaseModel

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chunks.chunks_service import Chunk
from private_gpt.server.chat.inline_citations import InlineCitationFormatter
from private_gpt.server.chat.rag_workflow import RagWorkflowFactory, RagWorkflowInput
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor


class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None


class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None


@dataclass
class ChatEngineInput:
    system_message: ChatMessage | None = None
    last_message: ChatMessage | None = None
    chat_history: list[ChatMessage] | None = None

    @classmethod
    def from_messages(cls, messages: list[ChatMessage]) -> "ChatEngineInput":
        # Detect if there is a system message, extract the last message and chat history
        system_message = (
            messages[0]
            if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM
            else None
        )
        last_message = (
            messages[-1]
            if len(messages) > 0 and messages[-1].role == MessageRole.USER
            else None
        )
        # Remove from messages list the system message and last message,
        # if they exist. The rest is the chat history.
        if system_message:
            messages.pop(0)
        if last_message:
            messages.pop(-1)
        chat_history = messages if len(messages) > 0 else None

        return cls(
            system_message=system_message,
            last_message=last_message,
            chat_history=chat_history,
        )


@singleton
class ChatService:
    settings: Settings

    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.embedding_component = embedding_component
        self.vector_store_component = vector_store_component
        self.citation_formatter = InlineCitationFormatter()
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store_component.vector_store,
            storage_context=self.storage_context,
            llm=llm_component.llm,
            embed_model=embedding_component.embedding_model,
            show_progress=True,
        )
        self.rag_workflow_factory = RagWorkflowFactory(
            settings=settings,
            llm_component=llm_component,
            vector_store_component=vector_store_component,
            index=self.index,
        )

    def _workflow_input(
        self,
        system_prompt: str | None,
        use_context: bool,
        context_filter: ContextFilter | None,
    ) -> RagWorkflowInput:
        return RagWorkflowInput(
            system_prompt=system_prompt,
            use_context=use_context,
            context_filter=context_filter,
        )

    def _decorate_response(self, text: str, sources: list[Chunk], use_context: bool) -> str:
        if not use_context:
            return text
        return self.citation_formatter.decorate(text, sources)

    def _append_suffix_to_generator(self, generator: TokenGen, suffix: str) -> TokenGen:
        def _generator() -> TokenGen:
            for token in generator:
                yield token
            yield suffix

        return _generator()

    def stream_chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> CompletionGen:
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = (
            chat_engine_input.system_message.content
            if chat_engine_input.system_message
            else None
        )
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )

        workflow_input = self._workflow_input(
            system_prompt=system_prompt,
            use_context=use_context,
            context_filter=context_filter,
        )
        chat_engine = self.rag_workflow_factory.build_chat_engine(workflow_input)
        streaming_response = chat_engine.stream_chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in streaming_response.source_nodes]
        suffix = (
            self.citation_formatter.build_suffix(sources)
            if use_context and sources
            else ""
        )
        response_gen = (
            self._append_suffix_to_generator(streaming_response.response_gen, suffix)
            if suffix
            else streaming_response.response_gen
        )
        completion_gen = CompletionGen(
            response=response_gen,
            sources=sources,
        )
        return completion_gen

    def chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> Completion:
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = (
            chat_engine_input.system_message.content
            if chat_engine_input.system_message
            else None
        )
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )

        workflow_input = self._workflow_input(
            system_prompt=system_prompt,
            use_context=use_context,
            context_filter=context_filter,
        )
        chat_engine = self.rag_workflow_factory.build_chat_engine(workflow_input)
        wrapped_response = chat_engine.chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
        response_text = self._decorate_response(
            wrapped_response.response, sources, use_context
        )
        completion = Completion(response=response_text, sources=sources)
        return completion
