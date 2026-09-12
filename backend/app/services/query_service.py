import time

import structlog

from backend.app.models import (
    Citation,
    QueryRequest,
    QueryResponse,
    ResearchRequest,
    ResearchResponse,
    ResearchStep,
)
from backend.app.services.chroma_service import chroma_service
from backend.app.services.ollama_service import ollama_service
from config.settings import settings

logger = structlog.get_logger(__name__)


class QueryService:
    def __init__(self) -> None:
        self._top_k = settings.top_k

    def _build_rag_prompt(self, question: str, context: str, include_citations: bool = True) -> str:
        citation_instruction = (
            "\n\nWhen answering, cite your sources using [doc_id:chunk_index] format."
            if include_citations
            else ""
        )
        return (
            "You are a helpful research assistant. Use the following context to answer "
            "the question.\n"
            "If the context doesn't contain enough information, say so honestly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}{citation_instruction}\n\n"
            "Answer:"
        )

    def _build_summary_prompt(self, text: str, max_length: int) -> str:
        return f"""Summarize the following text in approximately {max_length} characters or less.
Focus on key points and main ideas.

Text:
{text}

Summary:"""

    def _build_research_prompt(self, topic: str, previous_steps: list[ResearchStep]) -> str:
        steps_summary = "\n".join(
            [
                f"Step {s.step}: {s.action} - {s.query}\nResult: {s.result[:500]}"
                for s in previous_steps
            ]
        )
        return (
            f'You are conducting multi-step research on the topic: "{topic}"\n\n'
            f"Previous research steps:\n{steps_summary}\n\n"
            "Based on the above, determine the next research step. Respond with:\n"
            "1. A specific question or action to investigate next\n"
            "2. Why this step is needed\n\n"
            'If you have enough information to provide a final answer, respond with "FINAL ANSWER" '
            "followed by your comprehensive answer."
        )

    async def query(self, request: QueryRequest) -> QueryResponse:
        start_time = time.time()
        try:
            results = await chroma_service.query(
                query_texts=[request.question],
                n_results=request.top_k or self._top_k,
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            if not documents:
                return QueryResponse(
                    answer="I couldn't find any relevant documents to answer your question.",
                    citations=[],
                    model_used=settings.ollama_model,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

            context_parts = []
            citations = []
            for i, (doc, meta, dist) in enumerate(
                zip(documents, metadatas, distances, strict=False)
            ):
                doc_id = meta.get("document_id", "unknown")
                chunk_idx = meta.get("chunk_index", i)
                context_parts.append(f"[Source {i + 1}: {doc_id}:{chunk_idx}]\n{doc}")
                citations.append(
                    Citation(
                        document_id=doc_id,
                        chunk_index=chunk_idx,
                        content=doc[:200] + "..." if len(doc) > 200 else doc,
                        score=1.0 - dist if dist else 0.0,
                    )
                )

            context = "\n\n".join(context_parts)
            prompt = self._build_rag_prompt(request.question, context, request.include_citations)

            answer = await ollama_service.generate(
                prompt=prompt,
                temperature=0.3,
            )

            processing_time = int((time.time() - start_time) * 1000)
            return QueryResponse(
                answer=answer.strip(),
                citations=citations if request.include_citations else [],
                model_used=settings.ollama_model,
                processing_time_ms=processing_time,
            )
        except Exception as e:
            logger.error("Query failed", error=str(e))
            raise

    async def summarize(self, document_ids: list[str], max_length: int) -> str:
        try:
            all_chunks = []
            for doc_id in document_ids:
                results = await chroma_service.query(
                    query_texts=[""],
                    n_results=1000,
                    where={"document_id": doc_id},
                )
                documents = results.get("documents", [[]])[0]
                all_chunks.extend(documents)

            if not all_chunks:
                return "No content found for the specified documents."

            combined_text = "\n\n".join(all_chunks)
            if len(combined_text) > 8000:
                combined_text = combined_text[:8000] + "..."

            prompt = self._build_summary_prompt(combined_text, max_length)
            summary = await ollama_service.generate(
                prompt=prompt,
                temperature=0.3,
            )
            return summary.strip()
        except Exception as e:
            logger.error("Summarization failed", error=str(e))
            raise

    async def research(self, request: ResearchRequest) -> ResearchResponse:
        start_time = time.time()
        steps: list[ResearchStep] = []
        all_citations: list[Citation] = []

        try:
            max_steps = request.max_steps or 5
            for step_num in range(1, max_steps + 1):
                previous_steps = steps[-3:] if steps else []
                prompt = self._build_research_prompt(request.topic, previous_steps)
                response = await ollama_service.generate(prompt=prompt, temperature=0.5)

                if response.strip().startswith("FINAL ANSWER"):
                    final_answer = response.replace("FINAL ANSWER", "").strip()
                    break

                action = "research"
                query = response.strip()

                results = await chroma_service.query(
                    query_texts=[query],
                    n_results=request.max_sources_per_step or self._top_k,
                )

                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]

                step_citations = []
                context_parts = []
                for i, (doc, meta, dist) in enumerate(
                    zip(documents, metadatas, distances, strict=False)
                ):
                    doc_id = meta.get("document_id", "unknown")
                    chunk_idx = meta.get("chunk_index", i)
                    context_parts.append(f"[Source {i + 1}: {doc_id}:{chunk_idx}]\n{doc}")
                    citation = Citation(
                        document_id=doc_id,
                        chunk_index=chunk_idx,
                        content=doc[:200] + "..." if len(doc) > 200 else doc,
                        score=1.0 - dist if dist else 0.0,
                    )
                    step_citations.append(citation)
                    all_citations.append(citation)

                context = (
                    "\n\n".join(context_parts) if context_parts else "No relevant sources found."
                )
                result = f"Found {len(documents)} relevant sources.\n{context}"

                steps.append(
                    ResearchStep(
                        step=step_num,
                        action=action,
                        query=query,
                        result=result,
                        citations=step_citations,
                    )
                )

            else:
                steps_summary = "\n".join(
                    [f"Step {s.step}: {s.query} -> {s.result[:300]}" for s in steps]
                )
                final_prompt = (
                    f"Based on all the research steps above, provide a comprehensive final answer "
                    f'for the topic: "{request.topic}"\n\n'
                    f"Research steps:\n{steps_summary}\n\n"
                    "Final Answer:"
                )
                final_answer = await ollama_service.generate(prompt=final_prompt, temperature=0.3)

            total_time = int((time.time() - start_time) * 1000)
            return ResearchResponse(
                topic=request.topic,
                steps=steps,
                final_answer=final_answer.strip(),
                all_citations=all_citations,
                model_used=settings.ollama_model,
                total_processing_time_ms=total_time,
            )
        except Exception as e:
            logger.error("Research failed", error=str(e))
            raise


query_service = QueryService()
