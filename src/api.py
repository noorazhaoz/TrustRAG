"""TrustRAG API with adaptive retrieval, generation and local storage.

Local development only. No authentication.
"""
import pandas as pd
from contextlib import asynccontextmanager
from typing import Any
from fastapi import File, Form, UploadFile
from src.pdf_ingestion import pdf_to_evidence, MAX_PDF_BYTES
from src.document_store import DocumentStore
from threading import Lock
from src.answer_service import AnswerService
from src.llm_backend import MLXBackend

import json
import logging
import uuid
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.policy_router import FrozenPolicyRouter
from src.adaptive_retriever import AdaptiveRetriever



class EvidenceUnit(BaseModel):
    paragraph_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    record_type: str = "paragraph"


class DocumentRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=128)
    units: list[EvidenceUnit] = Field(min_length=1, max_length=10000)


class QueryRequest(BaseModel):
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=5000)
    question_id: str = "online"

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value):
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value

class AnswerRequest(QueryRequest):
    max_tokens: int = Field(default=256, ge=1, le=1024)

def build_retriever():
    router = FrozenPolicyRouter()
    return AdaptiveRetriever(router=router)


def create_app(retriever=None, generator_backend=None, document_store=None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.retriever is None:
            app.state.retriever = build_retriever()

        # Restore saved documents only for the real application.
        # Tests using FakeRetriever remain isolated.
        if retriever is None:
            restored = app.state.document_store.restore_all(
                app.state.retriever
            )
            logging.getLogger("trustrag").info(
                "Restored %d document indexes", restored
            )

        yield

    app = FastAPI(
        title="TrustRAG",
        description="Cost-aware adaptive retrieval service",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.retriever = retriever
    app.state.document_store = (
        document_store
        if document_store is not None
        else (DocumentStore() if retriever is None else None)
    )
    app.state.generator_backend = generator_backend
    app.state.generator_lock = Lock()

    @app.get("/health")
    def health(request: Request):
        return {
            "status": "ready"
            if request.app.state.retriever is not None
            else "initializing",
            "service": "TrustRAG",
        }

    @app.post("/answer")
    def answer_question(payload: AnswerRequest, request: Request):
        trace_id = uuid.uuid4().hex
        start = perf_counter()
        engine = request.app.state.retriever

        if payload.document_id not in engine.documents:
            raise HTTPException(
                status_code=404,
                detail="Document not found",
            )

        try:
            with request.app.state.generator_lock:
                if request.app.state.generator_backend is None:
                    request.app.state.generator_backend = MLXBackend()

                service = AnswerService(
                    retriever=engine,
                    generator=request.app.state.generator_backend,
                    max_tokens=payload.max_tokens,
                )

                result = service.answer(
                    document_id=payload.document_id,
                    question=payload.question,
                    question_id=payload.question_id,
                )

            request_latency_ms = (
                                         perf_counter() - start
                                 ) * 1000

            log_record = {
                "event": "answer_completed",
                "trace_id": trace_id,
                "document_id": payload.document_id,
                "retrieval_policy": result.retrieval_policy,
                "retrieval_latency_ms": result.retrieval_latency_ms,
                "generation_latency_ms": result.generation_latency_ms,
                "pipeline_latency_ms": result.pipeline_latency_ms,
                "request_processing_latency_ms": request_latency_ms,
                "citations_valid": result.citation_validation["valid"],
                "evidence_truncated": result.evidence_truncated,
            }

            logging.getLogger("trustrag").info(
                json.dumps(log_record)
            )

            return {
                "trace_id": trace_id,
                "question_id": payload.question_id,
                "document_id": payload.document_id,
                "answer": result.answer,
                "model_id": result.model_id,
                "retrieval_policy": result.retrieval_policy,
                "predicted_delta": result.predicted_delta,
                "evidence": result.evidence,
                "evidence_truncated": result.evidence_truncated,
                "retrieval_latency_ms": result.retrieval_latency_ms,
                "generation_latency_ms": result.generation_latency_ms,
                "pipeline_latency_ms": result.pipeline_latency_ms,
                "request_processing_latency_ms": request_latency_ms,
                "citation_validation": result.citation_validation,
                "generation_tokens": result.generation_tokens,
                "finish_reason": result.finish_reason,
                "max_tokens": payload.max_tokens,
            }

        except ValueError as exc:
            logging.getLogger("trustrag").warning(
                json.dumps({
                    "event": "answer_failed",
                    "trace_id": trace_id,
                    "error_type": type(exc).__name__,
                })
            )
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception:
            logging.getLogger("trustrag").exception(
                "Answer request failed trace_id=%s",
                trace_id,
            )
            raise

    @app.post("/query")
    def query(payload: QueryRequest, request: Request):
        engine = request.app.state.retriever

        try:
            result = engine.query(
                document_id=payload.document_id,
                question=payload.question,
                question_id=payload.question_id,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Document not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=str(exc)
            ) from exc

        return {
            "question_id": payload.question_id,
            "document_id": payload.document_id,
            "retrieval_policy": str(result.action),
            "predicted_delta": float(result.predicted_delta),
            "hybrid_executed": bool(result.hybrid_executed),
            "features": {
                name: (
                    None if pd.isna(value) else float(value)
                )
                for name, value in result.features.items()
            },
            "evidence": result.evidence,
            "retrieval_latency_ms": float(result.latency_ms),
        }

    @app.post("/documents/upload", status_code=201)
    async def upload_pdf(
            request: Request,
            document_id: str = Form(..., min_length=1, max_length=128),
            file: UploadFile = File(...),
    ):
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported",
            )

        pdf_bytes = await file.read(MAX_PDF_BYTES + 1)
        await file.close()

        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise HTTPException(
                status_code=413,
                detail="PDF exceeds the 10 MB limit",
            )

        from starlette.concurrency import run_in_threadpool

        try:
            units = await run_in_threadpool(
                pdf_to_evidence, pdf_bytes, document_id
            )
            await run_in_threadpool(
                request.app.state.retriever.index_document,
                document_id,
                units,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        if request.app.state.document_store is not None:
            try:
                await run_in_threadpool(
                    request.app.state.document_store.save_document,
                    request.app.state.retriever,
                    document_id,
                )
            except Exception as exc:
                request.app.state.retriever.remove_document(document_id)
                logging.getLogger("trustrag").exception(
                    "PDF persistence failed: %s", document_id
                )
                raise HTTPException(
                    status_code=500,
                    detail="Document persistence failed",
                ) from exc

        return {
            "document_id": document_id,
            "status": "indexed",
            "indexed_units": len(units),
            "pages_with_evidence": len({unit["page"] for unit in units}),
        }

    @app.post("/documents", status_code=201)
    def index_document(
            payload: DocumentRequest,
            request: Request,
    ):
        engine = request.app.state.retriever

        units = [
            {
                **unit.model_dump(),
                "document_id": payload.document_id,
            }
            for unit in payload.units
        ]

        try:
            engine.index_document(payload.document_id, units)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        if request.app.state.document_store is not None:
            try:
                request.app.state.document_store.save_document(
                    engine,
                    payload.document_id,
                )
            except Exception:
                engine.remove_document(payload.document_id)
                logging.getLogger("trustrag").exception(
                    "Document persistence failed: %s",
                    payload.document_id,
                )
                raise HTTPException(
                    status_code=500,
                    detail="Document persistence failed",
                )

        return {
            "document_id": payload.document_id,
            "indexed_units": len(units),
            "status": "indexed",
        }

    return app


app = create_app()
