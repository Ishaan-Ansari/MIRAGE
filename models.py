import json
from typing import Tuple, Type

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declared_attr, relationship
from sqlalchemy.sql import func

from database import PostGresBase


class RAGDocument(PostGresBase):
    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    text_hash = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    summary = Column(String, nullable=False)
    created_on = Column(DateTime(timezone=True), server_default=func.now())
    updated_on = Column(DateTime(timezone=True), onupdate=func.now())
    document_filepath = Column(String, nullable=False)
    summary_embedding = Column(Vector(1536))
    num_chunks = Column(Integer, default=0)
    chunks_added_on = Column(DateTime(timezone=True), server_default=func.now())
    verified = Column(Boolean, default=False)
    verified_on = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)

    @declared_attr
    def chunks(cls):
        pass

    def __repr__(self) -> str:
        return json.dumps({
            'id': self.id,
            "title": self.title,
            "document_filepath": self.document_filepath,
            "summary": self.summary,
            "verified": self.verified,
        },
        indent=4,
    )

class RAGDocumentChunk(PostGresBase):
    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_text = Column(String, nullable=False)
    chunk_embedding = Column(Vector(1536))

    @declared_attr
    def docment_id(cls):
        pass

    @declared_attr
    def document(cls):
        pass

    def __repr__(self) -> str:
        return json.dumps(
            {"document_id": self.docment_id, "chunk_text": self.chunk_text, "chunk_embedding": self.chunk_embedding},
            indent=4,
        )

class RAGClassFactory:
    # Class registry dictionary schema
    # {"pipeline_name": Type[RAGDocument], Type[RAGDocumentChunk]}
    _CLASS_REGISTRY = {}

    @classmethod
    def create_rag_class(
            cls,
            pipeline_name: str,
    ) -> Tuple[Type[RAGDocument], Type[RAGDocumentChunk]]:
        pass

