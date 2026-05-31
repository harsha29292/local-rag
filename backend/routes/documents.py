"""Document management routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Response, UploadFile, status

from backend.models.domain import User
from backend.schemas.documents import DocumentIngestResponse, DocumentResponse
from backend.services.auth_service import get_current_user
from backend.services.document_service import DocumentService

router = APIRouter()


@router.get("", response_model=list[DocumentResponse])
async def list_documents(current_user: User = Depends(get_current_user)) -> list[DocumentResponse]:
    """List indexed documents for the authenticated user."""

    return await DocumentService().list_documents(current_user)


@router.post("", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> DocumentIngestResponse:
    """Upload and index a document."""

    document = await DocumentService().ingest_upload(current_user, file)
    return DocumentIngestResponse(document=document, message="Document indexed")


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_document(document_id: int, current_user: User = Depends(get_current_user)) -> Response:
    """Delete a document and its chunks."""

    await DocumentService().delete_document(current_user, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{document_id}/reindex", response_model=DocumentIngestResponse)
async def reindex_document(document_id: int, current_user: User = Depends(get_current_user)) -> DocumentIngestResponse:
    """Re-index an existing uploaded document."""

    document = await DocumentService().reindex_document(current_user, document_id)
    return DocumentIngestResponse(document=document, message="Document re-indexed")
