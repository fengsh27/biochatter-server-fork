
from biochatter.vectorstore import DocumentEmbedder, DocumentReader
from typing import Any, Dict, List, Optional
import logging

from src.utils import get_azure_embedding_deployment

logger = logging.getLogger(__name__)

def new_embedder_document(authKey: str, tmpFile: str, filename: str, rag_config: Any, useRAG: Optional[bool] = False):
    api_key = authKey
    (is_azure, azure_deployment, endpoint) = get_azure_embedding_deployment()
    rag_agent = DocumentEmbedder(
      used=True,
      use_prompt=useRAG,
      chunk_size=rag_config["chunkSize"],
      chunk_overlap=rag_config["overlapSize"],
      split_by_characters=rag_config["splitByChar"],
      n_results=rag_config["resultNum"],
      api_key=api_key,     
      is_azure=is_azure,
      azure_deployment=azure_deployment, 
      azure_endpoint=endpoint,
      connection_args=rag_config["connectionArgs"]
    )
    rag_agent.connect()
    reader = DocumentReader()
    docs = reader.load_document(tmpFile)
    if len(docs) > 0:
        for doc in docs:
            doc.metadata.update({"source": filename})
    logger.info('save_document')
    logger.error('save_document')
    return rag_agent.save_document(docs)
    
def get_all_documents(authKey: str, connection_args: Dict, doc_ids: Optional[List[str]] = None):
    api_key = authKey
    (is_azure, azure_deployment, endpoint) = get_azure_embedding_deployment()
    rag_agent = DocumentEmbedder(
        api_key=api_key,
        connection_args=connection_args,
        documentids_workspace=doc_ids,
        is_azure=is_azure,
        azure_endpoint=endpoint,
        azure_deployment=azure_deployment,
    )
    rag_agent.connect()
    return rag_agent.get_all_documents()

def remove_document(docId: str, authKey: str, connection_args, doc_ids: Optional[List[str]] = None):
    api_key = authKey
    (is_azure, azure_deployment, endpoint) = get_azure_embedding_deployment()
    rag_agent = DocumentEmbedder(
        api_key=api_key,
        connection_args=connection_args,
        documentids_workspace=doc_ids,
        is_azure=is_azure,
        azure_deployment=azure_deployment,
        azure_endpoint=endpoint
    )
    rag_agent.connect()
    rag_agent.remove_document(doc_id=docId)

def get_connection_status(connection_args: Optional[Dict]=None, authKey: Optional[str] = None) -> bool:
    if connection_args is None:
        return False
    (is_azure, azure_deployment, endpoint) = get_azure_embedding_deployment()
    try:
        rag_agent = DocumentEmbedder(
            api_key=authKey,
            is_azure=is_azure,
            azure_deployment=azure_deployment,
            azure_endpoint=endpoint,
            connection_args=connection_args
        )
        rag_agent.connect()
        return True
    except Exception as e:
        return False
