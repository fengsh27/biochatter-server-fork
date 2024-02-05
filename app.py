import json
from typing import Optional, Any, List
from flask import Flask, request
from dotenv import load_dotenv
import atexit
import logging
import os
from pymilvus import MilvusException
import pymilvus
from src.constants import (
    ARGS_CONNECTION_ARGS, 
    ERROR_MILVUS_CONNECT_FAILED, 
    ERROR_MILVUS_UNKNOWN, 
    ERROR_OK, 
    ERROR_UNKNOW, 
    ERRSTR_MILVUS_CONNECT_FAILED
)
from src.conversation_manager import (
    chat,
    has_conversation, 
    initialize_conversation
)

from src.document_embedder import (
    get_all_documents,
    get_connection_status as get_vectorstore_connection_status, 
    new_embedder_document,
    remove_document
)
from src.kg_agent import get_connection_status as get_kg_connection_status
from src.utils import get_auth
from src.job_recycle_conversations import run_scheduled_job_continuously

# prepare logger
logging.basicConfig(level=logging.INFO)
file_handler = logging.FileHandler("./logs/app.log")
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)


# run scheduled job: recycle unused session
cease_event = run_scheduled_job_continuously()
def onExit():
    cease_event.set()
atexit.register(onExit)

load_dotenv()
app = Flask(__name__)

DEFAULT_RAGCONFIG = {
    "splitByChar": True,
    "chunkSize": 1000,
    "overlapSize": 0,
    "resultNum": 3
}
RAG_KG = "KG"
RAG_VECTORSTORE = "VS"

def process_connection_args(rag: str, connection_args: dict) -> dict:
    if rag == RAG_VECTORSTORE:
        if connection_args.get('host', '').lower() == 'local':
            connection_args['host'] = (
                "127.0.0.1" if "HOST" not in os.environ else os.environ["HOST"]
            )
    elif rag == RAG_KG:
        if connection_args.get('host', '').lower() == 'local':
            connection_args['host'] = (
                "127.0.0.1" if "KGHOST" not in os.environ else os.environ["KGHOST"]
            )
    return connection_args

def extract_and_process_params_from_json_body(
        json: Optional[dict], 
        name: str, 
        defaultVal: Optional[Any]
    ) -> Optional[Any]:
    if not json:
        return defaultVal
    val = json.get(name, defaultVal)
    return val

@app.route('/v1/chat/completions', methods=['POST'])
def handle():
    auth = get_auth(request)
    jsonBody = request.json
    sessionId = extract_and_process_params_from_json_body(jsonBody, "session_id", defaultVal="")
    messages = extract_and_process_params_from_json_body(jsonBody, "messages", defaultVal=[])
    model = extract_and_process_params_from_json_body(jsonBody, "model", defaultVal="gpt-3.5-turbo")
    temperature = extract_and_process_params_from_json_body(jsonBody, "temperature", defaultVal=0.7)
    presence_penalty = extract_and_process_params_from_json_body(jsonBody, "presence_penalty", defaultVal=0)
    frequency_penalty = extract_and_process_params_from_json_body(jsonBody, "frequency_penalty", defaultVal=0)
    top_p = extract_and_process_params_from_json_body(jsonBody, "top_p", defaultVal=1)
    ragConfig = extract_and_process_params_from_json_body(jsonBody, "ragConfig", defaultVal=DEFAULT_RAGCONFIG)
    ragConfig[ARGS_CONNECTION_ARGS] = process_connection_args(RAG_VECTORSTORE, ragConfig[ARGS_CONNECTION_ARGS])
    useRAG = extract_and_process_params_from_json_body(jsonBody, "useRAG", defaultVal=False)
    kgConfig = extract_and_process_params_from_json_body(jsonBody, "kgConfig", defaultVal={})
    kgConfig[ARGS_CONNECTION_ARGS] = process_connection_args(RAG_KG, kgConfig[ARGS_CONNECTION_ARGS])
    useKG = extract_and_process_params_from_json_body(jsonBody, "useKG", defaultVal=False)

    if not has_conversation(sessionId):
        initialize_conversation(
            sessionId=sessionId,
            modelConfig={
                "temperature": temperature,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "top_p": top_p,
                "model": model,
                "auth": auth
            }
        )
    try:
        (msg, usage) = chat(
            sessionId, messages, auth, ragConfig, useRAG, kgConfig, useKG
        )
        return {
            "choices": [{
                "index": 0, 
                "message": {"role": "assistant", "content": msg}, 
                "finish_reason": "stop"}
            ], 
            "usage": usage, 
            "code": ERROR_OK,
        }
    except MilvusException as e:
        if e.code == pymilvus.Status.CONNECT_FAILED:
            return {
                "error": ERRSTR_MILVUS_CONNECT_FAILED, 
                "code": ERROR_MILVUS_CONNECT_FAILED,
            }
        else:
            return {"error": e.message, "code": ERROR_MILVUS_UNKNOWN}
    except Exception as e:
        return {"error": str(e)}

@app.route('/v1/rag/newdocument', methods=['POST'])
def newDocument():
    jsonBody = request.json
    tmpFile = extract_and_process_params_from_json_body(
        jsonBody, 'tmpFile', ''
    )
    filename = extract_and_process_params_from_json_body(
        jsonBody, 'filename', ''
    )
    ragConfig = extract_and_process_params_from_json_body(
        jsonBody, 'ragConfig', DEFAULT_RAGCONFIG
    )
    if type(ragConfig) is str:
        ragConfig = json.loads(ragConfig)
    ragConfig[ARGS_CONNECTION_ARGS] \
        = process_connection_args(RAG_VECTORSTORE, ragConfig[ARGS_CONNECTION_ARGS])
    auth = get_auth(request)
    # TODO: consider to be compatible with XinferenceDocumentEmbedder
    try:
        doc_id = new_embedder_document(
            authKey=auth,
            tmpFile=tmpFile, 
            filename=filename, 
            rag_config=ragConfig
        )
        return {"id": doc_id, "code": ERROR_OK}
    except MilvusException as e:
        if e.code == pymilvus.Status.CONNECT_FAILED:
            return {
                "error": ERRSTR_MILVUS_CONNECT_FAILED, 
                "code": ERROR_MILVUS_CONNECT_FAILED
            }
        else:
            return {"error": e.message, "code": ERROR_MILVUS_UNKNOWN}
    except Exception as e:
        return {"error": str(e), "code": ERROR_UNKNOW}
    
@app.route('/v1/rag/alldocuments', methods=['POST'])
def getAllDocuments():
    def post_process(docs: List[Any]):
        for doc in docs:
            doc['id'] = str(doc['id'])
        return docs
    auth = get_auth(request)
    jsonBody = request.json
    connection_args = extract_and_process_params_from_json_body(
        jsonBody, ARGS_CONNECTION_ARGS, None
    )
    connection_args = process_connection_args(RAG_VECTORSTORE, connection_args)
    doc_ids = extract_and_process_params_from_json_body(
        jsonBody, "docIds", None
    )
    try:
        docs = get_all_documents(auth, connection_args, doc_ids=doc_ids)
        docs = post_process(docs)
        return {"documents": docs, "code": ERROR_OK}
    except MilvusException as e:
        if e.code == pymilvus.Status.CONNECT_FAILED:
            return {
                "error": ERRSTR_MILVUS_CONNECT_FAILED, 
                "code": ERROR_MILVUS_CONNECT_FAILED
            }
        else:
            return {"error": e.message, "code": ERROR_MILVUS_UNKNOWN}
    except Exception as e:
        return {"error": str(e), "code": ERROR_UNKNOW}
    
@app.route('/v1/rag/document', methods=['DELETE'])
def removeDocument():
    jsonBody = request.json
    auth = get_auth(request)
    docId = extract_and_process_params_from_json_body(jsonBody, 'docId', '')
    connection_args = extract_and_process_params_from_json_body(
        jsonBody, ARGS_CONNECTION_ARGS, None
    )
    connection_args = process_connection_args(RAG_VECTORSTORE, connection_args)
    doc_ids = extract_and_process_params_from_json_body(jsonBody, "docIds", None)
    if len(docId) == 0:
        return {"error": "Failed to find document"}
    try:
        remove_document(
            docId, 
            authKey=auth, 
            connection_args=connection_args, 
            doc_ids=doc_ids
        )
        return {"id": docId, "code": ERROR_OK}
    except MilvusException as e:
        if e.code == pymilvus.Status.CONNECT_FAILED:
            return {
                "error": ERRSTR_MILVUS_CONNECT_FAILED, 
                "code": ERROR_MILVUS_CONNECT_FAILED
            }
        else:
            return {"error": e.message, "code": ERROR_MILVUS_UNKNOWN}
    except Exception as e:
        return {"error": str(e), "code": ERROR_UNKNOW}
    
@app.route('/v1/rag/connectionstatus', methods=['POST'])
def getConnectionStatus():
    try:
        auth = get_auth(request)
        jsonBody = request.json
        connection_args = extract_and_process_params_from_json_body(
            jsonBody, ARGS_CONNECTION_ARGS, None
        )
        connection_args = process_connection_args(RAG_VECTORSTORE, connection_args)
        connected = get_vectorstore_connection_status(
            connection_args, auth
        )
        return {
            "status": "connected" if connected else "disconnected", 
            "code": ERROR_OK
        }
    except MilvusException as e:
        return {"error": e.message, "code": ERROR_MILVUS_UNKNOWN}
    except Exception as e:
        return {"error": str(e), "code": ERROR_UNKNOW}
    
    
@app.route('/v1/kg/connectionstatus', methods=['POST'])
def getKGConnectionStatus():
    try:
        jsonBody = request.json
        connection_args = extract_and_process_params_from_json_body(
            jsonBody, ARGS_CONNECTION_ARGS, None
        )
        connection_args = process_connection_args(RAG_KG, connection_args)
        connected = get_kg_connection_status(connection_args)
        return {
            "status": "connected" if connected else "disconnected", 
            "code": ERROR_OK
        }
    except Exception as e:
        return {"error": str(e), "code": ERROR_UNKNOW}

