

import os
from typing import Optional, Tuple

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
# from langchain_community.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from src.constants import AZURE_COMMUNITY, GPT_COMMUNITY, OPENAI_API_KEY, OPENAI_MODEL
from src.datatypes import AuthTypeEnum
from src.utils import encode_user_name

def _parse_api_key(bearToken: str) -> str:
    if not bearToken:
        return ""
    bearToken = bearToken.strip()
    bearToken = bearToken.replace("Bearer ", "")
    return bearToken

def llm_get_auth_type(client_key: Optional[str]=None) -> AuthTypeEnum:
    if client_key is not None and len(client_key.strip()) > 0:
        return AuthTypeEnum.ClientOpenAI
    
    if os.environ.get("OPENAI_API_TYPE", "") == "azure":
        return AuthTypeEnum.ServerAzureOpenAI
    if len(os.environ.get("OPENAI_API_KEY", "") > 0):
        return AuthTypeEnum.ServerOpenAI
    
    return AuthTypeEnum.Unknown

def llm_get_user_name_and_model(
        client_key: Optional[str]=None,
        session_id: Optional[str]=None,
        model: Optional[str]=None
    ) -> Tuple[str, str]:
    auth_type = llm_get_auth_type(client_key=client_key)
    if auth_type == AuthTypeEnum.ClientOpenAI:
        return session_id, model
    mod = os.environ.get(
        OPENAI_MODEL,
        model if model is not None else "gpt-3.5-turbo",
    )
    return (
        AZURE_COMMUNITY if auth_type == AuthTypeEnum.ServerAzureOpenAI else GPT_COMMUNITY ,
        mod
    )

def llm_get_client_auth(client_key: str | None) -> str | None:
    # try to parse bearer key first
    key = _parse_api_key(client_key)
    
    return key if key is not None else ""

def llm_get_auth_key(client_key: Optional[str]=None):
    auth = None
    if client_key is None or len(client_key) > 0:
        auth = _parse_api_key(client_key)
    
    if auth is not None and len(auth) > 0:
        return auth
    
    # OPENAI_API_KEY is provided by server
    if OPENAI_API_KEY in os.environ and os.environ[OPENAI_API_KEY]:
        auth = os.environ[OPENAI_API_KEY]
    
    return auth

def llm_get_embedding_function(
        client_key: Optional[str]=None,
        model: Optional[str] = "text-embedding-ada-002",
    ) -> OpenAIEmbeddings | AzureOpenAIEmbeddings | None:
    auth_type = llm_get_auth_type(client_key)
    if auth_type == AuthTypeEnum.ClientOpenAI:
        return OpenAIEmbeddings(
            api_key=client_key,
            model=model,
        )
    elif auth_type == AuthTypeEnum.ServerOpenAI:
        return OpenAIEmbeddings(
            api_key=llm_get_auth_key(),
        )
    elif auth_type == AuthTypeEnum.ServerAzureOpenAI:
        deployment = os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", "")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        return AzureOpenAIEmbeddings(
            api_key=llm_get_auth_key(),
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            model=model,
        )
    else:
        return None