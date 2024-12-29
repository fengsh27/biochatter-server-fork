
import base64
from typing import List, Optional, Tuple
import os

from flask import Request
from src.constants import OPENAI_API_KEY
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings

def get_rag_agent_prompts() -> List[str]:
    return [
        "The user has provided additional background information from scientific "
        "articles.",
        "Take the following statements into account and specifically comment on "
        "consistencies and inconsistencies with all other information available to "
        "you: {statements}",
    ]
    
def build_user_name(session_id: str, api_key: str):
    username = f"{session_id}:-:{api_key}"
    encoded = base64.b64encode(username.encode('utf-8'))
    return encoded.decode('utf-8')

