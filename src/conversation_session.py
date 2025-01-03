from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
import os

from biochatter.llm_connect import (
    AzureGptConversation,
    GptConversation,
    WasmConversation,
)
from biochatter.rag_agent import RagAgent, RagAgentModeEnum
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from src.constants import (
    ARGS_CONNECTION_ARGS,
    ARGS_DOCIDS_WORKSPACE,
    ARGS_RESULT_NUM,
    ARGS_USE_REFLEXION,
    AZURE_COMMUNITY,
    AZURE_OPENAI_ENDPOINT,
    GPT_COMMUNITY,
    OPENAI_API_KEY,
    OPENAI_API_VERSION,
    OPENAI_DEPLOYMENT_NAME,
    OPENAI_MODEL,
)
from src.datatypes import ModelConfig, AuthTypeEnum
from src.kg_agent import find_schema_info_node
from src.llm_auth import llm_get_embedding_function
from src.token_usage_database import update_token_usage
from src.utils import get_rag_agent_prompts

logger = logging.getLogger(__name__)

MAX_AGE = 3 * 24 * 3600 * 1000  # 3 days

defaultModelConfig = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "sendMemory": True,
    "historyMessageCount": 4,
    "compressMessageLengthThreshold": 2000,
    "chatter_type": AuthTypeEnum.Unknown.value,
    "openai_api_key": None,
}

class SessionData:
    def __init__(
        self,
        sessionId: str,
        modelConfig: dict,
    ):
        self.sessionId = sessionId
        merged_model_config = {**defaultModelConfig, **modelConfig}
        self.modelConfig = ModelConfig(**merged_model_config)
        self.createdAt = int(datetime.now().timestamp() * 1000)  # in milliseconds
        self.refreshedAt = self.createdAt
        self.maxAge = MAX_AGE

class ConversationSession:
    def __init__(
        self,
        sessionId: str,
        modelConfig: dict
    ):
        self.sessionData = SessionData(sessionId, modelConfig)
        self.chatter = self._create_conversation()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        useRAG: bool = False,
        useKG: bool = False,
        useAutoAgent = False,
        ragConfig: Optional[Dict] = None,
        kgConfig: Optional[Dict] = None,
        oncokbConfig: Optional[Dict] = None,
        modelConfig: Optional[Dict] = None,
    ):
        if self.chatter is None:
            return None
        if not messages or len(messages) == 0:
            return None
        client_key = modelConfig.get("openai_api_key", None)
        model = modelConfig.get("model", None)
        embedding_func = llm_get_embedding_function(client_key=client_key)
        self._validate_chatter(modelConfig)        
        
        api_key = self.sessionData.modelConfig.openai_api_key
        if not isinstance(
            self.chatter, AzureGptConversation
        ) and isinstance(
            self.chatter, GptConversation
        ):  # chatter is instance of GptConversation
            if not hasattr(self.chatter, "chat"):
                if not api_key:
                    return False
                session_id = self.sessionData.sessionId
                auth_type = self.sessionData.modelConfig.chatter_type
                user_name = session_id if auth_type is AuthTypeEnum.ClientOpenAI else GPT_COMMUNITY
                self.chatter.set_api_key(api_key, user_name)
        chatter = self.chatter
        # embedding_func = llm_get_embedding_function()
        self._update_biochatter_agents(
            useRAG=useRAG,
            ragConfig=ragConfig,
            useKG=useKG,
            kgConfig=kgConfig,
            oncokbConfig=oncokbConfig,
            useAutoAgent=useAutoAgent,
            embedding_function=embedding_func
        )

        text = messages[-1]["content"]
        messages = messages[:-1]
        # pprint(messages)
        self._setup_messages(messages)
        try:
            (msg, usage, _) = self.chatter.query(text)
            contexts = self.chatter.get_last_injected_context()
            return (msg, usage, contexts)
        except Exception as e:
            logger.error(e)
            raise e

    def _create_conversation(self):
        modelConfig = self.sessionData.modelConfig
        openai_key = modelConfig.openai_api_key
        if modelConfig.chatter_type == AuthTypeEnum.ClientOpenAI:
            logger.info("create GptConversation")
            chatter = GptConversation(
                modelConfig.model, # "gpt-3.5-turbo", 
                prompts={"rag_agent_prompts": get_rag_agent_prompts()},
                update_token_usage=self._update_token_usage,
            )
            user_name = self.sessionData.sessionId
            chatter.set_api_key(openai_key, user_name) # 
        elif modelConfig.chatter_type == AuthTypeEnum.ClientWASM:
            logger.info("create WasmConversation")
            chatter = WasmConversation("mistral-wasm", prompts={})
        elif modelConfig.chatter_type == AuthTypeEnum.ServerAzureOpenAI:
            logger.info("create AzureGptConversation")
            chatter = AzureGptConversation(
                deployment_name=os.environ[OPENAI_DEPLOYMENT_NAME],
                model_name=os.environ[OPENAI_MODEL],
                prompts={"rag_agent_prompts": get_rag_agent_prompts()},
                version=os.environ[OPENAI_API_VERSION],
                base_url=os.environ[AZURE_OPENAI_ENDPOINT],
                update_token_usage=self._update_token_usage,
            )
            user_name = AZURE_COMMUNITY
            chatter.set_api_key(os.environ[OPENAI_API_KEY], user_name)
        elif modelConfig.chatter_type == AuthTypeEnum.ServerOpenAI:
            logger.info("create GptConversation")
            chatter = GptConversation(
                os.environ.get(OPENAI_MODEL, "gpt-3.5-turbo"),
                prompts={"rag_agent_prompts": get_rag_agent_prompts()},
                update_token_usage=self._update_token_usage,
            )  
            temp_api_key = os.environ.get("OPENAI_API_KEY", None)
            user_name = GPT_COMMUNITY
            chatter.set_api_key(temp_api_key, user_name)
        else:
            chatter = None
    
        return chatter

    def _setup_messages(self, openai_msgs: List[Any]):
        if self.chatter is None:
            return False
        self.chatter.messages = []
        for msg in openai_msgs:
            if msg["role"] == "system":
                self.chatter.append_system_message(msg["content"])
            elif msg["role"] == "assistant":
                self.chatter.append_ai_message(msg["content"])
            elif msg["role"] == "user":
                self.chatter.append_user_message(msg["content"])

    def _disable_biochatter_agent(self, agent_mode: str):
        if self.chatter is None:
            return None
        _, agent = self.chatter.find_rag_agent(agent_mode)
        if agent is None:
            return None
        agent.use_prompt = False
        self.chatter.set_rag_agent(agent)

    def _update_vectorstore_agent(
        self,
        useRAG: bool,
        ragConfig: Optional[Dict]=None,
        useAutoAgent: bool=False,
        embedding_function: OpenAIEmbeddings | AzureOpenAIEmbeddings | None=None,
    ):
        if ragConfig is None:
            # disabled
            self._disable_biochatter_agent(RagAgentModeEnum.VectorStore)            
            return None
        # update rag_agent
        try:
            doc_ids = ragConfig.get(ARGS_DOCIDS_WORKSPACE, None)
                        
            rag_agent = RagAgent(
                mode=RagAgentModeEnum.VectorStore,
                model_name=os.environ.get(OPENAI_MODEL, "gpt-3.5-turbo"),
                connection_args=ragConfig[ARGS_CONNECTION_ARGS],
                use_prompt=useRAG or useAutoAgent,
                embedding_func=embedding_function,
                documentids_workspace=doc_ids,
                n_results=ragConfig.get(ARGS_RESULT_NUM, 3),
            )
            if "description" in ragConfig and ragConfig["description"] is not None:
                rag_agent.agent_description = ragConfig["description"]
            self.chatter.set_rag_agent(rag_agent)
        except Exception as e:
            logger.error(e)
    
    def _update_kg_agent(
        self,
        useKG: bool,
        kgConfig: Optional[Dict]=None,
        useAutoAgent: bool=False,
    ):
        if kgConfig is None:
            self._disable_biochatter_agent(RagAgentModeEnum.KG)
            return None
        try:
            schema_info = find_schema_info_node(kgConfig["connectionArgs"])
            if schema_info is not None:
                kg_agent = RagAgent(
                    mode=RagAgentModeEnum.KG,
                    model_name=os.environ.get(OPENAI_MODEL, "gpt-3.5-turbo"),
                    connection_args=kgConfig["connectionArgs"],
                    use_prompt=useKG or useAutoAgent,
                    schema_config_or_info_dict=schema_info,
                    conversation_factory=self._create_conversation,
                    n_results=kgConfig.get(ARGS_RESULT_NUM, 3),
                    use_reflexion=kgConfig.get(ARGS_USE_REFLEXION, False),
                )
                if "description" in kgConfig and kgConfig["description"] is not None:
                    kg_agent.agent_description = kgConfig["description"]
                self.chatter.set_rag_agent(kg_agent)
        except Exception as e:
            logger.error(e)

    def _update_oncokb_agent(
        self,
        oncokbConfig: Optional[Dict]=None,
        useAutoAgent: bool=False,
    ):
        if oncokbConfig is None:
            self._disable_biochatter_agent(RagAgentModeEnum.API_ONCOKB)
            return None
        oncokb_agent = RagAgent(
            mode=RagAgentModeEnum.API_ONCOKB,
            conversation_factory=self._create_conversation,
            use_prompt=oncokbConfig["useOncoKB"] or useAutoAgent ,
        )
        if "description" in oncokbConfig:
            oncokb_agent.agent_description = oncokbConfig["description"]
        self.chatter.set_rag_agent(oncokb_agent)

    def _update_biochatter_agents(
        self,
        useRAG: bool,
        useKG: bool,
        useAutoAgent: bool,
        ragConfig: Optional[Dict]=None,
        kgConfig: Optional[Dict]=None,
        oncokbConfig: Optional[Dict]=None,
        embedding_function: OpenAIEmbeddings | AzureOpenAIEmbeddings | None=None,
    ):
        self._update_vectorstore_agent(
            useRAG=useRAG,
            ragConfig=ragConfig,
            useAutoAgent=useAutoAgent,
            embedding_function=embedding_function,
        )
        self._update_kg_agent(useKG=useKG, kgConfig=kgConfig, useAutoAgent=useAutoAgent)
        self._update_oncokb_agent(oncokbConfig=oncokbConfig, useAutoAgent=useAutoAgent)
        
        self.chatter.use_ragagent_selector = useAutoAgent

    def _merge_modelConfig(self, modelConfig: Dict):
        selfmodelConfig = self.sessionData.modelConfig.model_dump()
        modelConfig = {**selfmodelConfig, **modelConfig} \
            if modelConfig is not None else selfmodelConfig
        self.sessionData.modelConfig = ModelConfig(**modelConfig)

    def _validate_chatter(self, modelConfig: Optional[Dict]=None):
        if self.chatter is None:
            self._merge_modelConfig(modelConfig)
            self.chatter = self._create_conversation()
            return

        if modelConfig is None:
            return

        if isinstance(self.chatter, AzureGptConversation) and (\
            modelConfig["chatter_type"] == AuthTypeEnum.ClientOpenAI.value or \
            modelConfig["chatter_type"] == AuthTypeEnum.ClientWASM.value \
        ):
            self._merge_modelConfig(modelConfig)
            self.chatter = self._create_conversation()
        elif isinstance(self.chatter, GptConversation) \
            and modelConfig["openai_api_key"] != \
                self.sessionData.modelConfig.openai_api_key:
            self._merge_modelConfig(modelConfig)
            self.chatter = self._create_conversation()
        elif modelConfig["chatter_type"] == AuthTypeEnum.ClientOpenAI.value and \
            ((modelConfig["model"] is not None and
              modelConfig["model"] != self.sessionData.modelConfig.model) or
             (modelConfig["openai_api_key"] is not None and
              modelConfig["openai_api_key"] != self.sessionData.modelConfig.openai_api_key)):
            ## self.chatter has been created and its type remains the same, but we need to
            ## change its configuration (model or openai_api_key)
            self._merge_modelConfig(modelConfig)
            sessionData = self.sessionData
            self.chatter.model_name = sessionData.modelConfig.model
            username = sessionData.sessionId
            self.chatter.set_api_key(self.sessionData.modelConfig.openai_api_key, username)
        else:
            self._merge_modelConfig(modelConfig)

    def _update_token_usage(self, user: str, model: str, usage: dict):
        session_id = self.sessionData.sessionId
        update_token_usage(user, session_id, model, usage)
