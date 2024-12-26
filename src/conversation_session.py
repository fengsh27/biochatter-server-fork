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

from src.constants import ARGS_CONNECTION_ARGS, ARGS_DOCIDS_WORKSPACE, ARGS_RESULT_NUM, AZURE_OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_TYPE, OPENAI_API_VERSION, OPENAI_DEPLOYMENT_NAME, OPENAI_MODEL
from src.datatypes import ModelConfig, ChatterTypeEnum
from src.kg_agent import find_schema_info_node
from src.utils import build_user_name, get_azure_embedding_deployment, get_embedding_function, get_rag_agent_prompts

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
    "openai_api_key_type": ChatterTypeEnum.Unknown,
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
        authKey: str,
        useRAG: bool = False,
        useKG: bool = False,
        useAutoAgent = False,
        ragConfig: Optional[Dict] = None,
        kgConfig: Optional[Dict] = None,
        oncokbConfig: Optional[Dict] = None,
        modelConfig: Optional[Dict] = None,
    ):
        if self.chatter is None:
            return
        if not messages or len(messages) == 0:
            return
        selfmodelConfig = self.sessionData.modelConfig
        modelConfig = {**selfmodelConfig, **modelConfig} \
            if modelConfig is not None else selfmodelConfig
        self.sessionData.modelConfig = ModelConfig(**modelConfig)
        self._validate_chatter()
        api_key = authKey
        if not isinstance(
            self.chatter, AzureGptConversation
        ):  # chatter is instance of GptConversation
            import openai

            if not hasattr(self.chatter, "chat"):
                if not authKey:
                    return False
                user_name = build_user_name(self.sessionId, api_key)
                self.chatter.set_api_key(api_key, user_name)
        chatter = self.chatter
        chatter.token_usage = {}
        self._update_rags(
            useRAG=useRAG,
            ragConfig=ragConfig,
            useKG=useKG,
            kgConfig=kgConfig,
            oncokbConfig=oncokbConfig,
            useAutoAgent=useAutoAgent,
        )

        text = messages[-1]["content"]
        messages = messages[:-1]
        # pprint(messages)
        self._setup_messages(messages)
        try:
            (msg, usage, _) = self.chatter.query(text)
            contexts = self.chatter.get_last_injected_context()
            token_usage = chatter.token_usage
            return (msg, usage, contexts)
        except Exception as e:
            logger.error(e)
            raise e

    def _create_conversation(self):
        modelConfig = self.sessionData.modelConfig
        openai_key = modelConfig.openai_api_key
        if modelConfig.chatter_type == ChatterTypeEnum.ClientOpenAI:
            logger.info("create GptConversation")
            chatter = GptConversation(
                modelConfig.model, # "gpt-3.5-turbo", 
                prompts={"rag_agent_prompts": get_rag_agent_prompts()},
                update_token_usage=self._update_token_usage,
            )
            user_name = build_user_name(self.sessionId, openai_key)
            chatter.set_api_key(openai_key, user_name) # self.sessionId)
        elif modelConfig.chatter_type == ChatterTypeEnum.ClientWASM:
            logger.info("create WasmConversation")
            chatter = WasmConversation("mistral-wasm", prompts={})
        elif modelConfig.chatter_type == ChatterTypeEnum.ServerChatter:
            if OPENAI_API_TYPE in os.environ and os.environ[OPENAI_API_TYPE] == "azure": # create AzureGptConversation
                logger.info("create AzureGptConversation")
                chatter = AzureGptConversation(
                    deployment_name=os.environ[OPENAI_DEPLOYMENT_NAME],
                    model_name=os.environ[OPENAI_MODEL],
                    prompts={"rag_agent_prompts": get_rag_agent_prompts()},
                    version=os.environ[OPENAI_API_VERSION],
                    base_url=os.environ[AZURE_OPENAI_ENDPOINT],
                    update_token_usage=self._update_token_usage,
                )
                chatter.set_api_key(os.environ[OPENAI_API_KEY], "Azure Community")
            elif OPENAI_API_TYPE in os.environ and os.environ[OPENAI_API_TYPE] == "wasm":
                logger.info("create WasmConversation")
                chatter = WasmConversation("mistral-wasm", prompts={})
            else:
                logger.info("create GptConversation")
                chatter = GptConversation(
                    os.environ.get(OPENAI_MODEL, "gpt-3.5-turbo"),
                    prompts={"rag_agent_prompts": get_rag_agent_prompts()},
                    update_token_usage=self._update_token_usage,
                )  
                temp_api_key = os.environ.get("OPENAI_API_KEY", None)
                if temp_api_key is not None:
                    chatter.set_api_key(temp_api_key, "Gpt Community")
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
            return
        _, agent = self.chatter.find_rag_agent(agent_mode)
        if agent is None:
            return
        agent.use_prompt = False
        self.chatter.set_rag_agent(agent)

    def _update_vectorstore_agent(
        self,
        useRAG: bool,
        ragConfig: Optional[Dict]=None,
        useAutoAgent: bool=False,
    ):
        if ragConfig is None:
            # disabled
            self._disable_agent(RagAgentModeEnum.VectorStore)            
            return
        # update rag_agent
        try:
            (is_azure, azure_deployment, endpoint) = get_azure_embedding_deployment()
            doc_ids = ragConfig.get(ARGS_DOCIDS_WORKSPACE, None)
            embedding_func = get_embedding_function(
                is_azure=is_azure,
                azure_deployment=azure_deployment,
                azure_endpoint=endpoint,
            )
            rag_agent = RagAgent(
                mode=RagAgentModeEnum.VectorStore,
                model_name=os.environ.get(OPENAI_MODEL, "gpt-3.5-turbo"),
                connection_args=ragConfig[ARGS_CONNECTION_ARGS],
                use_prompt=useRAG or useAutoAgent,
                embedding_func=embedding_func,
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
            self._disable_agent(RagAgentModeEnum.KG)
            return
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
            self._disable_agent(RagAgentModeEnum.API_ONCOKB)
            return
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
    ):
        self._update_vectorstore_agent(
            useRAG=useRAG, ragConfig=ragConfig, useAutoAgent=useAutoAgent
        )
        self._update_kg_agent(useKG=useKG, kgConfig=kgConfig, useAutoAgent=useAutoAgent)
        self._update_oncokb_agent(oncokbConfig=oncokbConfig, useAutoAgent=useAutoAgent)
        
        self.chatter.use_ragagent_selector = useAutoAgent

    def _validate_chatter(self):
        if self.chatter is None:
            self.chatter = self._create_conversation()
            return
        if (self.sessionData.modelConfig.chatter_type 
            == ChatterTypeEnum.ClientOpenAI or
            self.sessionData.modelConfig.chatter_type \
            == ChatterTypeEnum.ServerOpenAI
            ):
            if isinstance(self.chatter, AzureGptConversation):
                self.chatter = self._create_conversation()
        else:
            if isinstance(self.chatter, GptConversation):
                self.chatter = self._create_conversation()
