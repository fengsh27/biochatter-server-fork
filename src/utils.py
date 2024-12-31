
import base64
from typing import List, Optional, Tuple

def get_rag_agent_prompts() -> List[str]:
    return [
        "The user has provided additional background information from scientific "
        "articles.",
        "Take the following statements into account and specifically comment on "
        "consistencies and inconsistencies with all other information available to "
        "you: {statements}",
    ]
    
USERNAME_SEPARATOR = ":-:"
def encode_user_name(api_key: str, session_id: Optional[str]=None):
    if session_id is not None:
        username = f"{api_key}{USERNAME_SEPARATOR}{session_id}"
    else:
        username = api_key
    encoded = base64.b64encode(username.encode('utf-8'))
    return encoded.decode('utf-8')

def decode_user_name(name: str) -> Tuple[str, str | None]:
    decoded_bytes = base64.b64decode(name.encode("utf-8"))
    decoded_username = decoded_bytes.decode("utf-8")
    if USERNAME_SEPARATOR in decoded_username:
        arr = decoded_username.split(USERNAME_SEPARATOR)
        assert len(arr) == 2
        return arr[0], arr[1]
    else:
        return decoded_username, None
