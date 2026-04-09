from dataclasses import dataclass

USER_PROFILE_CONTEXT = "user_profile"

def get_session_context_id(session_id):
    return f'{session_id}_metadata'

def get_session_entity_id(session_id):
    return f'{session_id}_entities'

@dataclass
class Context:
    user_id: str
    conversation_id: str


class BackendMixin:
    def _load_json(self, data, default):
        import json
        if not data:
            return default
        if isinstance(data, (dict, list)):
            return data
        try:
            return json.loads(data)
        except Exception:
            return default

