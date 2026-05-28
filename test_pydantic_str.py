from pydantic import BaseModel
class ChatCompletion(BaseModel):
    choices: list
try:
    c = ChatCompletion.model_validate("{\"choices\": []}")
    print(type(c))
except Exception as e:
    print(repr(e))
