from openai import OpenAI
import httpx
client = OpenAI(api_key="test", base_url="https://httpbin.org/anything")
try:
    completion = client.chat.completions.create(model="test", messages=[{"role": "user", "content": "hi"}])
    print(type(completion))
except Exception as e:
    print(repr(e))
