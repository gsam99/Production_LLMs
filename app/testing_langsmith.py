import os
from config import get_settings

settings = get_settings()

os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.langchain_tracing_v2 else "false"
os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=settings.primary_model, api_key=settings.openai_api_key)
response = llm.invoke("Say hello")
print(response.content)