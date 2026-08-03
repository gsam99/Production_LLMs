from pydantic_settings import BaseSettings
from functools import lru_cache
import boto3
import json

def get_secrets_from_aws():
    client= boto3.client("secretsmanager",region_name="us-east-1")
    response= client.get_secret_value(SecretId="RAG-prod-secretsmanager")
    return json.loads(response["SecretString"])



class Settings(BaseSettings):
    upstash_redis_rest_url: str= ""
    upstash_redis_rest_token: str = ""
    openai_api_key:str =""
    primary_model:str ="gpt-4o-mini"
    fallback_model:str="gpt-4o-mini"

    anthropic_api_key:str =""
    anthropic_primary_model:str = "claude-haiku-4-5"
    anthropic_fallback_model:str= "claude-haiku-4-5"

    langchain_tracing_v2: bool=True
    langchain_api_key: str=""
    langchain_project:str="production-api"

    app_env: str= "development"
    log_level: str= "INFO"
    rate_limit: str="20/minute"
    cache_ttl_seconds: int=300
    max_retries: int=3

    model_config={"env_file":".env","extra":"ignore"}

    @property
    def is_production(self)->bool:
        return self.app_env=="production"
@lru_cache
def get_settings() ->Settings:
    settings= Settings()
    if settings.is_production:
        try:
            secrets= get_secrets_from_aws()
            settings= Settings(
                openai_api_key=secrets.get("OPENAI_API_KEY", ""),
                anthropic_api_key=secrets.get("ANTHROPIC_API_KEY", ""),
                langchain_api_key=secrets.get("LANGCHAIN_API_KEY", ""),
                langchain_tracing_v2=secrets.get("LANGCHAIN_TRACING_V2", "true") == "true",
                langchain_project=secrets.get("LANGCHAIN_PROJECT", "production-api"),
                app_env="production"
            )
        except Exception as e:
            raise RuntimeError(e)
    return settings





