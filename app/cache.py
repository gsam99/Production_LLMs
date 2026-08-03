import time
from app.config import get_settings
from upstash_redis import Redis
from langchain_openai import OpenAIEmbeddings
import chromadb


class CacheManager:

    def __init__(self):
        self.settings = get_settings()
        self.ttl_seconds = self.settings.cache_ttl_seconds
        self.redis = Redis(
            url=self.settings.upstash_redis_rest_url,
            token=self.settings.upstash_redis_rest_token
        )

        self.chroma_client = chromadb.PersistentClient(path="./chroma_cache")
        self.collection = self.chroma_client.get_or_create_collection(
            name="semantic_cache"
        )
        self.embedder = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.settings.openai_api_key
        )

        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0
        }

    def get_embeddings(self, text):
        return self.embedder.embed_query(text)

    def set(self, question, answer):
        timestamp = time.time()
        self.redis.setex(question, self.ttl_seconds, answer)

        embeddings = self.get_embeddings(question)
        self.collection.add(
            ids=[question],
            embeddings=[embeddings],
            documents=[question],
            metadatas=[{"answer": answer, "timestamp": timestamp}]
        )

    def delete_question(self, question):
        self.collection.delete(ids=[question])
        self.redis.delete(question)

    def get_exact_match(self, question):
        cached = self.redis.get(question)
        if cached is None:
            return None
        return cached if isinstance(cached, str) else cached.decode("utf-8")

    def get_answer(self, question):
        asked_timestamp = time.time()
        exact_answer = self.get_exact_match(question)
        if exact_answer is not None:
            self.stats["exact_hits"] += 1
            return exact_answer, "exact_hit"

        embedding = self.get_embeddings(question)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1
        )

        if not results["ids"][0]:
            self.stats["misses"] += 1
            return None, "miss"

        distance = results["distances"][0][0]
        metadata = results["metadatas"][0][0]
        matched_question = results["ids"][0][0]

        SIMILARITY_THRESHOLD = 0.2

        if distance > SIMILARITY_THRESHOLD:
            self.stats["misses"] += 1
            return None, "miss"

        age = asked_timestamp - metadata["timestamp"]
        if age > self.ttl_seconds:
            self.delete_question(matched_question)
            self.stats["misses"] += 1
            return None, "miss"

        self.stats["semantic_hits"] += 1
        return metadata["answer"], "semantic_hit"

    def get_stats(self):
        total = (
            self.stats["exact_hits"]
            + self.stats["semantic_hits"]
            + self.stats["misses"]
        )
        if total == 0:
            return {"hit_rate": 0.0, "total_requests": 0, **self.stats}

        hits = self.stats["exact_hits"] + self.stats["semantic_hits"]
        hit_rate = round((hits / total) * 100, 2)

        return {
            "hit_rate": hit_rate,
            "total_requests": total,
            **self.stats
        }

    def reset_stats(self):
        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0
        }