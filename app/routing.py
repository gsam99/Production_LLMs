from langchain_openai import ChatOpenAI
from app.config import get_settings



class ModelRouter:
    SIMPLE_PHRASES = ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay"]
    COMPLEX_KEYWORDS = [
        "analyze", "compare", "summarize in detail",
        "write a comprehensive", "step by step"
    ]

    def __init__(self):
        self.settings = get_settings()
        self.llm_classifier = ChatOpenAI(
            model=self.settings.primary_model,
            api_key=self.settings.openai_api_key
        )

    def initial_classification(self,text):
        text_clean = text.strip().lower()
        text_length = len(text)

        if text_clean in self.SIMPLE_PHRASES:
            return "simple"
        if text_length < 20:
            return "simple"
        if text_length > 500:
            return "complex"
        if any(keyword in text_clean for keyword in self.COMPLEX_KEYWORDS):
            return "complex"

        return "ambiguous"

    def llm_classify(self,text):
        prompt = (
            "Classify the following question as exactly one word: "
            "SIMPLE or COMPLEX. SIMPLE means a quick factual or casual "
            "question. COMPLEX means it needs deep reasoning, multi-step "
            "analysis, or detailed writing.\n\n"
            f"Question: {text}\n\nAnswer with one word only."
        )

        response=self.llm_classifier.invoke(prompt)
        result= response.content.strip().lower()

        if "simple" in result:
            return "simple"
        else:
            return "complex"

    def route(self,text):

        result=self.initial_classification(text)
        if result == "ambiguous":
            result = self.llm_classify(text)
        if result == "simple":
            return self.settings.primary_model
        return self.settings.complex_model




