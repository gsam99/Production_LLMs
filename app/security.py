import re
from typing import Optional
from langsmith import traceable


class InputSanitizer:
    INJECTION_PATTERNS = [
        r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions?',
        r'you\s+are\s+now\s+a?\s*different',
        r'disregard\s+(all\s+)?(previous|prior)\s+',
        r'system\s+prompt',
        r'forget\s+(everything|all)\s+',
    ]

    def __init__(self):
        self.patterns=[
            re.compile(p,re.IGNORECASE)
            for p in self.INJECTION_PATTERNS
        ]

    def check(self,text:str)-> tuple[bool,Optional[str]]:
        for pattern in self.patterns:
            if pattern.search(text):
                return False, "Blocked: potential prompt injection"
        return True, None

    def clean(self, text: str) -> str:
        """Remove potentially dangerous delimiters from input."""
        text = re.sub(r"[-]{3,}", "", text)
        text = re.sub(r"[=]{3,}", "", text)
        text = text.replace("{{", "{ {").replace("}}", "} }")
        return text.strip()

class PIIDetector:
    """
    Detect and mask personally identifiable information.
    Works on BOTH input (before LLM) and output (before client).
    """

    PATTERNS = {
        "email": re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ),
        "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "credit_card": re.compile(
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        ),
    }

    REDACTION_MARKERS = {
        "email": "[EMAIL ADDRESS]",
        "phone": "[PHONE NUMBER]",
        "ssn": "[SSN]",
        "credit_card": "[CARD DETAILS]",
    }

    def detect(self, text: str) -> list[str]:
        """Return a list of PII types found in the text."""
        found = []
        for label, pattern in self.PATTERNS.items():
            if pattern.search(text):
                found.append(label)
        return found

    def mask(self, text: str) -> str:
        """Replace any detected PII with its redaction marker."""
        masked = text
        for label, pattern in self.PATTERNS.items():
            marker = self.REDACTION_MARKERS[label]
            masked = pattern.sub(marker, masked)
        return masked

class OutputValidator:
    """
    Validate LLM output before returning to the client.
    Catches PII leakage and harmful content in responses.
    """

    HARMFUL_PATTERNS = [
        re.compile(r"here('s| is) (how|the way) to (hack|steal|attack)", re.I),
        re.compile(r"password\s+is\s+", re.I),
        re.compile(r"api[_\s]?key\s*[:=]", re.I),
    ]

    def __init__(self):
        self.pii_detector = PIIDetector()

    def validate(self, output: str) -> tuple[str, list[str]]:
        """
        Validate and clean output.
        Returns: (cleaned_output, list_of_warnings)
        """
        warnings = []
        cleaned = output

        # Check for PII leakage in the LLM's response
        pii_found = self.pii_detector.detect(cleaned)
        if pii_found:
            warnings.append(f"PII detected in output: {', '.join(pii_found)}")
            cleaned = self.pii_detector.mask(cleaned)

        # Check for harmful content patterns
        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(cleaned):
                warnings.append(f"Harmful pattern detected: {pattern.pattern}")
                cleaned = pattern.sub("[CONTENT REMOVED]", cleaned)

        return cleaned, warnings

class SecurityPipeline:
    def __init__(self):
        self.input_sanitizer = InputSanitizer()
        self.output_validator = OutputValidator()
        self.pii_detector = PIIDetector()

    def process_input(self,text):
        safe,reason = self.input_sanitizer.check(text)
        if safe:
            clean= self.input_sanitizer.clean(text)
            PII_check=self.pii_detector.detect(clean)
            if PII_check:
                clean=self.pii_detector.mask(clean)
        else:
            return False, text, reason
        return True, clean, None

    def process_output(self,text):
        cleaned,warnings = self.output_validator.validate(text)
        return cleaned, warnings






