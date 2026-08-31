"""
Hugging Face zero-shot classification using BART-MNLI.

Why zero-shot classification?
- We have no labeled training data ($0 budget, no annotation effort)
- MNLI (Multi-Genre NLI) is trained to judge entailment between premise + hypothesis
- "This article is about monetary policy" is just a hypothesis; article text is premise
- Model scores how likely hypothesis is entailed by premise (0-1 confidence)

Why these labels?
- Interviewed traders about what moves their portfolios
- event_type: ["monetary policy", "inflation data", "earnings", "geopolitics", "liquidity/credit", "regulation", "other"]
- risk_level: ["low", "medium", "high"]
- These are orthogonal: "Fed hikes rates" = monetary policy + medium risk; "Tech earnings beat" = earnings + low risk

Tradeoff: Zero-shot has lower accuracy than fine-tuning, but no data cost.
Confidence threshold (e.g., >0.5) catches high-confidence predictions.
Low confidence (<0.3) means "uncertain"; downstream can ignore or escalate to human.

Why not multi-class/multi-label?
- Simple: run two independent classifiers (event_type, risk_level), each scores all labels
- Model outputs confidence per label; we take argmax for simplicity (L5 agents can refine)
- Could use structured output (e.g., JSON schema), but that's overkill for MVP
"""

import json
from typing import Optional

from src.hf_tasks.client import HFClient, HFCache
from src.logging_utils import get_request_logger

logger = get_request_logger(__name__)


class Classifier(HFClient):
    """Zero-shot classification client using BART-MNLI."""

    DEFAULT_MODEL = "facebook/bart-large-mnli"

    # Labels for classifying article type
    EVENT_TYPES = [
        "monetary policy",
        "inflation data",
        "earnings",
        "geopolitics",
        "liquidity/credit",
        "regulation",
        "other",
    ]

    # Labels for risk assessment
    RISK_LEVELS = ["low", "medium", "high"]

    def __init__(self, model_id: Optional[str] = None, cache: Optional[HFCache] = None):
        """
        Initialize classifier.

        Args:
            model_id: HF model ID. If None, uses DEFAULT_MODEL (BART-MNLI).
            cache: Shared HFCache instance (optional)
        """
        model_id = model_id or self.DEFAULT_MODEL
        super().__init__(model_id, cache)

    def _call_model(
        self,
        input_text: str,
        labels: list[str],
        hypothesis_template: str = "This text is about {}.",
        **kwargs
    ) -> Optional[str]:
        """
        Call zero-shot classification API.

        Args:
            input_text: Text to classify (article title + summary)
            labels: Candidate labels to score
            hypothesis_template: Template for converting label to hypothesis
                Default: "This text is about {}."
            **kwargs: Ignored (for compatibility with base class)

        Returns:
            JSON string: {"scores": [...], "labels": [...], "sequence": "..."}
            Or None if API call fails
        """
        # Truncate very long text to avoid token limits
        truncated = input_text[:512]

        try:
            result = self.client.zero_shot_classification(
                truncated,
                labels,
                hypothesis_template=hypothesis_template,
                multi_class=False,  # One label per classification task
            )

            # HF returns dict with keys: "scores", "labels", "sequence"
            # Scores are probabilities (sum to 1.0)
            return json.dumps(result)

        except Exception as e:
            logger.error(f"Classification API call failed: {e}")
            return None

    @classmethod
    def classify_event_type(
        cls, text: str, model_id: Optional[str] = None
    ) -> tuple[Optional[str], Optional[float]]:
        """
        Classify article as one of the event types.

        Args:
            text: Article text (title + summary)
            model_id: Optional model override

        Returns:
            Tuple of (event_type, confidence) or (None, None) if classification fails
        """
        client = cls(model_id=model_id)
        result = client(
            text,
            labels=cls.EVENT_TYPES,
            hypothesis_template="This article is about {}.",
        )

        if result:
            try:
                parsed = json.loads(result)
                # Model returns scores sorted by probability (descending)
                # We take the highest-confidence label
                if parsed.get("labels") and parsed.get("scores"):
                    label = parsed["labels"][0]
                    score = float(parsed["scores"][0])
                    return label, score
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                logger.warning(f"Failed to parse classification result: {e}")

        return None, None

    @classmethod
    def classify_risk_level(
        cls, text: str, model_id: Optional[str] = None
    ) -> tuple[Optional[str], Optional[float]]:
        """
        Classify article risk level.

        Args:
            text: Article text (title + summary)
            model_id: Optional model override

        Returns:
            Tuple of (risk_level, confidence) or (None, None) if classification fails
        """
        client = cls(model_id=model_id)
        result = client(
            text,
            labels=cls.RISK_LEVELS,
            hypothesis_template="This article describes a {} risk to financial markets.",
        )

        if result:
            try:
                parsed = json.loads(result)
                if parsed.get("labels") and parsed.get("scores"):
                    label = parsed["labels"][0]
                    score = float(parsed["scores"][0])
                    return label, score
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                logger.warning(f"Failed to parse classification result: {e}")

        return None, None

    @classmethod
    def classify_batch(
        cls,
        articles: list[dict],
        model_id: Optional[str] = None,
        skip_existing: bool = True,
    ) -> list[dict]:
        """
        Classify multiple articles in batch.

        Args:
            articles: List of dicts with 'title' and 'summary' fields
            model_id: Optional model override
            skip_existing: If True, skip articles that already have 'event_type' and 'risk_level'

        Returns:
            Same articles list, with 'event_type', 'risk_level', 'risk_score' fields filled
        """
        for i, article in enumerate(articles):
            # Skip if already classified
            if skip_existing and article.get("event_type") and article.get("risk_level"):
                logger.debug(f"[{i}/{len(articles)}] Skipping (already classified)")
                continue

            # Prepare text for classification: title + summary
            title = article.get("title", "")
            summary = article.get("summary", "")
            text = f"{title}. {summary}"

            if not text.strip():
                logger.warning(f"[{i}/{len(articles)}] No text to classify")
                article["event_type"] = "other"
                article["risk_level"] = "low"
                article["risk_score"] = 0.5
                continue

            logger.info(f"[{i}/{len(articles)}] Classifying: {title[:50]}...")

            # Classify event type
            event_type, event_conf = cls.classify_event_type(text, model_id=model_id)
            if event_type and event_conf:
                article["event_type"] = event_type
            else:
                article["event_type"] = "other"
                logger.warning(f"Failed to classify event_type for: {title[:50]}")

            # Classify risk level
            risk_level, risk_conf = cls.classify_risk_level(text, model_id=model_id)
            if risk_level and risk_conf:
                article["risk_level"] = risk_level
                article["risk_score"] = float(risk_conf)
            else:
                article["risk_level"] = "low"
                article["risk_score"] = 0.5
                logger.warning(f"Failed to classify risk_level for: {title[:50]}")

        logger.info(f"✓ Classified {len(articles)} articles")
        return articles
