import re
from typing import Dict, Any, List, Optional, Tuple
from router_engine.models.catalog import ModelCatalog, LLMModelProfile

class ProfileMatcher:
    """
    Evaluates queries against market LLM profiles to determine
    the most suitable model and provider based on capabilities, latency, and cost.
    """
    def __init__(self, catalog: Optional[ModelCatalog] = None):
        self.catalog = catalog or ModelCatalog()

    def evaluate(self, query: str, domain: str = "general") -> Dict[str, Any]:
        """
        Scores all candidate models for a given query and selects the optimal one.
        Returns:
            {
                "optimal_model": str,
                "provider": str,
                "selection_reason": str,
                "model_profile": dict,
                "candidate_scores": dict
            }
        """
        q_lower = query.lower()
        q_len = len(query)

        # Helper for word boundary regex matching to avoid substring false positives (e.g. 'ratio' in 'migration')
        def has_keywords(text: str, words: List[str]) -> bool:
            pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
            return bool(re.search(pattern, text, re.IGNORECASE))

        # 1. Feature Detection
        has_visual_keywords = (
            domain in ["data_analytics", "visual_reporting", "multimodal_vision", "fast_vision"]
            or has_keywords(query, [
                "image", "visual", "dashboard", "chart", "graph", "trend", "report", "diagram", "screenshot", "video", "ocr"
            ])
        )
        is_math_or_stem = (
            domain in ["metrics_reasoning", "stem_reasoning"]
            or has_keywords(query, [
                "calculate", "solve", "equation", "formula", "ratio", "algebra", "arithmetic", "probability", "integral"
            ])
        )
        is_support_or_sentiment = (
            domain in ["customer_sentiment", "customer_support"]
            or has_keywords(query, [
                "refund", "order", "shipping", "discount", "complaint", "dissatisfied", "support inquiry", "cancel subscription", "tracking"
            ])
        )
        is_deep_strategy_or_architecture = (
            (q_len > 250 or domain in ["business_strategy", "marketing_copywriting"] or has_keywords(query, [
                "strategy", "repositioning", "architecture", "tradeoff", "roadmap", "proposal", "persuasive", "framework"
            ]))
            and not is_math_or_stem
        )
        is_lightweight = (
            q_len < 100
            and not is_math_or_stem
            and not is_deep_strategy_or_architecture
            and not has_visual_keywords
            and not is_support_or_sentiment
        )

        candidate_scores: Dict[str, Dict[str, Any]] = {}

        for model in self.catalog.get_all():
            # Base Capability Fit (0.0 to 1.0)
            fit_score = 0.50

            # Domain & Strengths Alignment
            if domain in model.best_domains:
                fit_score += 0.30

            # Visual / Multimodal Query
            if has_visual_keywords:
                if model.multimodal and "analytics_dashboards" in model.strengths:
                    fit_score += 0.45
                elif model.multimodal:
                    fit_score += 0.35
                else:
                    fit_score -= 0.40

            # Math / STEM Query
            if is_math_or_stem:
                if "mathematics" in model.strengths or "stem_reasoning" in model.strengths:
                    fit_score += 0.45
                elif model.cost_tier == "premium":
                    fit_score += 0.15
                else:
                    fit_score -= 0.25

            # Customer Support / Short Sentiment Query
            if is_support_or_sentiment:
                if "customer_support" in model.strengths or "sentiment_analysis" in model.strengths:
                    fit_score += 0.45
                if model.latency_tier == "ultra_fast":
                    fit_score += 0.20
                if model.cost_tier == "premium":
                    fit_score -= 0.35  # Strong penalty against wasting expensive models on simple support

            # Deep Strategy / Complex Long-context Query
            if is_deep_strategy_or_architecture:
                if "deep_strategy" in model.strengths or "long_context_analysis" in model.strengths:
                    fit_score += 0.50
                if model.context_window_adequacy(q_len):
                    fit_score += 0.15
                if model.cost_tier == "economy":
                    fit_score -= 0.35  # Economy models struggle on deep strategic synthesis

            # Lightweight Simple Query
            if is_lightweight:
                if model.cost_tier == "economy" and model.latency_tier == "ultra_fast":
                    fit_score += 0.45
                elif model.cost_tier == "premium":
                    fit_score -= 0.40

            # Clamp fit score between 0.05 and 1.0
            fit_score = max(0.05, min(1.0, fit_score))

            # Economic & Latency Utility:
            # Normalize cost: input cost per 1M tokens scaled
            cost_factor = model.input_cost_per_million / 3.0  # normalized to 0.0 - 1.0
            latency_factor = model.typical_latency_ms / 1000.0  # normalized

            # Overall Utility Score
            utility = fit_score - (0.15 * cost_factor) - (0.10 * latency_factor)

            candidate_scores[model.id] = {
                "fit_score": round(fit_score, 4),
                "utility": round(utility, 4),
                "cost_tier": model.cost_tier,
                "latency_ms": model.typical_latency_ms,
                "provider": model.provider
            }

        # Select model with highest utility
        best_model_id = max(candidate_scores, key=lambda m_id: candidate_scores[m_id]["utility"])
        best_model = self.catalog.get(best_model_id)

        reason = self._build_selection_reason(
            best_model, has_visual_keywords, is_math_or_stem,
            is_support_or_sentiment, is_deep_strategy_or_architecture, is_lightweight, q_len
        )

        return {
            "optimal_model": best_model.id,
            "provider": best_model.provider,
            "selection_reason": reason,
            "model_profile": best_model.to_dict(),
            "candidate_scores": candidate_scores
        }

    def _build_selection_reason(
        self,
        model: LLMModelProfile,
        visual: bool,
        math: bool,
        support: bool,
        strategy: bool,
        lightweight: bool,
        q_len: int
    ) -> str:
        if visual:
            return (
                f"Selected {model.name} ({model.provider}): Optimized for multimodal analytics, "
                f"dashboard metrics, and visual reporting with {model.latency_tier} latency."
            )
        elif math:
            return (
                f"Selected {model.name} ({model.provider}): Superior algorithmic STEM and mathematical "
                f"reasoning capability required for metric word problems."
            )
        elif support:
            return (
                f"Selected {model.name} ({model.provider}): Optimal cost-efficiency (${model.input_cost_per_million}/1M tokens) "
                f"and ultra-fast response ({model.typical_latency_ms}ms) for customer dialogue/sentiment."
            )
        elif strategy:
            return (
                f"Selected {model.name} ({model.provider}): Specialized in multi-step reasoning, architectural "
                f"planning, and long-context strategic copywriting ({q_len} chars)."
            )
        elif lightweight:
            return (
                f"Selected {model.name} ({model.provider}): Low-latency lightweight execution with lowest cost overhead."
            )
        else:
            return (
                f"Selected {model.name} ({model.provider}): Highest overall capability-to-cost utility match."
            )


# Helper method on LLMModelProfile for context check
def _context_adequacy(self: LLMModelProfile, char_len: int) -> bool:
    estimated_tokens = char_len // 4
    return self.max_input_tokens >= (estimated_tokens + 2048)

LLMModelProfile.context_window_adequacy = _context_adequacy
