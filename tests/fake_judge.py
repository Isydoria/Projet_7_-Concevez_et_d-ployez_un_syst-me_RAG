"""Faux juge et faux embeddings pour Ragas, partagés par les tests de l'évaluation (aucun appel à Mistral).

Module séparé de factories.py : il importe Ragas, lourd à charger, dont les autres tests n'ont pas besoin.
"""

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM


class FakeJudgeLLM(InstructorBaseRagasLLM):
    """Faux juge : deux affirmations, la première appuyée, la seconde non ; garde le format de chaque appel.

    `error` : exception levée au lieu de répondre, pendant les `failures` premiers appels (refus de Mistral simulé).
    """

    def __init__(self, error: Exception | None = None, failures: int = 0):
        self.calls, self.error, self.failures = [], error, failures

    def generate(self, prompt, response_model):
        raise NotImplementedError("les métriques appellent agenerate")

    async def agenerate(self, prompt, response_model):
        name = response_model.__name__
        self.calls.append(name)
        if self.error is not None and self.failures > 0:
            self.failures -= 1
            raise self.error
        verdicts = [{"statement": "A", "reason": "appuyée", "verdict": 1},
                    {"statement": "B", "reason": "absente des événements", "verdict": 0}]
        replies = {
            "StatementGeneratorOutput": {"statements": ["A", "B"]},
            "NLIStatementOutput": {"statements": verdicts},
            "AnswerRelevanceOutput": {"question": "Un concert de jazz ?", "noncommittal": 0},
            "ContextRecallOutput": {"classifications": [{"statement": "A", "reason": "trouvée", "attributed": 1},
                                                        {"statement": "B", "reason": "absente", "attributed": 0}]},
            "ClaimDecompositionOutput": {"claims": ["A", "B"]},
        }
        return response_model(**replies[name])


class FakeEmbeddings(BaseRagasEmbedding):
    """Faux embeddings : le même vecteur pour tout texte (similarité 1)."""

    def embed_text(self, text, **kwargs):
        return [1.0, 0.0]

    async def aembed_text(self, text, **kwargs):
        return [1.0, 0.0]
