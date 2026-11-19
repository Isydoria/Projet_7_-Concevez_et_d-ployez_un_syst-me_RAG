"""Contrôle des dépendances d'évaluation : Ragas doit s'importer avec les versions figées du projet.

Ragas 0.4.3 importait un module supprimé de langchain-community 0.4 (`chat_models.vertexai`) :
`import ragas` échouait, sans qu'aucun autre test ne le signale. Corrigé dans Ragas 0.4.4.
Ce test repère ce type d'incompatibilité dès l'installation, avant le lancement de l'évaluation.
"""


def test_ragas_and_its_metrics_import():
    from ragas import EvaluationDataset, evaluate  # noqa: F401
    from ragas.metrics.collections import (  # noqa: F401
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
    )
