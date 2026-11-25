"""Contrôle du workflow d'intégration continue (.github/workflows/tests.yml), avant tout push.

Une erreur d'indentation dans le YAML ne se verrait sinon que sur GitHub, après le push.
"""

import re

import pytest
import yaml

from rag import config

WORKFLOW_FILE = config.PROJECT_ROOT / ".github" / "workflows" / "tests.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW_FILE.read_text(encoding="utf-8"))


def triggers(workflow: dict) -> dict:
    """Déclencheurs du workflow. PyYAML lit la clé « on » comme le booléen True (YAML 1.1)."""
    return workflow.get("on", workflow.get(True))


def commands(job: dict) -> list[str]:
    return [step["run"] for step in job["steps"] if "run" in step]


def test_the_workflow_runs_on_every_push_and_pull_request(workflow):
    assert {"push", "pull_request"} <= set(triggers(workflow))


def test_the_tests_run_on_linux_and_windows(workflow):
    job = workflow["jobs"]["tests"]
    assert {"ubuntu-latest", "windows-latest"} <= set(job["strategy"]["matrix"]["os"])
    assert job["strategy"]["fail-fast"] is False        # un échec sous Windows n'interrompt pas Linux
    assert any("pip install -r requirements-dev.txt" in command for command in commands(job))
    assert any(command.startswith("python -m pytest") for command in commands(job))


def test_the_ci_does_not_hide_encoding_problems(workflow):
    """Sous Windows, l'encodage par défaut n'est pas UTF-8 : forcer UTF-8 masquerait un oubli d'encoding=."""
    assert "PYTHONUTF8" not in WORKFLOW_FILE.read_text(encoding="utf-8")


def test_the_ci_uses_the_same_python_as_the_docker_image(workflow):
    dockerfile = (config.PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    docker_python = re.search(r"^FROM python:(\d+\.\d+)", dockerfile, re.MULTILINE).group(1)
    setup = next(step for step in workflow["jobs"]["tests"]["steps"]
                 if step.get("uses", "").startswith("actions/setup-python"))
    assert str(setup["with"]["python-version"]) == docker_python


def test_the_docker_image_is_built(workflow):
    assert any(command.startswith("docker build") for command in commands(workflow["jobs"]["docker"]))
