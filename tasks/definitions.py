from openenv.core.env_server.types import TaskDefinition

TASK_DEFINITIONS = [
    TaskDefinition(
        id="cargo_food",
        name="cargo_food",
        description="Bilateral food-compliance screening for an agricultural shipment.",
        difficulty="easy",
        grader="deterministic_programmatic",
        pass_score=0.70,
    ),
    TaskDefinition(
        id="cargo_electronics",
        name="cargo_electronics",
        description="Export-control obligations for a dual-use electronics shipment.",
        difficulty="medium",
        grader="deterministic_programmatic",
        pass_score=0.78,
    ),
    TaskDefinition(
        id="cargo_pharma",
        name="cargo_pharma",
        description="Pharmaceutical API compliance across both jurisdictions.",
        difficulty="hard",
        grader="deterministic_programmatic",
        pass_score=0.85,
    ),
]