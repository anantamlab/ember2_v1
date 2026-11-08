"""Independent scenario generator and leakage-safe split manifests (P1-02, issue #6).

CPU only, standard library only. Derived from general evacuation factors, not
from any benchmark's scenarios, results or code. Design and evidence:
docs/P1_SCENARIO_GENERATOR.md.
"""
from .generator import (Family, GenerationError, Generated, GeneratorBug, build_family, build_lineage,
                        generate_family, generate_lineage, provenance)
from .spec import GENERATOR_REVISION, GeneratorSpec

__all__ = ["Family", "GenerationError", "Generated", "GeneratorBug", "build_family", "build_lineage",
           "generate_family", "generate_lineage", "provenance", "GENERATOR_REVISION", "GeneratorSpec"]
