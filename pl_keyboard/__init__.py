"""Pure, dependency-light library for building a Polish FUTO Keyboard LM.

No heavy ML imports (torch / transformers / datasets) at module load time — that
keeps the whole library fast and 100% unit-testable. Heavy work lives behind the
CLI layer and injected interfaces.
"""

__version__ = "0.0.1"
