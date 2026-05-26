"""Config validation schemas for T-ChatKBQA.

Pydantic models that validate all YAML configs before launching jobs.
Catches typos, missing fields, and invalid values early.

Usage:
    python configs/schema.py                    # Validate all configs
    python configs/schema.py --config <path>     # Validate a specific file
    python configs/schema.py --quiet             # Only output errors
"""

from __future__ import annotations

import argparse
import sys
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional

import yaml

try:
    from pydantic import BaseModel, Field, field_validator
    from pydantic import ValidationInfo
except ImportError:
    print("ERROR: pydantic>=2.0 is required. Run: pip install pydantic")
    sys.exit(1)


# ── Enums ────────────────────────────────────────────────────────────────

class TemplateFamily(str, Enum):
    simple = "simple"
    during = "during"
    first = "first"
    last = "last"
    before = "before"
    after = "after"


class TemporalFieldType(str, Enum):
    date_range = "date_range"
    date_point = "date_point"
    ordering = "ordering"


class AnswerType(str, Enum):
    entity = "entity"
    date = "date"
    literal = "literal"


class DatasetSplit(str, Enum):
    train = "train"
    test = "test"
    validation = "validation"


# ── Relation Whitelist ───────────────────────────────────────────────────

class FamilyConfig(BaseModel):
    """A single relation family entry in relation_whitelist.yaml."""
    family: str = Field(..., min_length=1, description="Short identifier")
    anchor_relation: str = Field(..., min_length=1, description="Freebase CVT anchor relation (dot format)")
    answer_relation: str = Field(..., min_length=1, description="Points to answer entity within CVT")
    temporal_field: str = Field(..., min_length=1, description="CVT property holding date/ordering info")
    temporal_field_type: TemporalFieldType
    second_temporal: Optional[str] = Field(None, description="Optional second temporal field (date_range)")
    answer_type: AnswerType
    topic_label_hint: str = Field("", description="Human-readable topic role name")
    answer_label_hint: str = Field("", description="Human-readable answer role name")
    supported_templates: List[TemplateFamily] = Field(..., min_length=1)

    @field_validator("anchor_relation", "answer_relation", "temporal_field")
    @classmethod
    def dot_format(cls, v: str) -> str:
        if "/" in v and "." not in v:
            raise ValueError(f"Use dot-format for relations: '{v}' -> use '.' not '/'")
        parts = v.split(".")
        if len(parts) < 2:
            raise ValueError(f"Relation must have at least domain.type format: '{v}'")
        if any(not p or " " in p for p in parts):
            raise ValueError(f"Relation segments must be non-empty and space-free: '{v}'")
        return v

    @field_validator("supported_templates")
    @classmethod
    def compatible_templates(cls, v, info: ValidationInfo):
        """Template families must be compatible with temporal_field_type."""
        tft = info.data.get("temporal_field_type") if info.data else None
        if tft and tft == TemporalFieldType.ordering and any(
            t in v for t in (TemplateFamily.before, TemplateFamily.after)
        ):
            raise ValueError(
                f"before/after templates require date fields, not '{tft.value}'"
            )
        return v


class RelationWhitelist(BaseModel):
    """Validation schema for configs/relation_whitelist.yaml."""
    families: List[FamilyConfig] = Field(..., min_length=1)
    limits: dict = Field(default_factory=lambda: {
        "max_facts_per_family": 200,
        "sample_date_min_year": 1800,
        "sample_date_max_year": 2025,
    })

    @field_validator("families")
    @classmethod
    def unique_families(cls, v):
        names = [f.family for f in v]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate family names: {dupes}")
        return v


# ── Training Config ──────────────────────────────────────────────────────

class TrainConfig(BaseModel):
    """Validation schema for training YAML configs."""
    stage: Literal["sft", "pt", "rm", "ppo", "dpo"] = "sft"
    model_name_or_path: str = Field(..., min_length=1, description="HF model or local path")
    do_train: bool = True
    dataset_dir: str = Field(default="LLMs/data")
    dataset: str = Field(..., min_length=1, description="Dataset name in dataset_info.json")
    template: Literal["llama2", "alpaca", "vicuna", "chatglm2", "qwen"] = "llama2"
    finetuning_type: Literal["lora", "full", "freeze"] = "lora"
    lora_target: str = Field(default="q_proj,v_proj")
    output_dir: str = Field(..., min_length=1, description="Checkpoint output directory")
    per_device_train_batch_size: int = Field(default=4, ge=1, le=256)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
    lr_scheduler_type: Literal["cosine", "linear", "constant", "polynomial"] = "cosine"
    learning_rate: float = Field(default=5e-5, gt=0, le=1e-2)
    num_train_epochs: float = Field(default=50.0, gt=0, le=1000.0)
    fp16: bool = True
    logging_steps: Optional[int] = Field(default=10, ge=1)
    save_steps: Optional[int] = Field(default=500, ge=1)
    warmup_ratio: float = Field(default=0.05, ge=0, le=1.0)
    max_samples: Optional[int] = Field(default=None, ge=1)

    @field_validator("output_dir")
    @classmethod
    def writable_output_dir(cls, v):
        """Output dir doesn't need to exist yet, but path must be valid."""
        try:
            Path(v)
        except Exception:
            raise ValueError(f"Invalid output_dir path: {v}")
        return v

    @field_validator("model_name_or_path")
    @classmethod
    def model_path_format(cls, v):
        if v.startswith("/") and not Path(v).exists():
            print(f"  [WARN] Model path does not exist locally: {v}")
        return v


# ── Inference Config ─────────────────────────────────────────────────────

class InferenceConfig(BaseModel):
    """Validation schema for inference YAML configs."""
    model_path: str = Field(..., min_length=1, description="Path to fine-tuned checkpoint")
    entity_surface_index_path: Optional[str] = Field(None, description="FACC1 surface index")
    beam_size: int = Field(default=15, ge=1, le=100)
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    max_retries: int = Field(default=2, ge=0, le=10)
    eval_dataset: str = Field(default="TempQuestions")
    eval_split: Literal["train", "test", "validation"] = "test"


# ── Discovery & CLI ──────────────────────────────────────────────────────

CONFIG_SCHEMAS = {
    "relation_whitelist.yaml": RelationWhitelist,
    "train_temporal.yaml": TrainConfig,
    "train_tchatkbqa.yaml": TrainConfig,
    "train_tchatkbqa_trial.yaml": TrainConfig,
    "inference.yaml": InferenceConfig,
}


def detect_config_type(path: Path) -> Optional[type]:
    """Detect which schema applies to a config file."""
    filename = path.name
    if filename in CONFIG_SCHEMAS:
        return CONFIG_SCHEMAS[filename]

    # Heuristic: if it's a train config, use TrainConfig
    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError:
            return None

    if data is None:
        return None

    if "stage" in data and data["stage"] in ("sft", "pt", "rm", "ppo", "dpo"):
        return TrainConfig
    if "families" in data and isinstance(data.get("families"), list):
        return RelationWhitelist
    if "beam_size" in data:
        return InferenceConfig

    return None


def validate_config(path: str, quiet: bool = False) -> bool:
    """Validate a single config file. Returns True if valid."""
    config_path = Path(path)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {path}")
        return False

    if not quiet:
        print(f"Validating: {config_path.name}")

    # Load YAML
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"  ERROR: Invalid YAML — {e}")
        return False

    if raw is None:
        print(f"  ERROR: Empty config file")
        return False

    # Detect or use explicit schema
    schema = detect_config_type(config_path)
    if schema is None:
        if not quiet:
            print(f"  [SKIP] Unknown config format, no schema matched")
        return True  # Not a failure — just unknown

    try:
        schema(**raw)
        if not quiet:
            print(f"  OK   ({schema.__name__}, {len(raw)} fields)")
        return True
    except Exception as e:
        print(f"  FAIL — {e}")
        return False


def validate_all(config_dir: str = "configs", quiet: bool = False) -> dict:
    """Validate all known configs. Returns {path: valid}."""
    results = {}
    config_root = Path(config_dir)

    if not config_root.is_dir():
        print(f"ERROR: configs directory not found: {config_dir}")
        return results

    for yaml_file in sorted(config_root.glob("*.yaml")):
        results[yaml_file.name] = validate_config(str(yaml_file), quiet=quiet)

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate T-ChatKBQA YAML configs")
    parser.add_argument("--config", default=None, help="Path to a single config file")
    parser.add_argument("--config-dir", default="configs", help="Directory of configs (default: configs/)")
    parser.add_argument("--quiet", action="store_true", help="Only show failures")
    args = parser.parse_args()

    if args.config:
        ok = validate_config(args.config, quiet=args.quiet)
        sys.exit(0 if ok else 1)

    results = validate_all(config_dir=args.config_dir, quiet=args.quiet)
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    skipped = sum(1 for v in results.values() if v is None)

    if not args.quiet:
        print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped (of {total})")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
