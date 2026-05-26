"""End-to-end inference pipeline for Temporal KBQA.

Wraps the full ChatKBQA generate-then-retrieve flow, adding temporal-aware
LLM generation and timestamp retrieval.

Usage:
    from src.pipeline import TemporalKBQAPipeline
    pipeline = TemporalKBQAPipeline.from_config('configs/inference.yaml')
    result = pipeline.answer("Who was US president before Obama?")
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import TEMPORAL_INSTRUCTION, STANDARD_INSTRUCTION


def _load_yaml_config(config_path: str) -> dict:
    """Load YAML config, falling back to JSON if PyYAML not available."""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        with open(config_path) as f:
            return json.load(f)


class TemporalKBQAPipeline:
    """End-to-end pipeline: question → answer via generate-then-retrieve.

    Two modes:
      'temporal' — LLM generates S-expression with TC/ordinal operators;
                   retrieval also handles timestamp binding.
      'standard' — Standard ChatKBQA pipeline without temporal extensions.
    """

    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        entity_list_path: Optional[str] = None,
        entity_surface_index_path: Optional[str] = None,
        relation_index_path: Optional[str] = None,
        beam_size: int = 15,
        device: str = 'cuda',
    ):
        self.model_path = model_path
        self.base_model_path = base_model_path or os.environ.get('TKBQA_BASE_MODEL_PATH')
        self.entity_list_path = entity_list_path
        self.entity_surface_index_path = entity_surface_index_path
        self.beam_size = beam_size
        self.device = device
        self._llm = None
        self._tokenizer = None
        self._surface_index = None
        self._simcse_model = None
        self._surface_index_error = None
        self._llm_error = None

        if entity_surface_index_path and os.path.exists(entity_surface_index_path):
            self._load_surface_index(entity_surface_index_path)

    @classmethod
    def from_config(cls, config_path: str) -> 'TemporalKBQAPipeline':
        cfg = _load_yaml_config(config_path)
        return cls(
            model_path=cfg['model_path'],
            base_model_path=cfg.get('base_model_path'),
            entity_list_path=cfg.get('entity_list_path'),
            entity_surface_index_path=cfg.get('entity_surface_index_path'),
            relation_index_path=cfg.get('relation_index_path'),
            beam_size=cfg.get('beam_size', 15),
            device=cfg.get('device', 'cuda'),
        )

    def _load_surface_index(self, path: str):
        """Load FACC1 surface index for entity retrieval."""
        from entity_retrieval import surface_index_memory
        entity_list_path = self.entity_list_path or self._infer_entity_list_path(path)
        if not entity_list_path or not os.path.exists(entity_list_path):
            self._surface_index_error = (
                f"Entity list file not found for surface index: {entity_list_path}"
            )
            return
        try:
            self._surface_index = surface_index_memory.EntitySurfaceIndexMemory(
                entity_list_path,
                path,
                path,
            )
        except Exception as e:
            self._surface_index_error = str(e)
            self._surface_index = None

    def _infer_entity_list_path(self, surface_map_path: str) -> Optional[str]:
        candidate = surface_map_path.replace('surface_map_file', 'entity_list_file')
        if candidate != surface_map_path:
            return candidate
        surface_path = Path(surface_map_path)
        sibling = surface_path.with_name(
            surface_path.name.replace('surface_map', 'entity_list')
        )
        return str(sibling)

    def _resolve_model_path(self) -> str:
        if os.path.exists(self.model_path):
            return self.model_path

        candidates = [
            'models/LLaMA2-7b-tchatkbqa-trial/checkpoint',
            'models/LLaMA2-7b-temporal/checkpoint',
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return self.model_path

    def _resolved_entity_list_path(self) -> Optional[str]:
        if self.entity_list_path:
            return self.entity_list_path
        return None

    def _has_adapter_only_checkpoint(self, model_path: str) -> bool:
        return os.path.exists(os.path.join(model_path, 'adapter_config.json'))

    def runtime_status(self) -> dict:
        resolved_model_path = self._resolve_model_path()
        resolved_entity_list_path = self._resolved_entity_list_path()
        adapter_only = self._has_adapter_only_checkpoint(resolved_model_path)
        base_model_exists = bool(self.base_model_path and os.path.exists(self.base_model_path))
        return {
            'model_path': resolved_model_path,
            'model_exists': os.path.exists(resolved_model_path),
            'adapter_only_checkpoint': adapter_only,
            'surface_index_loaded': self._surface_index is not None,
            'surface_index_error': self._surface_index_error,
            'llm_loaded': self._llm is not None,
            'llm_error': self._llm_error,
            'base_model_path': self.base_model_path,
            'base_model_exists': base_model_exists,
            'entity_list_path': resolved_entity_list_path,
            'entity_list_exists': bool(
                resolved_entity_list_path and os.path.exists(resolved_entity_list_path)
            ),
            'surface_map_path': self.entity_surface_index_path,
            'surface_map_exists': bool(
                self.entity_surface_index_path and os.path.exists(self.entity_surface_index_path)
            ),
        }

    def _load_llm(self):
        """Lazy-load the fine-tuned LLM."""
        if self._llm is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            resolved_model_path = self._resolve_model_path()
            print(f"Loading LLM from {resolved_model_path}...")

            adapter_config_path = os.path.join(resolved_model_path, 'adapter_config.json')
            torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
            if os.path.exists(adapter_config_path):
                from peft import AutoPeftModelForCausalLM
                self._llm = AutoPeftModelForCausalLM.from_pretrained(
                    resolved_model_path,
                    torch_dtype=torch_dtype,
                    device_map='auto',
                )
            else:
                self._llm = AutoModelForCausalLM.from_pretrained(
                    resolved_model_path,
                    torch_dtype=torch_dtype,
                    device_map='auto',
                )
            print("LLM loaded.")
        except Exception as e:
            print(f"Warning: Could not load LLM ({e}). Using placeholder generation.")
            self._llm_error = str(e)
            self._llm = None

    def _generate_candidate_sexprs(self, question: str, mode: str) -> list[str]:
        """Generate candidate S-expressions using the fine-tuned LLM.

        In temporal mode, the instruction explicitly mentions TC operators.
        """
        if mode == 'temporal':
            instruction = TEMPORAL_INSTRUCTION
        else:
            instruction = STANDARD_INSTRUCTION

        prompt = f"{instruction}Question: {{ {question} }}"

        self._load_llm()
        if self._llm is None:
            # Fallback placeholder when LLM is not loaded
            return [f"(JOIN (R relation.placeholder) entity.placeholder)"]

        import torch
        inputs = self._tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self._llm.generate(
                **inputs,
                num_beams=self.beam_size,
                num_return_sequences=min(self.beam_size, 5),
                max_new_tokens=256,
                early_stopping=True,
            )
        candidates = [
            self._tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]
        # Strip echoed prompt prefix
        candidates = [c.replace(prompt, '').strip() for c in candidates]
        return candidates

    def _retrieve_and_ground(
        self, candidates: list[str], question: str, relaxed: bool = False
    ) -> tuple[list[str], Optional[str], Optional[str]]:
        """Replace entity/relation placeholders with Freebase IDs.

        Returns (answer_list, sparql_used, temporal_constraint_used).
        """
        from executor.logic_form_util import lisp_to_sparql
        from executor.sparql_executor import execute_query_with_odbc
        entity_label_map = {}
        type_label_map = {}

        for candidate in candidates:
            try:
                # Attempt direct execution first
                sparql = lisp_to_sparql(candidate)
                results = execute_query_with_odbc(sparql)
                if results:
                    constraint = self._extract_temporal_constraint(candidate)
                    if constraint is None and not relaxed:
                        constraint = self._bind_temporal_scope(candidate)
                    return list(results), sparql, constraint
            except Exception:
                pass

            # Try retrieval-grounded approach (requires surface index)
            if self._surface_index is None:
                continue

            try:
                from eval_final import denormalize_s_expr_new

                grounded_list = denormalize_s_expr_new(
                    candidate,
                    entity_label_map,
                    type_label_map,
                    self._surface_index,
                )
                # denormalize_s_expr_new returns a list of denormalized expressions
                if not grounded_list or grounded_list == 'null':
                    continue

                for grounded in grounded_list:
                    if not grounded or grounded == 'null':
                        continue
                    try:
                        normalized = grounded.replace('( ', '(').replace(' )', ')')
                        sparql = lisp_to_sparql(normalized)
                        results = execute_query_with_odbc(sparql)
                        if results:
                            constraint = self._extract_temporal_constraint(normalized)
                            if constraint is None and not relaxed:
                                constraint = self._bind_temporal_scope(normalized)
                            return list(results), sparql, constraint
                    except Exception:
                        continue
            except Exception:
                pass

        return [], None, None

    def _bind_temporal_scope(self, s_expr: str) -> Optional[str]:
        """Use get_temporal_scope() to bind actual CVT timestamps for TC operators.

        Parses the S-expression for entity and relation references, then queries
        Freebase for the temporal scope (from/to dates) of that entity-relation pair.
        """
        try:
            from executor.sparql_executor import get_temporal_scope
        except ImportError:
            return None

        # Extract entity MID and relation from TC S-expression
        # Pattern: (TC (JOIN ...) some.relation.from some_date)
        tc_match = re.search(
            r'TC\s+\([^)]+\)\s+(\S+)\s+(\S+)', s_expr
        )
        if not tc_match:
            return None

        relation = tc_match.group(1)
        time_point = tc_match.group(2)

        # Extract entity MID from the S-expression
        entity_match = re.search(r'(m\.[a-zA-Z0-9_]+)', s_expr)
        if not entity_match:
            return None

        entity = entity_match.group(1)
        # Derive base CVT relation (strip .from/.to suffix)
        base_relation = re.sub(
            r'\.(from|to|start_date|end_date|start|end)$', '', relation
        )

        try:
            scope = get_temporal_scope(entity, base_relation)
            if scope.get('from') or scope.get('to'):
                parts = []
                if scope.get('from'):
                    parts.append(f"from={scope['from']}")
                if scope.get('to'):
                    parts.append(f"to={scope['to']}")
                return f"{relation} ({', '.join(parts)})"
        except Exception:
            pass

        return None

    def _extract_temporal_constraint(self, s_expr: str) -> Optional[str]:
        """Extract human-readable temporal constraint from S-expression."""
        tc_match = re.search(r'TC \([^)]+\) (\S+) (\S+)', s_expr)
        if tc_match:
            relation = tc_match.group(1)
            time_point = tc_match.group(2)
            return f"{relation} = {time_point}"

        argmax_match = re.search(r'ARGMAX', s_expr, re.IGNORECASE)
        if argmax_match:
            return "most recent (ARGMAX)"

        argmin_match = re.search(r'ARGMIN', s_expr, re.IGNORECASE)
        if argmin_match:
            return "earliest (ARGMIN)"

        return None

    def answer(
        self,
        question: str,
        mode: str = 'temporal',
        relaxed: bool = False,
    ) -> dict:
        """Answer a question end-to-end.

        Args:
            question: Natural language question.
            mode: 'temporal' uses temporal-aware generation; 'standard' uses baseline.
            relaxed: If True, use standard mode even if temporal signals detected.

        Returns:
            {
                "answer": list[str],
                "sparql": str | None,
                "temporal_constraint": str | None,
                "candidates_tried": int,
            }
        """
        effective_mode = 'standard' if relaxed else mode
        candidates = self._generate_candidate_sexprs(question, mode=effective_mode)
        answers, sparql, constraint = self._retrieve_and_ground(
            candidates, question, relaxed=relaxed
        )
        return {
            'answer': answers,
            'sparql': sparql,
            'temporal_constraint': constraint,
            'candidates_tried': len(candidates),
        }
