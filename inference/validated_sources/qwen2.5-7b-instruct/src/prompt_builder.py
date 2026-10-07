"""Prompt construction for all experimental conditions.

Every prompt uses the same auditable response contract: predict the response
of one randomly selected respondent, then emit exactly one displayed option
label. The runner obtains a probability distribution from label likelihoods;
the model is never asked to print or parse a numeric probability vector.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from src.scoring import option_labels


class PromptBuilder:
    """Build forced-choice, label-scored prompts for each condition."""

    CORE_CONDITIONS = {"baseline", "country_label", "population_evidence", "conflict"}
    CORE_TEMPLATE = """You are a social science researcher. A single respondent is selected at random from the target population. Predict which option this one respondent selected.
{cue_block}
Question: {question}
Options:
{options}

{response_instruction}"""

    RESPONSE_INSTRUCTION = (
        "Reply with exactly one of these option labels: {label_choices}. "
        "Do not output anything else.\n"
        "Answer:"
    )

    TEMPLATES = {
        # Ablation conditions
        "persona": """You are a social science researcher. Imagine a single citizen selected at random from {country}. Predict which option this one citizen selected.

Question: {question}
Options:
{options}

{response_instruction}""",
        "evidence_numeric": """You are a social science researcher. A single respondent is selected at random from the population described by the numeric survey evidence below. Predict which option this one respondent selected.

Survey Evidence: {evidence_numeric}
Question: {question}
Options:
{options}

{response_instruction}""",
        "evidence_incomplete": """You are a social science researcher. A single respondent is selected at random from the population described by the related survey information below. Predict which option this one respondent selected.

Related Information: {related_info}
Question: {question}
Options:
{options}

{response_instruction}""",
        # Robustness conditions (template equivalence)
        "baseline_v2": """You are a social scientist. One adult is drawn at random from this society. Predict the option selected by that adult.

Question: {question}
Options:
{options}

{response_instruction}""",
        "baseline_v3": """As a researcher studying public opinion, consider one randomly selected member of this population. Predict that person's selected option.

Question: {question}
Options:
{options}

{response_instruction}""",
        "evidence_v2": """According to survey data from this population, the response distribution is shown below. One respondent is then selected at random from this population. Predict that respondent's selected option.

Survey Evidence: {evidence}
Question: {question}
Options:
{options}

{response_instruction}""",
    }

    def __init__(self) -> None:
        self.templates = self.TEMPLATES

    def build_prompt(
        self,
        condition: str,
        question: str,
        options: Sequence[object],
        country: Optional[str] = None,
        evidence: Optional[str] = None,
        evidence_numeric: Optional[str] = None,
        related_info: Optional[str] = None,
    ) -> str:
        """Build a prompt whose displayed order matches label-score mapping."""

        label_map = self.build_label_map(options)
        labels = list(label_map)
        response_instruction = self.RESPONSE_INSTRUCTION.format(
            label_choices=", ".join(labels)
        )
        kwargs = {
            "question": str(question),
            "options": self._format_options(options),
            "response_instruction": response_instruction,
        }

        if country is not None:
            kwargs["country"] = str(country)
        if evidence is not None:
            kwargs["evidence"] = str(evidence)
        if evidence_numeric is not None:
            kwargs["evidence_numeric"] = str(evidence_numeric)
        if related_info is not None:
            kwargs["related_info"] = str(related_info)

        if condition in self.CORE_CONDITIONS:
            cue_lines = []
            if condition in {"country_label", "conflict"}:
                if country is None:
                    raise ValueError(f"{condition} requires country")
                cue_lines.append(f"Target Country: {country}")
            if condition in {"population_evidence", "conflict"}:
                if evidence is None:
                    raise ValueError(f"{condition} requires evidence")
                cue_lines.append(f"Survey Response Statistics: {evidence}")
            return self.CORE_TEMPLATE.format(cue_block="\n".join(cue_lines), **kwargs)

        template = self.templates.get(condition)
        if template is None:
            raise ValueError(f"Unknown condition: {condition}")
        return template.format(**kwargs)

    @staticmethod
    def build_label_map(options: Sequence[object]) -> Dict[str, str]:
        """Return the displayed label-to-semantic-text mapping."""

        texts = [str(option) for option in options]
        if len(set(texts)) != len(texts):
            raise ValueError("answer options must be unique after string conversion")
        return dict(zip(option_labels(len(texts)), texts))

    def _format_options(self, options: Sequence[object]) -> str:
        """Format the exact mapping that the scoring backend will recover."""

        return "\n".join(
            f"{label}. {option}"
            for label, option in self.build_label_map(options).items()
        )


__all__ = ["PromptBuilder"]
