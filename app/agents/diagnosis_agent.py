from __future__ import annotations

import re
from typing import Any

import torch
from PIL import Image
from rank_bm25 import BM25Okapi
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"


KNOWLEDGE_BASE = [
    {
        "category": "pcb1",
        "defect": "Bent",
        "description": "A component or PCB element appears physically bent or mechanically deformed.",
        "action": "Inspect component alignment and mechanical deformation before further processing.",
    },
    {
        "category": "pcb1",
        "defect": "Melt",
        "description": "A localized melted or thermally altered region is present on the PCB assembly.",
        "action": "Inspect for thermal damage, excessive heat exposure, and affected component or solder regions.",
    },
    {
        "category": "pcb1",
        "defect": "Missing",
        "description": "An expected component or structural element appears absent.",
        "action": "Compare against the expected assembly configuration and verify component presence and placement.",
    },
    {
        "category": "pcb1",
        "defect": "Scratch",
        "description": "A visible surface scratch or linear surface defect is present.",
        "action": "Inspect scratch depth and determine whether conductive or protective layers are affected.",
    },
    {
        "category": "pcb2",
        "defect": "Bent",
        "description": "A component or PCB element appears physically bent or mechanically deformed.",
        "action": "Inspect component alignment and mechanical deformation before further processing.",
    },
    {
        "category": "pcb2",
        "defect": "Melt",
        "description": "A localized melted or thermally altered region is present on the PCB assembly.",
        "action": "Inspect for thermal damage, excessive heat exposure, and affected component or solder regions.",
    },
    {
        "category": "pcb2",
        "defect": "Missing",
        "description": "An expected component or structural element appears absent.",
        "action": "Compare against the expected assembly configuration and verify component presence and placement.",
    },
    {
        "category": "pcb2",
        "defect": "Scratch",
        "description": "A visible surface scratch or linear surface defect is present.",
        "action": "Inspect scratch depth and determine whether conductive or protective layers are affected.",
    },
    {
        "category": "pcb3",
        "defect": "Bent",
        "description": "A component or PCB element appears physically bent or mechanically deformed.",
        "action": "Inspect component alignment and mechanical deformation before further processing.",
    },
    {
        "category": "pcb3",
        "defect": "Melt",
        "description": "A localized melted or thermally altered region is present on the PCB assembly.",
        "action": "Inspect for thermal damage, excessive heat exposure, and affected component or solder regions.",
    },
    {
        "category": "pcb3",
        "defect": "Missing",
        "description": "An expected component or structural element appears absent.",
        "action": "Compare against the expected assembly configuration and verify component presence and placement.",
    },
    {
        "category": "pcb3",
        "defect": "Scratch",
        "description": "A visible surface scratch or linear surface defect is present.",
        "action": "Inspect scratch depth and determine whether conductive or protective layers are affected.",
    },
    {
        "category": "pcb4",
        "defect": "Burnt",
        "description": "A visibly burnt, charred, or strongly thermally discolored region is present.",
        "action": "Inspect for overheating, thermal damage, discoloration, and affected components or solder regions.",
    },
    {
        "category": "pcb4",
        "defect": "Scratch",
        "description": "A visible surface scratch or linear surface defect is present.",
        "action": "Inspect scratch depth and determine whether conductive or protective layers are affected.",
    },
    {
        "category": "pcb4",
        "defect": "Missing",
        "description": "An expected component or structural element appears absent.",
        "action": "Compare against the expected assembly configuration and verify component presence and placement.",
    },
    {
        "category": "pcb4",
        "defect": "Damage",
        "description": "A visible physical or structural defect is present on the PCB assembly.",
        "action": "Inspect the affected region for mechanical or electrical damage and determine whether the board is serviceable.",
    },
    {
        "category": "pcb4",
        "defect": "Extra",
        "description": "An unexpected additional component or material appears in the assembly.",
        "action": "Compare the region against the expected assembly configuration and verify whether the extra material is intentional.",
    },
    {
        "category": "pcb4",
        "defect": "Wrong Place",
        "description": "A component or structural element appears incorrectly positioned relative to the expected assembly.",
        "action": "Compare component placement with the expected assembly configuration.",
    },
    {
        "category": "pcb4",
        "defect": "Dirt",
        "description": "Foreign material or contamination is visibly present on the PCB surface.",
        "action": "Inspect and clean the affected region according to the appropriate PCB handling procedure.",
    },
]


class DiagnosisAgent:

    def __init__(self) -> None:

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.dtype = (
            torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=self.dtype,
        )

        self.model.to(self.device)
        self.model.eval()

        self.documents = KNOWLEDGE_BASE

        tokenized_documents = [
            self._tokenize(
                f"{item['category']} "
                f"{item['defect']} "
                f"{item['description']} "
                f"{item['action']}"
            )
            for item in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:

        return re.findall(
            r"[a-zA-Z]+",
            text.lower(),
        )

    def retrieve_knowledge(
        self,
        category: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:

        query_tokens = self._tokenize(
            f"{category} {query}"
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        candidates = []

        for index, item in enumerate(
            self.documents
        ):

            if item["category"] == category:

                candidates.append(
                    (
                        float(scores[index]),
                        item,
                    )
                )

        candidates.sort(
            key=lambda value: value[0],
            reverse=True,
        )

        results = []

        for score, item in candidates[:top_k]:

            result = dict(item)

            result["retrieval_score"] = round(
                score,
                4,
            )

            results.append(
                result
            )

        return results

    @staticmethod
    def _format_regions(
        regions: list[dict[str, Any]]
    ) -> str:

        if not regions:
            return (
                "No localized anomaly "
                "region was provided."
            )

        lines = []

        for index, region in enumerate(
            regions[:5],
            start=1,
        ):

            lines.append(
                f"Region {index}: "
                f"x1={region.get('x1')}, "
                f"y1={region.get('y1')}, "
                f"x2={region.get('x2')}, "
                f"y2={region.get('y2')}, "
                f"score={region.get('score')}"
            )

        return "\n".join(lines)

    def diagnose(
        self,
        image: Image.Image,
        category: str,
        anomaly_score: float,
        decision: str,
        confidence: float,
        regions: list[dict[str, Any]],
    ) -> dict[str, Any]:

        knowledge = self.retrieve_knowledge(
            category=category,
            query="PCB visual anomaly defect inspection",
            top_k=3,
        )

        knowledge_text = "\n".join(
            [
                (
                    f"- {item['defect']}: "
                    f"{item['description']} "
                    f"Recommended action: "
                    f"{item['action']}"
                )
                for item in knowledge
            ]
        )

        region_text = self._format_regions(
            regions
        )

        prompt = f"""
You are an industrial PCB visual inspection assistant.

Your task is to interpret a PCB image after an anomaly detector has identified a visual deviation.

IMPORTANT:
- PatchCore detects visual feature anomalies only.
- PatchCore does NOT identify the semantic defect type.
- Do not say that PatchCore detected a missing component, melt, scratch, burn, or any other specific defect.
- Determine the likely defect type only from the visible image and the provided technical candidates.
- Do not invent a defect that is not visually supported.
- If the image does not provide enough evidence, explicitly say "Uncertain".
- Do not claim certainty when the visual evidence is weak.
- The retrieved knowledge is supporting technical context, not ground truth.

PCB category: {category}

PatchCore anomaly score: {anomaly_score:.4f}

Inspection decision: {decision}

PatchCore confidence: {confidence:.3f}

Localized anomaly regions:
{region_text}

Technical defect candidates:
{knowledge_text}

Return exactly these four sections:

LIKELY DEFECT:
<one candidate defect or Uncertain>

VISUAL EVIDENCE:
<2 concise sentences describing only what is visibly supported by the image>

CONFIDENCE:
<High, Medium, or Low, followed by one short reason>

RECOMMENDED ACTION:
<one concise inspection/action recommendation>

Do not add any other sections.
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        # Build the textual prompt containing the image
        # placeholder, then explicitly pass the PIL image
        # to the processor.
        formatted_prompt = (
            self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
        )

        inputs = self.processor(
            text=formatted_prompt,
            images=[image],
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
            )

        prompt_length = inputs[
            "input_ids"
        ].shape[1]

        generated_tokens = generated_ids[
            :,
            prompt_length:,
        ]

        diagnosis = (
            self.processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )[0]
            .strip()
        )

        return {
            "model": MODEL_NAME,
            "device": str(self.device),
            "diagnosis": diagnosis,
            "retrieved_knowledge": knowledge,
        }


_AGENT: DiagnosisAgent | None = None


def get_diagnosis_agent() -> DiagnosisAgent:

    global _AGENT

    if _AGENT is None:

        _AGENT = DiagnosisAgent()

    return _AGENT