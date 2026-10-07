from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.agents.diagnosis_agent import get_diagnosis_agent
from app.inference.inspection_service import get_inspection_service


BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"
FEEDBACK_FILE = BASE_DIR / "data" / "inference" / "feedback.jsonl"

VALID_CATEGORIES = {
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
}

VALID_FEEDBACK = {
    "confirmed",
    "rejected",
}


app = FastAPI(
    title="IndustrialVision AI",
    description="Agentic multimodal industrial PCB inspection and defect diagnosis system.",
    version="0.3.0",
)


@app.on_event("startup")
def startup_event():
    get_inspection_service()


@app.get("/")
def root():
    if FRONTEND_FILE.exists():
        return FileResponse(FRONTEND_FILE)

    return {
        "name": "IndustrialVision AI",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    service = get_inspection_service()

    return {
        "status": "healthy",
        "device": str(service.device),
        "cuda_available": service.device.type == "cuda",
        "categories": sorted(VALID_CATEGORIES),
        "models_loaded": list(
            service.patchcore_models.keys()
        ),
    }


@app.post("/inspect")
async def inspect_pcb(
    category: str = Form(...),
    file: UploadFile = File(...),
):
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {category}",
        )

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    service = get_inspection_service()

    start_time = time.perf_counter()

    result = service.inspect(
        image_bytes=contents,
        category=category,
    )

    result["filename"] = file.filename
    result["processing_time_ms"] = (
        time.perf_counter() - start_time
    ) * 1000

    return result


@app.post("/diagnose")
async def diagnose_pcb(
    category: str = Form(...),
    anomaly_score: float = Form(...),
    decision: str = Form(...),
    confidence: float = Form(...),
    regions: str = Form("[]"),
    file: UploadFile = File(...),
):
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {category}",
        )

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:
        parsed_regions = json.loads(regions)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid regions JSON.",
        ) from exc

    if not isinstance(parsed_regions, list):
        raise HTTPException(
            status_code=400,
            detail="Regions must be a JSON list.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    from PIL import Image
    from io import BytesIO

    image = Image.open(
        BytesIO(contents)
    ).convert("RGB")

    agent = get_diagnosis_agent()

    start_time = time.perf_counter()

    result = agent.diagnose(
        image=image,
        category=category,
        anomaly_score=anomaly_score,
        decision=decision,
        confidence=confidence,
        regions=parsed_regions,
    )

    result["processing_time_ms"] = (
        time.perf_counter() - start_time
    ) * 1000

    return result


@app.post("/feedback")
async def submit_feedback(
    category: str = Form(...),
    filename: str = Form("unknown"),
    decision: str = Form(...),
    anomaly_score: float = Form(...),
    diagnosis: str = Form(...),
    feedback: str = Form(...),
):
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {category}",
        )

    if feedback not in VALID_FEEDBACK:
        raise HTTPException(
            status_code=400,
            detail=(
                "Feedback must be either "
                "'confirmed' or 'rejected'."
            ),
        )

    FEEDBACK_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "category": category,
        "filename": filename,
        "decision": decision,
        "anomaly_score": anomaly_score,
        "diagnosis": diagnosis,
        "feedback": feedback,
    }

    with FEEDBACK_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

    return {
        "status": "recorded",
        "feedback": feedback,
        "timestamp": record["timestamp"],
    }


@app.get("/feedback/stats")
def feedback_stats():
    if not FEEDBACK_FILE.exists():
        return {
            "total": 0,
            "confirmed": 0,
            "rejected": 0,
        }

    confirmed = 0
    rejected = 0
    total = 0

    with FEEDBACK_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1

            if record.get("feedback") == "confirmed":
                confirmed += 1

            elif record.get("feedback") == "rejected":
                rejected += 1

    return {
        "total": total,
        "confirmed": confirmed,
        "rejected": rejected,
    }