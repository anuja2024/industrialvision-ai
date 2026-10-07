# IndustrialVision AI

## Agentic Multimodal Industrial PCB Inspection & Defect Diagnosis

IndustrialVision AI is an end-to-end computer vision and multimodal AI system for automated PCB inspection.

The system combines:

- DINOv2 visual representations
- PatchCore anomaly detection
- Pixel-level anomaly localization
- Vision-Language Model reasoning
- BM25 technical knowledge retrieval
- Agentic diagnosis workflow
- Confidence-aware decisions
- Human-in-the-loop verification
- Persistent inspection feedback
- GPU-accelerated FastAPI inference
- Docker deployment
- Web-based inspection interface

The goal is not to replace an industrial inspector. Instead, the system acts as an **AI inspection assistant** that detects suspicious regions, provides visual evidence, retrieves relevant technical knowledge, proposes a diagnosis, and routes uncertain cases to human verification.

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │     PCB Image        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Inspection API     │
                    │      FastAPI         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     DINOv2           │
                    │ Visual Representation│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      PatchCore       │
                    │ Anomaly Detection    │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
             Anomaly Score          Heatmap / Regions
                    │                     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Decision Layer    │
                    │ PASS / REVIEW / HOLD │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      SmolVLM         │
                    │ Visual Reasoning      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   BM25 Retrieval     │
                    │ Technical Knowledge  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ AI Defect Diagnosis  │
                    │ Evidence + Action    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Human Verification   │
                    │ Confirm / Reject     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feedback Store       │
                    │ JSONL                │
                    └─────────────────────┘