"""
Polydrone Inference API — FastAPI skeleton (v0)

Run lokaal:
    uvicorn app:app --reload --port 8080
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# -------------------------------
# Settings & Paths
# -------------------------------
class Settings(BaseModel):
    env: Literal["dev", "staging", "prod"] = Field(default=os.getenv("POLYDRONE_ENV", "dev"))
    storage_root: str = Field(default=os.getenv("POLYDRONE_STORAGE", "./storage"))
    allow_origins: List[str] = Field(default_factory=lambda: [os.getenv("POLYDRONE_CORS", "*")])

    class Config:
        arbitrary_types_allowed = True

settings = Settings()
STORAGE_ROOT = Path(settings.storage_root)
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

# -------------------------------
# In-memory registries (swap for DB)
# -------------------------------
CASES: Dict[str, Dict[str, Any]] = {}
ASSETS: Dict[str, List[Dict[str, Any]]] = {}
JOBS: Dict[str, Dict[str, Any]] = {}
PREDICTIONS: Dict[str, Dict[str, Any]] = {}

# -------------------------------
# Utility helpers
# -------------------------------
def now_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def case_root(case_id: str) -> Path:
    return STORAGE_ROOT / "cases" / case_id

def ensure_case(case_id: str) -> Path:
    root = case_root(case_id)
    root.mkdir(parents=True, exist_ok=True)
    (root / "assets").mkdir(parents=True, exist_ok=True)
    (root / "overlays").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    return root

# -------------------------------
# Schemas
# -------------------------------
class CreateCaseFolderIn(BaseModel):
    caseId: str
    root: str = Field(description="e.g. polydrone:// (unused in skeleton)")
    subfolders: Optional[List[str]] = Field(default_factory=lambda: ["assets", "overlays", "reports"])

class CreateCaseFolderOut(BaseModel):
    caseId: str
    createdPaths: List[str]

class ListAssetsIn(BaseModel):
    caseId: str
    filter: Optional[Dict[str, Any]] = None

class Asset(BaseModel):
    path: str
    type: Literal["image", "video"] = "image"
    width: Optional[int] = None
    height: Optional[int] = None

class ListAssetsOut(BaseModel):
    assets: List[Asset]

class NormalizeFilenamesIn(BaseModel):
    caseId: str
    pattern: str = Field(description="Use {index} and original ext. Example: {date}_{caseId}_{index}")

class RenameResult(BaseModel):
    from_path: str = Field(alias="from")
    to_path: str = Field(alias="to")

class NormalizeFilenamesOut(BaseModel):
    renamed: List[RenameResult]

class StartRoofSegmentationJobIn(BaseModel):
    caseId: str
    assetPaths: List[str]

class JobStatusOut(BaseModel):
    status: Literal["queued", "running", "failed", "succeeded"]
    progress: float = 0.0
    resultsPath: Optional[str] = None
    error: Optional[str] = None

class GetJobStatusIn(BaseModel):
    jobId: str

class DamageItem(BaseModel):
    type: Literal["crack", "missing_tile", "lifted_edge", "corrosion", "tear", "ponding"]
    bbox: List[int] = Field(min_items=4, max_items=4)
    confidence: float = Field(ge=0, le=1)
    severity: int = Field(ge=1, le=5)

class MaskItem(BaseModel):
    label: Literal["roof", "tile", "metal", "epdm", "pv_panel", "window", "gutter"]
    polygon: List[float]

class PredictionAsset(BaseModel):
    assetPath: str
    masks: List[MaskItem] = []
    damages: List[DamageItem] = []

class PredictionsOut(BaseModel):
    items: List[PredictionAsset]

class RenderItem(BaseModel):
    assetPath: str
    boxes: Optional[List[DamageItem]] = None
    masks: Optional[List[MaskItem]] = None

class RenderOverlaysIn(BaseModel):
    items: List[RenderItem]
    style: Optional[Dict[str, Any]] = None

class OverlayOut(BaseModel):
    assetPath: str
    overlayPath: str

class RenderOverlaysOut(BaseModel):
    overlays: List[OverlayOut]

class Customer(BaseModel):
    name: str
    policyId: Optional[str] = None
    address: Optional[str] = None

class FindingsSummary(BaseModel):
    primaryDamageTypes: List[str] = []
    severityMax: int = 0
    confidenceMean: float = 0.0

class FindingsAsset(BaseModel):
    assetPath: str
    coverageOk: bool = True
    qualityFlags: List[Literal["ok", "blur", "exposure", "glare"]] = ["ok"]
    damages: List[DamageItem] = []
    materials: List[Literal["tile", "epdm", "metal"]] = []
    pvPanels: Optional[Dict[str, Any]] = None

class FindingsSchema(BaseModel):
    caseId: str
    summary: FindingsSummary
    assets: List[FindingsAsset]

class GenerateReportIn(BaseModel):
    caseId: str
    customer: Customer
    findings: FindingsSchema
    locale: Literal["nl-BE", "fr-BE"] = "nl-BE"
    template: Literal["claim", "maintenance"] = "claim"
    includeImages: bool = True

class GenerateReportOut(BaseModel):
    pdfPath: str
    jsonPath: str

class EmailReportIn(BaseModel):
    to: List[str]
    subject: str
    bodyHtml: str
    attachments: List[str] = []

class EmailReportOut(BaseModel):
    messageId: str

class WebhookNotifyIn(BaseModel):
    url: str
    payload: Dict[str, Any]

class WebhookNotifyOut(BaseModel):
    status: int

# -------------------------------
# App
# -------------------------------
app = FastAPI(title="Polydrone Inference API", version="0.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Forceer JSON bij fouten i.p.v. HTML ---
@app.exception_handler(HTTPException)
async def json_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail), "status_code": exc.status_code, "path": str(request.url.path)},
        media_type="application/json",
    )

# --- Root endpoint als snelle ping ---
@app.get("/")
async def root():
    return {"service": "polydrone", "ok": True}

# -------------------------------
# Case/Asset endpoints
# -------------------------------
@app.post("/cases", response_model=CreateCaseFolderOut)
async def create_case_folder(payload: CreateCaseFolderIn):
    root = ensure_case(payload.caseId)
    created = []
    for sub in (payload.subfolders or []):
        p = (root / sub)
        p.mkdir(parents=True, exist_ok=True)
        created.append(str(p))
    CASES[payload.caseId] = {"created": now_ts()}
    ASSETS.setdefault(payload.caseId, [])
    return CreateCaseFolderOut(caseId=payload.caseId, createdPaths=created)

@app.post("/assets:list", response_model=ListAssetsOut)
async def list_assets(payload: ListAssetsIn):
    if payload.caseId not in CASES:
        raise HTTPException(404, "case not found")

    assets_dir = case_root(payload.caseId) / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    assets_list: List[Asset] = []
    for f in assets_dir.glob("**/*"):
        if f.is_file():
            ext = f.suffix.lower()
            if ext in {".jpg", ".jpeg", ".png"}:
                assets_list.append(Asset(path=str(f), type="image"))
            elif ext in {".mp4", ".mov", ".avi"}:
                assets_list.append(Asset(path=str(f), type="video"))
    ASSETS[payload.caseId] = [a.model_dump() for a in assets_list]
    return ListAssetsOut(assets=assets_list)

@app.post("/assets:normalize", response_model=NormalizeFilenamesOut)
async def normalize_filenames(payload: NormalizeFilenamesIn):
    if payload.caseId not in CASES:
        raise HTTPException(404, "case not found")
    pattern = payload.pattern
    if "{index}" not in pattern:
        raise HTTPException(400, "pattern must contain {index}")

    assets_dir = case_root(payload.caseId) / "assets"
    files = sorted([p for p in assets_dir.iterdir() if p.is_file()])
    renamed: List[RenameResult] = []
    for idx, src in enumerate(files, start=1):
        new_name = pattern.replace("{index}", f"{idx:03}").replace("{caseId}", payload.caseId).replace("{date}", datetime.utcnow().strftime("%Y%m%d"))
        target = src.with_name(f"{new_name}{src.suffix.lower()}")
        if target.exists():
            target = src.with_name(f"{new_name}_{uuid.uuid4().hex[:6]}{src.suffix.lower()}")
        src.rename(target)
        renamed.append(RenameResult(**{"from": str(src), "to": str(target)}))
    return NormalizeFilenamesOut(renamed=renamed)

# -------------------------------
# Jobs (segmentation) — simulated
# -------------------------------
async def _run_segmentation_job(job_id: str, case_id: str, asset_paths: List[str]):
    JOBS[job_id]["status"] = "running"
    steps = max(5, min(20, len(asset_paths)))
    for i in range(steps):
        await asyncio.sleep(0.3)
        JOBS[job_id]["progress"] = (i + 1) / steps
    results_dir = case_root(case_id) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"predictions_{job_id}.json"
    fake_items = []
    for ap in asset_paths:
        fake_items.append({
            "assetPath": ap,
            "masks": [{"label": "roof", "polygon": [0.1,0.1, 0.9,0.1, 0.9,0.9, 0.1,0.9]}],
            "damages": [{"type": "missing_tile", "bbox": [100,120,180,200], "confidence": 0.8, "severity": 3}],
        })
    data = {"items": fake_items}
    results_path.write_text(json.dumps(data, indent=2))
    JOBS[job_id]["status"] = "succeeded"
    JOBS[job_id]["resultsPath"] = str(results_path)

@app.post("/jobs/roof-segmentation:start", response_model=Dict[str, str])
async def start_roof_segmentation_job(payload: StartRoofSegmentationJobIn, background_tasks: BackgroundTasks):
    if payload.caseId not in CASES:
        raise HTTPException(404, "case not found")
    if not payload.assetPaths:
        raise HTTPException(400, "assetPaths required")
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "caseId": payload.caseId,
        "assetPaths": payload.assetPaths,
        "created": now_ts(),
    }
    background_tasks.add_task(_run_segmentation_job, job_id, payload.caseId, payload.assetPaths)
    return {"jobId": job_id}

@app.post("/jobs:status", response_model=JobStatusOut)
async def get_job_status(payload: GetJobStatusIn):
    job = JOBS.get(payload.jobId)
    if not job:
        raise HTTPException(404, "job not found")
    return JobStatusOut(
        status=job["status"],
        progress=float(job.get("progress", 0.0)),
        resultsPath=job.get("resultsPath"),
        error=job.get("error"),
    )

@app.get("/predictions", response_model=PredictionsOut)
async def get_predictions(resultsPath: str):
    p = Path(resultsPath)
    if not p.exists():
        raise HTTPException(404, "resultsPath not found")
    data = json.loads(p.read_text())
    out = PredictionsOut(**data)  # validatie
    PREDICTIONS[resultsPath] = out.model_dump()
    return out

# -------------------------------
# Overlays (placeholder)
# -------------------------------
@app.post("/render/overlays", response_model=RenderOverlaysOut)
async def render_overlays(payload: RenderOverlaysIn):
    overlays: List[OverlayOut] = []
    for item in payload.items:
        src = Path(item.assetPath)
        if not src.exists():
            overlay_path = case_root(Path(item.assetPath).parts[-3]) / "overlays" / (Path(item.assetPath).stem + "_overlay.png")
        else:
            overlay_path = src.parent.parent / "overlays" / (src.stem + "_overlay.png")
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_bytes(b"PNGPLACEHOLDER")
        overlays.append(OverlayOut(assetPath=item.assetPath, overlayPath=str(overlay_path)))
    return RenderOverlaysOut(overlays=overlays)

# -------------------------------
# Reports (placeholder PDF + JSON)
# -------------------------------
@app.post("/reports:generate", response_model=GenerateReportOut)
async def generate_report(payload: GenerateReportIn):
    root = ensure_case(payload.caseId)
    json_path = root / "reports" / f"findings_{now_ts()}.json"
    json_path.write_text(payload.findings.model_dump_json(indent=2))

    pdf_path = root / "reports" / f"report_{now_ts()}.pdf"
    summary = (
        f"Polydrone Inspectierapport — Case {payload.caseId}\n"
        f"Klant: {payload.customer.name}\n"
        f"Locale: {payload.locale}, Template: {payload.template}\n"
        f"Items: {len(payload.findings.assets)}\n"
    )
    pdf_path.write_bytes(summary.encode("utf-8"))
    return GenerateReportOut(pdfPath=str(pdf_path), jsonPath=str(json_path))

# -------------------------------
# Email (placeholder)
# -------------------------------
@app.post("/email:send", response_model=EmailReportOut)
async def email_report(payload: EmailReportIn):
    return EmailReportOut(messageId=f"msg_{uuid.uuid4().hex[:10]}")

# -------------------------------
# Webhook notify (placeholder)
# -------------------------------
@app.post("/webhook:notify", response_model=WebhookNotifyOut)
async def webhook_notify(payload: WebhookNotifyIn):
    return WebhookNotifyOut(status=202)

# -------------------------------
# Health (JSON enforced)
# -------------------------------
@app.get("/healthz")
async def healthz():
    return JSONResponse(
        content={"status": "ok", "env": settings.env},
        media_type="application/json",
    )

# -------------------------------
# Dev convenience: seed assets
# -------------------------------
@app.post("/dev:seed-assets")
async def dev_seed_assets(caseId: str, count: int = 5):
    ensure_case(caseId)
    assets_dir = case_root(caseId) / "assets"
    for i in range(1, count + 1):
        (assets_dir / f"{datetime.utcnow().strftime('%Y%m%d')}_{caseId}_{i:03}.jpg").write_bytes(b"JPEGPLACEHOLDER")
    return {"seeded": count}
