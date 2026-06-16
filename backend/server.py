"""MediGuard AI – FastAPI backend.

Wires Auth (Emergent Google), patient profiles, prescriptions (OCR + safety),
adherence, ML risk engine, copilot chat, voice (STT/TTS), and OSM emergency.
"""
from __future__ import annotations

import os
import logging
import base64
import io
import json
import tempfile
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from pathlib import Path
import uuid

from fastapi import (
    FastAPI,
    APIRouter,
    Cookie,
    Header,
    HTTPException,
    UploadFile,
    File,
    Form,
    Query,
    Response,
)
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from PIL import Image

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

from db_models import (
    PatientProfile,
    Prescription,
    Medicine,
    SafetyAlert,
    DoseEvent,
    ChatMessage,
    new_id,
    now_utc,
    serialize_for_mongo,
)
from auth import build_auth_router, require_user
from ml_engine import ml_engine, compute_risk_score, SEVERITY_LABELS, URGENCY_LABELS
import ai_services
import voice as voice_service
import osm_service


logger = logging.getLogger("mediguard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

app = FastAPI(title="MediGuard AI")
api = APIRouter(prefix="/api")


@app.on_event("startup")
async def startup():
    result = ml_engine.train_or_load()
    logger.info("ML engine ready: %s", result)
    # Indexes
    await db.users.create_index("user_id", unique=True)
    await db.users.create_index("email", unique=True)
    await db.user_sessions.create_index("session_token", unique=True)
    await db.patient_profiles.create_index("user_id", unique=True)
    await db.prescriptions.create_index("user_id")
    await db.prescriptions.create_index("prescription_id", unique=True)
    await db.dose_events.create_index([("user_id", 1), ("scheduled_for", 1)])
    await db.chat_messages.create_index([("user_id", 1), ("session_id", 1)])


@app.on_event("shutdown")
async def shutdown():
    client.close()


# Auth
auth_router, _get_user_from_session = build_auth_router(db)
api.include_router(auth_router)


# Helper to read current user
async def current_user(
    session_token: Optional[str] = Cookie(default=None),
    authorization: Optional[str] = Header(default=None),
):
    print("CURRENT_USER AUTH =", authorization)
    return await require_user(db, session_token, authorization)


@api.get("/")
async def root():
    return {"service": "MediGuard AI", "status": "ok"}


@api.get("/health")
async def health():
    return {"status": "ok", "ml": "ready"}


# -----------------------------------------------------------------------------
# Patient profile
# -----------------------------------------------------------------------------
class ProfileInput(BaseModel):
    full_name: str
    age: int
    sex: str = "other"
    weight_kg: Optional[float] = None
    pregnant: bool = False
    trimester: Optional[int] = None
    chronic_conditions: List[str] = []
    allergies: List[str] = []
    language: str = "en"
    caregiver_name: Optional[str] = None
    caregiver_email: Optional[str] = None
    caregiver_phone: Optional[str] = None
    location: Optional[dict] = None


def _caregiver_required(profile: dict) -> dict:
    age = profile.get("age", 30)
    chronic = profile.get("chronic_conditions") or []
    high_risk_terms = {
        "heart disease", "cancer", "stroke", "dementia", "parkinson",
        "chronic kidney disease", "ckd", "alzheimer",
    }
    chronic_lower = {c.lower() for c in chronic}
    high_risk_match = any(any(t in c for t in high_risk_terms) for c in chronic_lower)
    if age < 18:
        return {"required": True, "reason": "Patient is a minor — guardian required."}
    if age >= 60 and (chronic or high_risk_match):
        return {"required": True, "reason": "Age 60+ with chronic condition."}
    if high_risk_match:
        return {"required": True, "reason": "High-risk condition detected."}
    if age >= 60:
        return {"required": False, "recommended": True, "reason": "Strongly recommended for 60+."}
    if age >= 41:
        return {"required": False, "recommended": True, "reason": "Recommended."}
    return {"required": False, "recommended": False, "reason": "Optional."}


from fastapi import Depends


@api.post("/profile")
async def upsert_profile(body: ProfileInput, user=Depends(current_user)):
    data = body.model_dump()
    data["user_id"] = user.user_id
    data["onboarded"] = True
    data["updated_at"] = now_utc().isoformat()
    await db.patient_profiles.update_one(
        {"user_id": user.user_id},
        {"$set": data},
        upsert=True,
    )
    profile = await db.patient_profiles.find_one({"user_id": user.user_id}, {"_id": 0})
    profile["caregiver_status"] = _caregiver_required(profile)
    return profile


@api.get("/profile")
async def get_profile(user=Depends(current_user)):
    profile = await db.patient_profiles.find_one({"user_id": user.user_id}, {"_id": 0})
    if not profile:
        return {"onboarded": False}
    profile["caregiver_status"] = _caregiver_required(profile)
    return profile


# -----------------------------------------------------------------------------
# Prescription OCR + analysis pipeline
# -----------------------------------------------------------------------------
async def _process_prescription(file_bytes: bytes, mime: str, user) -> Prescription:
    # Save image to temp for Gemini ingestion
    suffix = ".jpg"
    if mime == "image/png":
        suffix = ".png"
    elif mime == "image/webp":
        suffix = ".webp"
    elif mime == "application/pdf":
        suffix = ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Generate small thumbnail for storage (skip if PDF)
    thumb_b64 = None
    try:
        if mime.startswith("image/"):
            img = Image.open(io.BytesIO(file_bytes))
            img.thumbnail((640, 640))
            buf = io.BytesIO()
            (img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img).save(
                buf, format="JPEG", quality=72
            )
            thumb_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        thumb_b64 = None

    try:
        ocr = await ai_services.run_ocr(tmp_path, mime, session_id=f"ocr-{user.user_id}-{new_id('s')}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    medicines_raw = ocr.get("medicines") or []
    medicines: List[Medicine] = []
    for m in medicines_raw:
        try:
            tpd = int(m.get("times_per_day") or 1)
        except Exception:
            tpd = 1
        try:
            dd = int(m.get("duration_days")) if m.get("duration_days") is not None else None
        except Exception:
            dd = None
        medicines.append(
            Medicine(
                name=str(m.get("name") or "Unknown medicine"),
                generic=m.get("generic"),
                dosage=m.get("dosage"),
                frequency=m.get("frequency"),
                times_per_day=max(1, tpd),
                duration_days=dd,
                food=m.get("food"),
                instructions=m.get("instructions"),
            )
        )

    profile_doc = await db.patient_profiles.find_one({"user_id": user.user_id}, {"_id": 0}) or {}
    med_dicts = [m.model_dump() for m in medicines]

    # Run safety + explanations + adherence prediction in parallel
    safety_task = ai_services.analyse_safety(
        med_dicts,
        {
            "age": profile_doc.get("age"),
            "sex": profile_doc.get("sex"),
            "allergies": profile_doc.get("allergies", []),
            "chronic_conditions": profile_doc.get("chronic_conditions", []),
            "pregnant": profile_doc.get("pregnant"),
            "trimester": profile_doc.get("trimester"),
        },
        session_id=f"safety-{user.user_id}-{new_id('s')}",
    )
    explain_task = ai_services.explain_medicines(
        med_dicts,
        profile_doc.get("language", "en"),
        session_id=f"explain-{user.user_id}-{new_id('s')}",
    )
    alerts_raw, explanations = await asyncio.gather(safety_task, explain_task)

    alerts = [SafetyAlert(**a) for a in alerts_raw if a.get("category") and a.get("severity")]
    explanations_map = {}
    if explanations:
        # match by name
        for exp in explanations:
            name = (exp.get("name") or "").lower()
            for med in medicines:
                if name and name in med.name.lower():
                    explanations_map[med.medicine_id] = exp
                    break

    # ML predictions
    n_meds = len(medicines)
    complexity = max(1, sum(m.times_per_day for m in medicines) // max(1, n_meds or 1))
    age = int(profile_doc.get("age") or 35)
    chronic_count = len(profile_doc.get("chronic_conditions") or [])

    # Adherence: assume history default 0.6 if first time
    history_doc = await db.dose_events.find({"user_id": user.user_id}).to_list(500)
    if history_doc:
        taken = sum(1 for d in history_doc if d.get("status") == "taken")
        finished = sum(1 for d in history_doc if d.get("status") in ("taken", "missed"))
        history_rate = taken / finished if finished else 0.6
    else:
        history_rate = 0.65
    miss_prob, miss_level = ml_engine.predict_adherence(
        n_meds=n_meds, complexity=complexity, age=age,
        history=history_rate, reminders_used=0.5, chronic=1 if chronic_count else 0,
    )

    # Severity baseline (no symptoms => low)
    severity_label, sev_conf, _ = ml_engine.predict_severity(
        symptom_severity=2, chest_pain=0, breathing_difficulty=0, confusion=0,
        bleeding=0, n_meds=n_meds, age=age, chronic_count=chronic_count, fever_c=37.0,
    )

    # Risk score
    severe_alerts = sum(1 for a in alerts if a.severity in ("severe", "critical"))
    risk = compute_risk_score(
        [a.model_dump() for a in alerts], profile_doc, n_meds, miss_prob, severity_label
    )

    urgency_label, urg_conf, _ = ml_engine.predict_urgency(
        symptom_severity=2, risk_score=risk, severe_alerts=severe_alerts,
        age=age, missed_doses_7d=0, chronic_count=chronic_count,
    )

    rx = Prescription(
        user_id=user.user_id,
        image_b64=thumb_b64,
        source_mime=mime,
        doctor_name=ocr.get("doctor_name"),
        diagnosis=ocr.get("diagnosis"),
        medicines=medicines,
        explanations=explanations_map,
        alerts=alerts,
        risk_score=risk,
        severity_label=severity_label,
        severity_confidence=sev_conf,
        visit_urgency=urgency_label,
        visit_urgency_confidence=urg_conf,
        adherence_predicted_risk=miss_prob,
        raw_ocr_text=ocr.get("raw_text"),
    )
    await db.prescriptions.insert_one(serialize_for_mongo(rx))
    # Auto-create dose events for next 7 days
    await _create_dose_schedule(rx, user.user_id)
    return rx


async def _create_dose_schedule(rx: Prescription, user_id: str):
    """Build 7 days of dose events at sensible times based on times_per_day."""
    base = now_utc().replace(minute=0, second=0, microsecond=0)
    time_map = {
        1: [9],
        2: [9, 21],
        3: [8, 14, 21],
        4: [8, 13, 17, 22],
    }
    events = []
    for med in rx.medicines:
        slots = time_map.get(med.times_per_day, [9])
        duration = min(med.duration_days or 7, 14)
        for day in range(duration):
            for hour in slots:
                scheduled = base.replace(hour=hour) + timedelta(days=day)
                ev = DoseEvent(
                    user_id=user_id,
                    prescription_id=rx.prescription_id,
                    medicine_id=med.medicine_id,
                    medicine_name=med.name,
                    scheduled_for=scheduled,
                )
                events.append(serialize_for_mongo(ev))
    if events:
        await db.dose_events.insert_many(events)


@api.post("/prescriptions/upload")
async def upload_prescription(
    file: UploadFile = File(...),
    user=Depends(current_user),
):
    content = await file.read()
    if len(content) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 12MB)")
    mime = file.content_type or "image/jpeg"
    if mime not in ("image/jpeg", "image/png", "image/webp", "application/pdf"):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    try:
        rx = await _process_prescription(content, mime, user)
    except Exception as e:
        logger.exception("OCR pipeline failed")
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")
    return rx.model_dump()


@api.get("/prescriptions")
async def list_prescriptions(user=Depends(current_user)):
    docs = await db.prescriptions.find(
        {"user_id": user.user_id},
        {"_id": 0, "image_b64": 0, "raw_ocr_text": 0},
    ).sort("created_at", -1).to_list(50)
    return docs


@api.get("/prescriptions/{prescription_id}")
async def get_prescription(prescription_id: str, user=Depends(current_user)):
    doc = await db.prescriptions.find_one(
        {"prescription_id": prescription_id, "user_id": user.user_id},
        {"_id": 0},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return doc


@api.delete("/prescriptions/{prescription_id}")
async def delete_prescription(prescription_id: str, user=Depends(current_user)):
    await db.dose_events.delete_many({"user_id": user.user_id, "prescription_id": prescription_id})
    res = await db.prescriptions.delete_one({"prescription_id": prescription_id, "user_id": user.user_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# -----------------------------------------------------------------------------
# Adherence & dose events
# -----------------------------------------------------------------------------
@api.get("/dose-events")
async def list_doses(
    days: int = Query(7, ge=1, le=30),
    user=Depends(current_user),
):
    start = now_utc() - timedelta(days=1)
    end = now_utc() + timedelta(days=days)
    docs = await db.dose_events.find(
        {
            "user_id": user.user_id,
            "scheduled_for": {"$gte": start.isoformat(), "$lte": end.isoformat()},
        },
        {"_id": 0},
    ).sort("scheduled_for", 1).to_list(500)
    return docs


class DoseAction(BaseModel):
    status: str  # taken | missed | delayed


@api.post("/dose-events/{event_id}/action")
async def dose_action(event_id: str, body: DoseAction, user=Depends(current_user)):
    if body.status not in ("taken", "missed", "delayed"):
        raise HTTPException(status_code=400, detail="Bad status")
    update = {"status": body.status}
    if body.status == "taken":
        now = now_utc()
        ev = await db.dose_events.find_one(
            {"event_id": event_id, "user_id": user.user_id}, {"_id": 0}
        )
        if not ev:
            raise HTTPException(status_code=404, detail="Not found")
        sched = ev["scheduled_for"]
        if isinstance(sched, str):
            sched = datetime.fromisoformat(sched)
        if sched.tzinfo is None:
            sched = sched.replace(tzinfo=timezone.utc)
        delay = int((now - sched).total_seconds() / 60)
        update["taken_at"] = now.isoformat()
        update["delay_minutes"] = delay
        if delay > 60:
            update["status"] = "delayed"
    await db.dose_events.update_one(
        {"event_id": event_id, "user_id": user.user_id},
        {"$set": update},
    )
    return {"ok": True}


@api.get("/adherence/summary")
async def adherence_summary(user=Depends(current_user)):
    docs = await db.dose_events.find({"user_id": user.user_id}, {"_id": 0}).to_list(2000)
    now = now_utc()
    by_day: dict[str, dict] = {}
    taken = missed = pending = delayed = 0
    for d in docs:
        sched = d["scheduled_for"]
        if isinstance(sched, str):
            sched = datetime.fromisoformat(sched)
        if sched.tzinfo is None:
            sched = sched.replace(tzinfo=timezone.utc)
        day = sched.strftime("%Y-%m-%d")
        bucket = by_day.setdefault(day, {"taken": 0, "missed": 0, "pending": 0, "delayed": 0})
        status = d.get("status", "pending")
        if status == "pending" and sched < now - timedelta(hours=2):
            status = "missed"
        bucket[status] = bucket.get(status, 0) + 1
        if status == "taken":
            taken += 1
        elif status == "missed":
            missed += 1
        elif status == "delayed":
            delayed += 1
        else:
            pending += 1

    total_completed = taken + missed + delayed
    score = int(((taken + delayed * 0.5) / total_completed) * 100) if total_completed else 100
    trend = [{"date": k, **v} for k, v in sorted(by_day.items())][-14:]

    # ML-based future adherence risk
    profile = await db.patient_profiles.find_one({"user_id": user.user_id}, {"_id": 0}) or {}
    rx_count = await db.prescriptions.count_documents({"user_id": user.user_id})
    age = int(profile.get("age") or 35)
    chronic_count = len(profile.get("chronic_conditions") or [])
    history_rate = (taken / total_completed) if total_completed else 0.65
    miss_prob, level = ml_engine.predict_adherence(
        n_meds=max(1, rx_count), complexity=2, age=age,
        history=history_rate, reminders_used=0.5,
        chronic=1 if chronic_count else 0,
    )
    return {
        "score": score,
        "taken": taken,
        "missed": missed,
        "delayed": delayed,
        "pending": pending,
        "trend": trend,
        "future_miss_prob": miss_prob,
        "future_miss_level": level,
    }


# -----------------------------------------------------------------------------
# Risk dashboard
# -----------------------------------------------------------------------------
@api.get("/dashboard")
async def dashboard(user=Depends(current_user)):
    profile = await db.patient_profiles.find_one({"user_id": user.user_id}, {"_id": 0}) or {}
    prescriptions = await db.prescriptions.find(
        {"user_id": user.user_id},
        {"_id": 0, "image_b64": 0, "raw_ocr_text": 0},
    ).sort("created_at", -1).to_list(20)
    latest = prescriptions[0] if prescriptions else None
    adherence = await adherence_summary(user)  # reuse
    caregiver = _caregiver_required(profile) if profile else {"required": False}
    active_meds = []
    if latest:
        active_meds = latest.get("medicines", [])
    all_alerts = []
    for rx in prescriptions:
        all_alerts.extend(rx.get("alerts", []))
    critical_alerts = [a for a in all_alerts if a.get("severity") in ("severe", "critical")]
    return {
        "profile": profile,
        "latest_prescription": latest,
        "prescription_count": len(prescriptions),
        "active_medicines": active_meds,
        "adherence": adherence,
        "alerts": all_alerts[:10],
        "critical_alerts_count": len(critical_alerts),
        "caregiver_status": caregiver,
    }


# -----------------------------------------------------------------------------
# Emergency mode
# -----------------------------------------------------------------------------
class EmergencyInput(BaseModel):
    symptoms: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    location_query: Optional[str] = None


@api.post("/emergency/assess")
async def emergency_assess(body: EmergencyInput, user=Depends(current_user)):
    profile = await db.patient_profiles.find_one({"user_id": user.user_id}, {"_id": 0}) or {}
    triage = await ai_services.triage_symptoms(
        body.symptoms,
        {
            "age": profile.get("age"),
            "sex": profile.get("sex"),
            "chronic_conditions": profile.get("chronic_conditions", []),
            "allergies": profile.get("allergies", []),
            "pregnant": profile.get("pregnant"),
        },
        session_id=f"triage-{user.user_id}-{new_id('s')}",
    )
    age = int(profile.get("age") or 35)
    chronic_count = len(profile.get("chronic_conditions") or [])
    severity_label, sev_conf, sev_dist = ml_engine.predict_severity(
        symptom_severity=triage["symptom_severity"],
        chest_pain=triage["chest_pain"],
        breathing_difficulty=triage["breathing_difficulty"],
        confusion=triage["confusion"],
        bleeding=triage["bleeding"],
        n_meds=0, age=age, chronic_count=chronic_count,
        fever_c=triage["fever_c"],
    )
    # Latest risk
    latest = await db.prescriptions.find_one({"user_id": user.user_id}, {"_id": 0}, sort=[("created_at", -1)])
    risk_score = latest.get("risk_score", 30) if latest else 30
    severe_alerts = sum(1 for a in (latest.get("alerts", []) if latest else []) if a.get("severity") in ("severe", "critical"))
    urgency_label, urg_conf, urg_dist = ml_engine.predict_urgency(
        symptom_severity=triage["symptom_severity"],
        risk_score=risk_score,
        severe_alerts=severe_alerts,
        age=age,
        missed_doses_7d=0,
        chronic_count=chronic_count,
    )

    # Find facilities
    lat = body.lat
    lon = body.lon
    location_label = None
    if (lat is None or lon is None) and body.location_query:
        geo = await osm_service.geocode(body.location_query)
        if geo:
            lat, lon = geo["lat"], geo["lon"]
            location_label = geo["label"]
    if (lat is None or lon is None) and profile.get("location"):
        loc = profile["location"]
        lat = loc.get("lat")
        lon = loc.get("lon")
        location_label = loc.get("label")
    facilities = {"hospitals": [], "clinics": [], "pharmacies": []}
    if lat is not None and lon is not None:
        facilities = await osm_service.find_nearby_facilities(lat, lon)

    # Generate emergency summary text
    summary_parts = [
        f"Patient: {profile.get('full_name', 'Patient')} ({profile.get('age', '?')}y, {profile.get('sex', '')})",
        f"Symptoms reported: {body.symptoms}",
        f"Primary concern: {triage['primary_concern']}",
        f"AI severity: {severity_label.upper()} (confidence {int(sev_conf*100)}%).",
        f"Visit urgency: {urgency_label.replace('_',' ')} (confidence {int(urg_conf*100)}%).",
        f"Current risk score: {risk_score}/100.",
    ]
    if profile.get("allergies"):
        summary_parts.append("Allergies: " + ", ".join(profile["allergies"]))
    if profile.get("chronic_conditions"):
        summary_parts.append("Conditions: " + ", ".join(profile["chronic_conditions"]))
    if latest and latest.get("medicines"):
        med_names = [m.get("name") for m in latest["medicines"][:8]]
        summary_parts.append("Active medicines: " + ", ".join(med_names))
    summary = "\n".join(summary_parts)

    return {
        "triage": triage,
        "severity": {"label": severity_label, "confidence": sev_conf, "distribution": sev_dist},
        "urgency": {"label": urgency_label, "confidence": urg_conf, "distribution": urg_dist},
        "risk_score": risk_score,
        "facilities": facilities,
        "location": {"lat": lat, "lon": lon, "label": location_label},
        "summary": summary,
    }


# -----------------------------------------------------------------------------
# AI Copilot (streaming SSE)
# -----------------------------------------------------------------------------
class CopilotInput(BaseModel):
    message: str
    chat_session_id: Optional[str] = None
    language: Optional[str] = "en"


@api.post("/copilot/chat")
async def copilot_chat(body: CopilotInput, user=Depends(current_user)):
    chat_sess_id = body.chat_session_id or new_id("chat")
    profile = await db.patient_profiles.find_one({"user_id": user.user_id}, {"_id": 0}) or {}
    prescriptions = await db.prescriptions.find(
        {"user_id": user.user_id},
        {"_id": 0, "image_b64": 0, "raw_ocr_text": 0},
    ).sort("created_at", -1).to_list(5)
    # History
    history_docs = await db.chat_messages.find(
        {"user_id": user.user_id, "session_id": chat_sess_id},
        {"_id": 0},
    ).sort("created_at", 1).to_list(40)
    history = [{"role": h["role"], "content": h["content"]} for h in history_docs]

    # Persist user message
    user_msg = ChatMessage(
        user_id=user.user_id, session_id=chat_sess_id, role="user", content=body.message
    )
    await db.chat_messages.insert_one(serialize_for_mongo(user_msg))

    async def event_stream():
        collected = []
        try:
            async for chunk in ai_services.copilot_stream(
                session_id=f"copilot-{user.user_id}-{chat_sess_id}",
                profile=profile,
                prescriptions=prescriptions,
                history=history,
                user_text=body.message,
                language=body.language or profile.get("language", "en"),
            ):
                collected.append(chunk)
                yield f"data: {json.dumps({'delta': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        full = "".join(collected)
        # Persist assistant message
        if full:
            asst = ChatMessage(
                user_id=user.user_id, session_id=chat_sess_id, role="assistant", content=full
            )
            await db.chat_messages.insert_one(serialize_for_mongo(asst))
        yield f"data: {json.dumps({'done': True, 'session_id': chat_sess_id})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api.get("/copilot/sessions")
async def copilot_sessions(user=Depends(current_user)):
    pipeline = [
        {"$match": {"user_id": user.user_id}},
        {"$sort": {"created_at": 1}},
        {"$group": {
            "_id": "$session_id",
            "first": {"$first": "$content"},
            "last_at": {"$last": "$created_at"},
            "count": {"$sum": 1},
        }},
        {"$sort": {"last_at": -1}},
        {"$limit": 20},
    ]
    sessions = []
    async for doc in db.chat_messages.aggregate(pipeline):
        sessions.append({
            "session_id": doc["_id"],
            "preview": doc["first"][:80],
            "last_at": doc["last_at"],
            "count": doc["count"],
        })
    return sessions


@api.get("/copilot/messages/{chat_session_id}")
async def copilot_messages(chat_session_id: str, user=Depends(current_user)):
    docs = await db.chat_messages.find(
        {"user_id": user.user_id, "session_id": chat_session_id},
        {"_id": 0},
    ).sort("created_at", 1).to_list(200)
    return docs


# -----------------------------------------------------------------------------
# Voice
# -----------------------------------------------------------------------------
@api.post("/voice/transcribe")
async def voice_transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    user=Depends(current_user),
):
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio too large")
    try:
        text = await voice_service.transcribe_audio(content, file.filename or "audio.webm", language)
    except Exception as e:
        logger.exception("STT failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"text": text}


class TTSInput(BaseModel):
    text: str
    voice: Optional[str] = "nova"


@api.post("/voice/tts")
async def voice_tts(body: TTSInput, user=Depends(current_user)):
    try:
        audio = await voice_service.synthesize_speech(body.text, voice=body.voice or "nova")
    except Exception as e:
        logger.exception("TTS failed")
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=audio, media_type="audio/mpeg")


# -----------------------------------------------------------------------------
# Caregiver view (read-only)
# -----------------------------------------------------------------------------
@api.get("/caregiver/snapshot")
async def caregiver_snapshot(user=Depends(current_user)):
    profile = await db.patient_profiles.find_one(
        {"user_id": user.user_id},
        {"_id": 0}
    ) or {}

    latest = await db.prescriptions.find_one(
        {"user_id": user.user_id},
        {"_id": 0, "image_b64": 0, "raw_ocr_text": 0},
        sort=[("created_at", -1)],
    )

    adherence = await adherence_summary(user)

    invite = await db.caregiver_invites.find_one(
        {"user_id": user.user_id, "revoked": False},
        sort=[("created_at", -1)]
    )

    missed = adherence.get("missed", 0)

    if missed >= 5:
        escalation = {"level": "critical", "label": "Critical alert"}
    elif missed >= 3:
        escalation = {"level": "warning", "label": "Caregiver notification"}
    elif missed >= 2:
        escalation = {"level": "warning", "label": "Warning"}
    elif missed >= 1:
        escalation = {"level": "info", "label": "Reminder"}
    else:
        escalation = {"level": "ok", "label": "All good"}

    caregiver_status = _caregiver_required(profile) if profile else {"required": False}

    return {
        "profile": profile,

        "caregiver": {
            "name": invite.get("caregiver_name") if invite else None,
            "email": invite.get("caregiver_email") if invite else None,
        },

        "caregiver_status": caregiver_status,
        "latest_prescription": latest,
        "adherence": adherence,
        "escalation": escalation,
    }

# -----------------------------------------------------------------------------
# Pill confusion check (Gemini vision)
# -----------------------------------------------------------------------------
@api.post("/pill/check")
async def pill_check(
    file: UploadFile = File(...),
    medicine_name: Optional[str] = Form(default=None),
    user=Depends(current_user),
):
    content = await file.read()
    if len(content) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large")
    mime = file.content_type or "image/jpeg"
    if mime not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Only JPG/PNG/WEBP")

    _profile = await db.patient_profiles.find_one({"user_id": user.user_id}, {"_id": 0}) or {}
    latest = await db.prescriptions.find_one(
        {"user_id": user.user_id}, {"_id": 0, "image_b64": 0, "raw_ocr_text": 0},
        sort=[("created_at", -1)],
    )
    active_meds = [m.get("name") for m in (latest.get("medicines") if latest else []) or []]

    suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[mime]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = await ai_services.pill_visual_check(
            tmp_path, mime,
            claimed_medicine=medicine_name,
            active_medicines=active_meds,
            session_id=f"pill-{user.user_id}-{new_id('s')}",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    # Persist check
    await db.pill_checks.insert_one({
        "check_id": new_id("pc"),
        "user_id": user.user_id,
        "claimed_medicine": medicine_name,
        "result": result,
        "created_at": now_utc().isoformat(),
    })
    return result


# -----------------------------------------------------------------------------
# Caregiver magic-link invites & public read-only view
# -----------------------------------------------------------------------------
class CaregiverInviteInput(BaseModel):
    caregiver_name: Optional[str] = None
    caregiver_email: Optional[str] = None


@api.post("/caregiver/invite")
async def caregiver_invite(body: CaregiverInviteInput, user=Depends(current_user)):
    token = new_id("cgt") + uuid.uuid4().hex[:12]
    expires = now_utc() + timedelta(days=30)
    doc = {
        "token": token,
        "user_id": user.user_id,
        "caregiver_name": body.caregiver_name,
        "caregiver_email": body.caregiver_email,
        "expires_at": expires.isoformat(),
        "created_at": now_utc().isoformat(),
        "revoked": False,
    }
    await db.caregiver_invites.insert_one(doc)
    return {"token": token, "expires_at": expires.isoformat()}


@api.get("/caregiver/invites")
async def caregiver_invites(user=Depends(current_user)):
    docs = await db.caregiver_invites.find(
        {"user_id": user.user_id, "revoked": False}, {"_id": 0}
    ).sort("created_at", -1).to_list(20)
    return docs


@api.delete("/caregiver/invite/{token}")
async def revoke_invite(token: str, user=Depends(current_user)):
    await db.caregiver_invites.update_one(
        {"token": token, "user_id": user.user_id}, {"$set": {"revoked": True}}
    )
    return {"ok": True}


@api.get("/public/caregiver/{token}")
async def public_caregiver_view(token: str):
    invite = await db.caregiver_invites.find_one({"token": token, "revoked": False}, {"_id": 0})
    if not invite:
        raise HTTPException(status_code=404, detail="Invalid or revoked invite")
    exp = invite.get("expires_at")
    if isinstance(exp, str):
        exp = datetime.fromisoformat(exp)
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    if exp < now_utc():
        raise HTTPException(status_code=410, detail="Invite expired")
    user_id = invite["user_id"]
    profile = await db.patient_profiles.find_one({"user_id": user_id}, {"_id": 0}) or {}
    latest = await db.prescriptions.find_one(
        {"user_id": user_id}, {"_id": 0, "image_b64": 0, "raw_ocr_text": 0},
        sort=[("created_at", -1)],
    )
    # Compute adherence inline (no auth user)
    events = await db.dose_events.find({"user_id": user_id}, {"_id": 0}).to_list(1000)
    now = now_utc()
    taken = missed = pending = delayed = 0
    for d in events:
        sched = d["scheduled_for"]
        if isinstance(sched, str):
            sched = datetime.fromisoformat(sched)
        if sched.tzinfo is None:
            sched = sched.replace(tzinfo=timezone.utc)
        status = d.get("status", "pending")
        if status == "pending" and sched < now - timedelta(hours=2):
            status = "missed"
        if status == "taken":
            taken += 1
        elif status == "missed":
            missed += 1
        elif status == "delayed":
            delayed += 1
        else:
            pending += 1
    total = taken + missed + delayed
    score = int(((taken + delayed*0.5) / total) * 100) if total else 100
    if missed >= 5:
        escalation = {"level": "critical", "label": "Critical alert"}
    elif missed >= 3:
        escalation = {"level": "warning", "label": "Caregiver notification"}
    elif missed >= 2:
        escalation = {"level": "warning", "label": "Warning"}
    elif missed >= 1:
        escalation = {"level": "info", "label": "Reminder"}
    else:
        escalation = {"level": "ok", "label": "All good"}
    # Strip sensitive fields
    safe_profile = {
        "full_name": profile.get("full_name"),
        "age": profile.get("age"),
        "sex": profile.get("sex"),
        "allergies": profile.get("allergies", []),
        "chronic_conditions": profile.get("chronic_conditions", []),
    }
    safe_latest = None
    if latest:
        safe_latest = {
            "doctor_name": latest.get("doctor_name"),
            "diagnosis": latest.get("diagnosis"),
            "risk_score": latest.get("risk_score"),
            "severity_label": latest.get("severity_label"),
            "visit_urgency": latest.get("visit_urgency"),
            "medicines": [
                {"name": m.get("name"), "dosage": m.get("dosage"), "frequency": m.get("frequency")}
                for m in (latest.get("medicines") or [])
            ],
            "alerts": [
                {"category": a.get("category"), "severity": a.get("severity"), "title": a.get("title")}
                for a in (latest.get("alerts") or [])
            ],
        }
    return {
        "patient": safe_profile,
        "latest_prescription": safe_latest,
        "adherence": {"score": score, "taken": taken, "missed": missed, "delayed": delayed, "pending": pending},
        "escalation": escalation,
    }


# -----------------------------------------------------------------------------
# PDF export
# -----------------------------------------------------------------------------
@api.get("/reports/{prescription_id}/pdf")
async def export_pdf(prescription_id: str, user=Depends(current_user)):
    rx = await db.prescriptions.find_one(
        {"prescription_id": prescription_id, "user_id": user.user_id}, {"_id": 0}
    )
    if not rx:
        raise HTTPException(status_code=404, detail="Not found")
    profile = await db.patient_profiles.find_one({"user_id": user.user_id}, {"_id": 0}) or {}
    pdf_bytes = _build_pdf(profile, rx)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="mediguard-{prescription_id}.pdf"'},
    )


def _build_pdf(profile: dict, rx: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    )
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm, bottomMargin=16*mm, title="MediGuard Report",
    )
    styles = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=styles["Heading1"], fontSize=22, textColor=colors.HexColor("#2B4C3B"))
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#666666"))
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14)
    bold = ParagraphStyle("bold", parent=body, fontName="Helvetica-Bold")
    story = []
    story.append(Paragraph("MediGuard AI · Patient Report", h))
    story.append(Paragraph(f"Generated {now_utc().strftime('%Y-%m-%d %H:%M UTC')}", sub))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Patient", bold))
    story.append(Paragraph(
        f"{profile.get('full_name','—')} · {profile.get('age','?')}y · {profile.get('sex','—')}<br/>"
        f"Allergies: {', '.join(profile.get('allergies') or []) or 'None reported'}<br/>"
        f"Conditions: {', '.join(profile.get('chronic_conditions') or []) or 'None reported'}",
        body,
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Prescription summary", bold))
    story.append(Paragraph(
        f"Doctor: {rx.get('doctor_name') or '—'}<br/>"
        f"Diagnosis: {rx.get('diagnosis') or '—'}<br/>"
        f"Risk score: <b>{rx.get('risk_score',0)}/100</b> · Severity: <b>{(rx.get('severity_label','') or '').upper()}</b> · Visit urgency: {rx.get('visit_urgency','').replace('_',' ')}",
        body,
    ))
    story.append(Spacer(1, 10))
    meds = rx.get("medicines", []) or []
    if meds:
        story.append(Paragraph("Medicines", bold))
        data = [["Name", "Dose", "Frequency", "Food", "Instructions"]]
        for m in meds:
            data.append([
                m.get("name", ""), m.get("dosage", "") or "", m.get("frequency") or f"{m.get('times_per_day',1)}x daily",
                m.get("food") or "", (m.get("instructions") or "")[:80],
            ])
        t = Table(data, repeatRows=1, colWidths=[40*mm, 22*mm, 30*mm, 25*mm, 55*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2B4C3B")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FAFAF9")]),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))
    alerts = rx.get("alerts", []) or []
    if alerts:
        story.append(Paragraph("Safety alerts", bold))
        for a in alerts:
            color = "#EF4444" if a.get("severity") in ("severe","critical") else "#F97316" if a.get("severity") == "moderate" else "#FBBF24"
            story.append(Paragraph(
                f"<font color='{color}'><b>[{(a.get('severity','') or '').upper()}]</b></font> "
                f"<b>{a.get('title','')}</b> ({a.get('category','')})<br/>{a.get('detail','')}<br/>"
                f"<i>Action: {a.get('action','')}</i>",
                body,
            ))
            story.append(Spacer(1, 6))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<i>This report is generated by MediGuard AI from real OCR, ML and AI analysis. "
        "It is not a substitute for professional medical advice.</i>", sub,
    ))
    doc.build(story)
    return buf.getvalue()


# Register router and middleware
app.include_router(api)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "https://medi-guard-ai-8444.vercel.app").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
