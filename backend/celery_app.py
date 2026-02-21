# backend/celery_app.py

from celery import Celery

celery = Celery(
    "verify",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=["backend.video_service", "backend.audio_service"],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"]
)
