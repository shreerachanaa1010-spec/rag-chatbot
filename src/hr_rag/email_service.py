from __future__ import annotations

import smtplib
from email.message import EmailMessage

from hr_rag.config import (
    HR_EMAIL_TO,
    SMTP_FROM,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USERNAME,
)


def send_hr_email(subject: str, body: str) -> None:
    """Send an internal HR case email through the configured SMTP server."""
    if not SMTP_HOST or not SMTP_FROM or not HR_EMAIL_TO:
        raise RuntimeError("HR email is not configured. Set SMTP_HOST, SMTP_FROM, and HR_EMAIL_TO.")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = SMTP_FROM
    message["To"] = HR_EMAIL_TO
    message.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
        server.starttls()
        if SMTP_USERNAME:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(message)