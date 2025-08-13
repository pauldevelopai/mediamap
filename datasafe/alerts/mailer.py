import os
import smtplib
from email.mime.text import MIMEText
from typing import Optional
from ..config import ALERT_EMAIL_TO

def send_alert(subject: str, body: str, to: Optional[str] = None) -> bool:
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    to_addr = to or ALERT_EMAIL_TO
    if not (host and user and pwd and to_addr):
        return False
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    with smtplib.SMTP(host, 587, timeout=15) as s:
        s.starttls()
        s.login(user, pwd)
        s.sendmail(user, [to_addr], msg.as_string())
    return True


