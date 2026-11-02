"""Personalized email outreach through Gmail.

Commands:
    python send.py check               Validate the contact list and show who will be emailed
    python send.py preview [N]         Print the first N personalized emails (default 3)
    python send.py test                Send one sample email to YOURSELF
    python send.py send                Send today's batch (respects daily_limit)
    python send.py unsubscribe EMAIL   Add an address to the do-not-email list
"""

from __future__ import annotations

import csv
import os
import random
import re
import smtplib
import ssl
import sys
import time
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from pathlib import Path

import yaml
from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
SENT_LOG = HERE / "sent_log.csv"
UNSUBSCRIBED = HERE / "unsubscribed.txt"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SafeDict(dict):
    """Leaves unknown placeholders visible instead of crashing."""

    def __missing__(self, key):
        return "{" + key + "}"


# ---------------------------------------------------------------- loading

def load_config() -> dict:
    with open(HERE / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env() -> None:
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def normalize_header(h) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(h or "").strip().lower()).strip("_")


def load_contacts(path: Path) -> list[dict]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    headers = [normalize_header(h) for h in next(rows)]
    if "email" not in headers:
        sys.exit(f"Your Excel file needs an 'Email' column. Found: {headers}")
    contacts = []
    for i, row in enumerate(rows, start=2):
        rec = {h: ("" if v is None else str(v).strip()) for h, v in zip(headers, row) if h}
        if any(rec.values()):
            rec["_row"] = i
            contacts.append(rec)
    return contacts


def load_unsubscribed() -> set[str]:
    if not UNSUBSCRIBED.exists():
        return set()
    return {l.strip().lower() for l in UNSUBSCRIBED.read_text(encoding="utf-8").splitlines() if l.strip()}


def load_sent() -> list[dict]:
    if not SENT_LOG.exists():
        return []
    with open(SENT_LOG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def log_result(email: str, company: str, status: str, detail: str = "") -> None:
    new = not SENT_LOG.exists()
    with open(SENT_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "email", "company", "status", "detail"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), email, company, status, detail])


# ---------------------------------------------------------------- planning

def plan(cfg: dict) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Return (to_send, skipped_with_reason)."""
    contacts = load_contacts(HERE / cfg["contacts_file"])
    unsub = load_unsubscribed()
    already = {r["email"].lower() for r in load_sent() if r["status"] == "sent"}
    seen: set[str] = set()
    to_send, skipped = [], []
    for c in contacts:
        email = c.get("email", "").lower()
        if not email:
            skipped.append((c, "no email"))
        elif not EMAIL_RE.match(email):
            skipped.append((c, "invalid email"))
        elif email in unsub:
            skipped.append((c, "unsubscribed"))
        elif email in already:
            skipped.append((c, "already sent"))
        elif email in seen:
            skipped.append((c, "duplicate in file"))
        else:
            seen.add(email)
            to_send.append(c)
    return to_send, skipped


def sent_today() -> int:
    today = datetime.now().date().isoformat()
    return sum(1 for r in load_sent() if r["status"] == "sent" and r["timestamp"].startswith(today))


# ---------------------------------------------------------------- building

def build_message(cfg: dict, template: str, contact: dict, to_override: str | None = None) -> EmailMessage:
    s = cfg["sender"]
    fields = SafeDict(contact)
    fields.update(sender_name=s["name"], sender_business=s["business_name"])

    subject = cfg["subject"].format_map(fields)
    if cfg.get("adv_label"):
        subject = "<ADV> " + subject

    footer_lines = [
        "",
        "--",
        s["business_name"],
        s["business_address"],
    ]
    if s.get("phone"):
        footer_lines.append(s["phone"])
    footer_lines += [
        "",
        f"Don't want to hear from us? Reply with \"UNSUBSCRIBE\" or email "
        f"{s['email']} and you will not be contacted again.",
    ]

    body = template.format_map(fields).rstrip() + "\n" + "\n".join(footer_lines) + "\n"

    msg = EmailMessage()
    msg["From"] = formataddr((s["name"], s["email"]))
    msg["To"] = to_override or contact["email"]
    msg["Subject"] = subject
    msg["Message-ID"] = make_msgid(domain=s["email"].split("@")[-1])
    msg["List-Unsubscribe"] = f"<mailto:{s['email']}?subject=UNSUBSCRIBE>"
    msg.set_content(body)
    return msg


def smtp_connect(cfg: dict) -> smtplib.SMTP_SSL:
    load_env()
    pw = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "")
    if not pw:
        sys.exit("Missing GMAIL_APP_PASSWORD. Copy .env.example to .env and paste your App Password.")
    server = smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=60)
    server.login(cfg["sender"]["email"], pw)
    return server


# ---------------------------------------------------------------- commands

def cmd_check(cfg):
    to_send, skipped = plan(cfg)
    print(f"Ready to email: {len(to_send)}")
    print(f"Skipped:        {len(skipped)}")
    for c, why in skipped:
        print(f"  row {c['_row']}: {c.get('email') or '(blank)'} - {why}")
    print(f"\nSent today: {sent_today()} / {cfg['daily_limit']}")
    missing = [c for c in to_send if not c.get("company_name")]
    if missing:
        print(f"\nWarning: {len(missing)} contacts have no company name "
              f"(rows {', '.join(str(c['_row']) for c in missing[:10])}).")


def cmd_preview(cfg, n=3):
    template = (HERE / cfg["template_file"]).read_text(encoding="utf-8")
    to_send, _ = plan(cfg)
    for c in to_send[:n]:
        m = build_message(cfg, template, c)
        print("=" * 70)
        print(f"To:      {m['To']}\nSubject: {m['Subject']}\n")
        print(m.get_content())


def cmd_test(cfg):
    template = (HERE / cfg["template_file"]).read_text(encoding="utf-8")
    to_send, _ = plan(cfg)
    sample = to_send[0] if to_send else {"email": cfg["sender"]["email"], "company_name": "Sample Company"}
    msg = build_message(cfg, template, sample, to_override=cfg["sender"]["email"])
    with smtp_connect(cfg) as server:
        server.send_message(msg)
    print(f"Test email (personalized for {sample.get('company_name')}) sent to {cfg['sender']['email']}.")


def cmd_send(cfg):
    template = (HERE / cfg["template_file"]).read_text(encoding="utf-8")
    to_send, _ = plan(cfg)
    remaining = cfg["daily_limit"] - sent_today()
    if remaining <= 0:
        print("Daily limit reached. Run again tomorrow.")
        return
    batch = to_send[:remaining]
    if not batch:
        print("Nobody left to email.")
        return

    print(f"About to send {len(batch)} emails (daily limit {cfg['daily_limit']}). "
          f"{len(to_send) - len(batch)} will wait for later days.")
    if input("Type YES to start: ").strip() != "YES":
        print("Cancelled.")
        return

    server = smtp_connect(cfg)
    failures_in_a_row = 0
    try:
        for i, c in enumerate(batch, 1):
            msg = build_message(cfg, template, c)
            try:
                server.send_message(msg)
                log_result(c["email"], c.get("company_name", ""), "sent")
                failures_in_a_row = 0
                print(f"[{i}/{len(batch)}] sent -> {c['email']}")
            except smtplib.SMTPServerDisconnected:
                server = smtp_connect(cfg)
                server.send_message(msg)
                log_result(c["email"], c.get("company_name", ""), "sent")
                print(f"[{i}/{len(batch)}] sent -> {c['email']} (reconnected)")
            except smtplib.SMTPRecipientsRefused as e:
                log_result(c["email"], c.get("company_name", ""), "failed", str(e))
                print(f"[{i}/{len(batch)}] REFUSED {c['email']}")
            except smtplib.SMTPException as e:
                failures_in_a_row += 1
                log_result(c["email"], c.get("company_name", ""), "failed", str(e))
                print(f"[{i}/{len(batch)}] FAILED {c['email']}: {e}")
                if failures_in_a_row >= 3:
                    print("3 failures in a row - Gmail may be limiting you. Stopping for today.")
                    break
            if i < len(batch):
                time.sleep(random.uniform(cfg["delay_seconds_min"], cfg["delay_seconds_max"]))
    except KeyboardInterrupt:
        print("\nStopped. Progress is saved - run 'send' again to continue.")
    finally:
        try:
            server.quit()
        except Exception:
            pass
    print(f"Done. Sent today: {sent_today()} / {cfg['daily_limit']}")


def cmd_unsubscribe(email):
    email = email.strip().lower()
    if email in load_unsubscribed():
        print(f"{email} is already unsubscribed.")
        return
    with open(UNSUBSCRIBED, "a", encoding="utf-8") as f:
        f.write(email + "\n")
    print(f"{email} will never be emailed again.")


def main():
    args = sys.argv[1:]
    if not args or args[0] not in {"check", "preview", "test", "send", "unsubscribe"}:
        print(__doc__)
        return
    if args[0] == "unsubscribe":
        if len(args) < 2:
            sys.exit("Usage: python send.py unsubscribe someone@company.com")
        return cmd_unsubscribe(args[1])
    cfg = load_config()
    {"check": cmd_check, "test": cmd_test, "send": cmd_send}.get(
        args[0], lambda c: cmd_preview(c, int(args[1]) if len(args) > 1 else 3)
    )(cfg)


if __name__ == "__main__":
    main()
