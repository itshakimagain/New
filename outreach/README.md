# Email Outreach Tool (Gmail)

Sends a personalized email to each company in your Excel file, e.g.
**"Hi Acme Logistics team,"**, slowly and safely through your Gmail account.

## One-time setup

1. **Install Python 3** (python.org), then in this folder run:
   ```
   pip install pyyaml openpyxl
   ```
2. **Create a Gmail App Password.** Your normal password will not work.
   - Turn on 2-Step Verification: https://myaccount.google.com/security
   - Create an App Password: https://myaccount.google.com/apppasswords
   - Copy `.env.example` to `.env` and paste the 16-character password into it.
3. **Edit `config.yaml`:** your name, Gmail address, business name and address, and subject line.
4. **Write your email in `template.txt`.**
5. **Prepare your Excel file** and save it here as `contacts.xlsx`.
   It needs an **Email** column and a **Company Name** column. Any other
   columns can be used as placeholders. See `contacts_sample.xlsx`.

## Daily use

| Command | What it does |
|---|---|
| `python send.py check` | Shows who will be emailed and who is skipped (invalid, duplicate, unsubscribed, already sent) |
| `python send.py preview 3` | Shows the first 3 emails exactly as they will look |
| `python send.py test` | Sends one sample email **to yourself** |
| `python send.py send` | Sends today's batch, up to `daily_limit`, with a random 45–120s gap between emails |
| `python send.py unsubscribe someone@co.com` | Never email this address again |

Run `send` once a day. It remembers who was already emailed (`sent_log.csv`),
so you can add more rows to your Excel file at any time. If you stop it
with Ctrl+C, progress is saved.

**When someone replies "UNSUBSCRIBE", run the `unsubscribe` command for them straight away.**

## Singapore compliance (Spam Control Act)

This is general information, not legal advice. The tool builds in:
- the `<ADV>` label at the start of the subject (required for bulk unsolicited
  commercial email, e.g. more than 1,000 within 30 days),
- an unsubscribe option in every email, plus a `List-Unsubscribe` header,
- your real name, business name and address,
- no fake headers and no misleading subjects. Keep your subject honest.

Honour unsubscribe requests within 10 business days (do it the same day if you can).
Email business addresses, and only for a genuine business reason.

## Tips to stay out of spam

- In the first week, set `daily_limit` to 20–30, then raise it gradually to 80–100.
- Keep emails short and plain, with at most 1–2 links and no attachments.
- Reply quickly to anyone who responds. Replies improve your sender reputation.
- If you see "3 failures in a row", Gmail is limiting you. Stop for 24 hours.
