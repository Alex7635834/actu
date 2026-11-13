#!/usr/bin/env python3
"""Envoie le digest du jour par mail via l'API Resend (HTTPS).

Variables d'environnement : RESEND_API_KEY, MAIL_TO (destinataire), MAIL_FROM (facultatif :
expéditeur sur un domaine vérifié chez Resend), SITE_URL (facultatif).
Usage : python sendmail.py            -> envoie
        python sendmail.py apercu.html -> écrit seulement le HTML dans ce fichier
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from html import escape as esc
from pathlib import Path

from build import DATA, date_fr

ACCENT = "#b4451f"


def render(day: dict, site_url: str) -> tuple[str, str]:
    """Renvoie (html, texte brut). Styles en ligne : les messageries ignorent souvent <style>."""
    h = [f'<div style="max-width:640px;margin:0 auto;font-family:Arial,Helvetica,sans-serif;'
         f'color:#1c1b19;font-size:15px;line-height:1.5">'
         f'<h1 style="font-family:Georgia,serif;font-size:26px;margin:0">Actu du jour</h1>'
         f'<p style="color:#6b6862;margin:2px 0 18px">{date_fr(day["date"])}</p>']
    t = [f"ACTU DU JOUR - {date_fr(day['date'])}", ""]
    if day["essentiel"]:
        h.append(f'<div style="border-left:4px solid {ACCENT};background:#f7f6f3;padding:12px 16px">'
                 f'<div style="font-size:12px;letter-spacing:1px;color:{ACCENT};font-weight:bold">'
                 f"L'ESSENTIEL</div><ul style=\"margin:6px 0 0;padding-left:18px\">"
                 + "".join(f'<li style="margin:5px 0">{esc(s)}</li>' for s in day["essentiel"])
                 + "</ul></div>")
        t += ["L'ESSENTIEL"] + [f"- {s}" for s in day["essentiel"]] + [""]
    for cat in day["categories"]:
        if not cat["items"]:
            continue
        h.append(f'<h2 style="font-family:Georgia,serif;font-size:20px;margin:28px 0 0;'
                 f'padding-bottom:5px;border-bottom:2px solid #1c1b19">{esc(cat["name"])}</h2>')
        t += [cat["name"].upper(), ""]
        for item in cat["items"]:
            h.append(f'<div style="padding:12px 0;border-bottom:1px solid #e4e1da">'
                     f'<div style="font-size:11px;color:#6b6862;text-transform:uppercase;'
                     f'letter-spacing:.5px">{esc(item["source"])}</div>'
                     f'<a href="{esc(item["link"])}" style="color:#1c1b19;font-weight:bold;'
                     f'font-size:16px;text-decoration:none">{esc(item["title"])}</a>'
                     f'<div style="margin-top:4px;color:#3a3936">{esc(item["resume"])}</div></div>')
            t += [f"{item['title']} ({item['source']})", item["resume"], item["link"], ""]
    note = ("Sélection et résumés générés par IA à partir des titres et extraits des flux RSS."
            if day["ai"] else "Résumés IA indisponibles ce jour : extraits bruts des flux RSS.")
    link = f' · <a href="{esc(site_url)}" style="color:{ACCENT}">Voir sur le site</a>' if site_url else ""
    h.append(f'<p style="color:#6b6862;font-size:12px;margin-top:24px">{note}{link}</p></div>')
    t.append(note)
    return "".join(h), "\n".join(t)


def main() -> None:
    files = sorted(DATA.glob("*.json"))
    if not files:
        sys.exit("Aucun digest dans data/ : lancer build.py d'abord.")
    day = json.loads(files[-1].read_text())
    html_body, text_body = render(day, os.environ.get("SITE_URL", ""))

    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(f'<!doctype html><meta charset="utf-8">{html_body}')
        print(f"Aperçu écrit dans {sys.argv[1]}")
        return

    key, to = os.environ.get("RESEND_API_KEY"), os.environ.get("MAIL_TO")
    if not key or not to:
        print("RESEND_API_KEY ou MAIL_TO absent : pas d'envoi de mail.")
        return
    body = json.dumps({
        "from": os.environ.get("MAIL_FROM") or "Actu du jour <onboarding@resend.dev>",
        "to": [to],
        "subject": f"Actu du {date_fr(day['date'])}",
        "html": html_body,
        "text": text_body,
    }).encode()
    for attempt in range(3):
        req = urllib.request.Request("https://api.resend.com/emails", data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "User-Agent": "actu-digest/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"Mail envoyé (id {json.loads(resp.read()).get('id')}).")
                return
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code != 429 and exc.code < 500:
                sys.exit(f"Resend refuse l'envoi : HTTP {exc.code} {detail}")
            print(f"  ! tentative {attempt + 1} : HTTP {exc.code} {detail}", file=sys.stderr)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  ! tentative {attempt + 1} : {exc}", file=sys.stderr)
        time.sleep(15)
    sys.exit("Envoi impossible après 3 tentatives.")

if __name__ == "__main__":
    main()
