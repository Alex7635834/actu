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

from build import DATA, date_fr, footnote

ACCENT = "#b4451f"


def a(href: str, text: str, style: str) -> str:
    return f'<a href="{esc(href)}" style="{style}">{esc(text)}</a>'


def render(day: dict, site_url: str) -> tuple[str, str]:
    """Renvoie (html, texte brut). Styles en ligne : les messageries ignorent souvent <style>."""
    label = f"font-size:12px;letter-spacing:1px;color:{ACCENT};font-weight:bold"
    serif = "font-family:Georgia,serif;font-weight:bold"
    h = [f'<div style="max-width:640px;margin:0 auto;font-family:Arial,Helvetica,sans-serif;'
         f'color:#1c1b19;font-size:15px;line-height:1.5">'
         f'<h1 style="{serif};font-size:26px;margin:0">Actu du jour</h1>'
         f'<p style="color:#6b6862;margin:2px 0 18px">{date_fr(day["date"])}</p>']
    t = [f"ACTU DU JOUR - {date_fr(day['date'])}", ""]
    if day.get("une"):
        h.append(f'<div style="{label}">À LA UNE</div>')
        t += ["À LA UNE", ""]
        for topic in day["une"]:
            refs = " · ".join(a(s["link"], s["source"], f"color:{ACCENT};text-decoration:none")
                              for s in topic["sources"])
            h.append(f'<div style="border:1px solid #e4e1da;border-radius:8px;padding:12px 16px;'
                     f'margin:8px 0"><div style="{serif};font-size:18px;line-height:1.3">'
                     f'{esc(topic["titre"])}</div><div style="margin-top:5px">'
                     f'{esc(topic["resume"])}</div><div style="font-size:13px;color:#6b6862;'
                     f'margin-top:6px">Sources : {refs}</div></div>')
            t += [topic["titre"], topic["resume"]] + [s["link"] for s in topic["sources"]] + [""]
    if day.get("essentiel"):  # ancien format
        h.append(f'<div style="border-left:4px solid {ACCENT};background:#f7f6f3;padding:12px 16px">'
                 f'<div style="{label}">L\'ESSENTIEL</div><ul style="margin:6px 0 0;padding-left:18px">'
                 + "".join(f'<li style="margin:5px 0">{esc(s)}</li>' for s in day["essentiel"])
                 + "</ul></div>")
        t += ["L'ESSENTIEL"] + [f"- {s}" for s in day["essentiel"]] + [""]
    concept = day.get("concept")
    if concept:
        ref = (f'<div style="font-size:13px;color:#6b6862;margin-top:6px">Vu dans : '
               + a(concept["link"], concept["source"], f"color:{ACCENT};text-decoration:none")
               + "</div>") if concept.get("link") else ""
        h.append(f'<div style="border-left:4px solid {ACCENT};background:#f7f6f3;padding:12px 16px;'
                 f'margin-top:16px"><div style="{label}">CONCEPT DU JOUR</div>'
                 f'<div style="{serif};font-size:17px;margin:4px 0">{esc(concept["terme"])}</div>'
                 f'<div>{esc(concept["explication"])}</div>{ref}</div>')
        t += ["CONCEPT DU JOUR : " + concept["terme"], concept["explication"], ""]
    def side(heading: str, title: str, text: str, refs: str) -> None:
        h.append(f'<div style="border-left:4px solid {ACCENT};background:#f7f6f3;padding:12px 16px;'
                 f'margin-top:16px"><div style="{label}">{esc(heading.upper())}</div>'
                 f'<div style="{serif};font-size:17px;margin:4px 0">{esc(title)}</div>'
                 f'<div>{esc(text)}</div><div style="font-size:13px;color:#6b6862;margin-top:6px">'
                 f'{refs}</div></div>')
        t.extend([f"{heading.upper()} : {title}", text, ""])

    ref_style = f"color:{ACCENT};text-decoration:none"
    art = day.get("art")
    if art:
        side(art["rubrique"], art["titre"], art["texte"],
             "Source : " + a(art["link"], art["source"], ref_style))
    for cat in day["categories"]:
        if not cat["items"]:
            continue
        h.append(f'<h2 style="{serif};font-size:20px;margin:28px 0 0;'
                 f'padding-bottom:5px;border-bottom:2px solid #1c1b19">{esc(cat["name"])}</h2>')
        t += [cat["name"].upper(), ""]
        if cat.get("bref"):
            h.append(f'<div style="background:#f7f6f3;border-radius:6px;padding:10px 14px;'
                     f'margin-top:10px">{esc(cat["bref"])}</div>')
            t += [cat["bref"], ""]
        for item in cat["items"]:
            titre = item.get("titre") or item["title"]
            partial = " · d'après l'extrait" if item.get("full") is False else ""
            h.append(f'<div style="padding:12px 0;border-bottom:1px solid #e4e1da">'
                     f'<div style="font-size:11px;color:#6b6862;letter-spacing:.5px">'
                     f'{esc(item["source"].upper())}{partial}</div>'
                     + a(item["link"], titre, "color:#1c1b19;font-weight:bold;font-size:16px;"
                                              "text-decoration:none")
                     + f'<div style="margin-top:4px;color:#3a3936">{esc(item["resume"])}</div></div>')
            t += [f"{titre} ({item['source']})", item["resume"], item["link"], ""]
    hist, pays = day.get("histoire"), day.get("pays")
    if hist:
        side(f"Ce jour-là · {hist['annee']}", hist["titre"], hist["texte"],
             "Source : " + a(hist["link"], hist["source"], ref_style))
    if pays:
        side("Le pays du jour", f"{pays['nom']} (capitale : {pays['capitale']})", pays["texte"],
             a(pays["wiki"], "Wikipédia", ref_style) + " · Actualité : "
             + a(pays["link"], pays["source"], ref_style))
    note = footnote(day)
    more = (" · " + a(site_url, "Voir sur le site", f"color:{ACCENT}")) if site_url else ""
    h.append(f'<p style="color:#6b6862;font-size:12px;margin-top:24px">{esc(note)}{more}</p></div>')
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
