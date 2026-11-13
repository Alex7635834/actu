#!/usr/bin/env python3
"""Digest quotidien : collecte RSS -> sélection et résumés IA -> site statique dans _site/."""
from __future__ import annotations

import calendar
import html
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "_site"
TZ = ZoneInfo("America/Toronto")  # fuseau de la date et de l'heure affichées
MAX_AGE_H = 36          # on ignore les articles plus vieux
PER_FEED = 8            # articles max lus par flux
PER_CATEGORY = 40       # candidats max envoyés à l'IA par catégorie
SEEN_DAYS = 3           # un article déjà publié ces N derniers jours n'est pas repris
# Modèles Gemini essayés dans l'ordre (tous disponibles dans l'offre gratuite).
MODELS = (os.environ.get("ACTU_MODELS") or "gemini-4-hydrogen,gemini-3.8-flash,gemini-3.5-flash-lite").split(",")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
UA = "Mozilla/5.0 (compatible; actu-digest/1.0)"

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
        "septembre", "octobre", "novembre", "décembre"]


# ---------- collecte ----------

def clean(text: str, limit: int = 0) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if limit and len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text


def norm_link(link: str) -> str:
    u = urllib.parse.urlsplit(link)
    query = [(k, v) for k, v in urllib.parse.parse_qsl(u.query) if not k.startswith("utm_")]
    return urllib.parse.urlunsplit((u.scheme, u.netloc.lower(), u.path.rstrip("/"),
                                    urllib.parse.urlencode(query), ""))


def fetch(feed: dict) -> list[dict]:
    """Lit un flux. Une source en panne renvoie [] sans faire échouer le reste."""
    try:
        req = urllib.request.Request(feed["url"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            parsed = feedparser.parse(resp.read())
    except Exception as exc:  # réseau, HTTP, flux invalide : on continue sans cette source
        print(f"  ! {feed['name']}: {exc}", file=sys.stderr)
        return []
    now, items = time.time(), []
    for entry in parsed.entries[:30]:
        link, title = entry.get("link"), clean(entry.get("title", ""))
        if not link or not link.startswith("http") or not title:
            continue
        stamp = entry.get("published_parsed") or entry.get("updated_parsed")
        if stamp and now - calendar.timegm(stamp) > MAX_AGE_H * 3600:
            continue
        desc = clean(entry.get("summary", ""), 400)
        if desc.startswith("Article URL:"):  # Hacker News : pas d'extrait, juste des liens
            desc = ""
        items.append({"title": title, "link": link, "source": feed["name"], "desc": desc})
        if len(items) >= PER_FEED:
            break
    if not items:
        print(f"  ! {feed['name']}: aucun article récent", file=sys.stderr)
    return items


def seen_links(today: str) -> set[str]:
    seen = set()
    previous = [p for p in sorted(DATA.glob("*.json")) if p.stem < today]
    for path in previous[-SEEN_DAYS:]:
        for cat in json.loads(path.read_text())["categories"]:
            seen.update(norm_link(i["link"]) for i in cat["items"])
    return seen


def collect(categories: list[dict], today: str) -> dict[str, list[dict]]:
    feeds = [f for c in categories for f in c["feeds"]]
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = dict(zip((f["url"] for f in feeds), pool.map(fetch, feeds)))
    seen, seen_titles, next_id = seen_links(today), set(), 1
    out = {}
    for cat in categories:
        per_feed = [results[f["url"]] for f in cat["feeds"]]
        cands = []
        for rank in range(PER_FEED):  # round-robin : on alterne les sources
            for items in per_feed:
                if rank >= len(items) or len(cands) >= PER_CATEGORY:
                    continue
                item = items[rank]
                key, tkey = norm_link(item["link"]), item["title"].lower()
                if key in seen or tkey in seen_titles:
                    continue
                seen.add(key)
                seen_titles.add(tkey)
                cands.append({**item, "id": next_id})
                next_id += 1
        out[cat["id"]] = cands
        print(f"  {cat['name']}: {len(cands)} candidats")
    return out


# ---------- sélection + résumés ----------

SYSTEM = """Tu es le rédacteur d'une veille d'actualité quotidienne pour un ingénieur francophone.
On te fournit des articles candidats (titre, source, extrait du flux RSS) classés par catégorie.

Ta tâche :
1. Pour chaque catégorie, choisis le nombre d'articles demandé, les plus importants et les plus
   instructifs. Un sujet repris par plusieurs sources est un signal d'importance : ne garde alors
   qu'un seul article, le plus complet. Écarte les promotions, bons plans, tests de produits
   mineurs et contenus sans information. S'il n'y a pas assez d'articles valables, prends-en moins.
2. Pour chaque article retenu, écris un résumé en français de 1 à 2 phrases (40 mots maximum) :
   le fait, puis pourquoi il compte. Ne répète pas le titre.
3. Dans « essentiel », écris 3 à 5 phrases courtes en français sur ce qu'il faut retenir de la
   journée, toutes catégories confondues, en privilégiant tech, IA et cybersécurité.

Tu ne disposes que du titre et de l'extrait, pas de l'article complet : appuie-toi uniquement sur
ces éléments et n'ajoute aucun fait, chiffre ou nom qui n'y figure pas. Si l'extrait est trop mince
pour dire pourquoi l'article compte, contente-toi de reformuler le fait.

Réponds en JSON : {"essentiel": [phrases], "picks": [{"id": numéro entre crochets de l'article,
"resume": "..."}]}."""


def fallback_picks(categories, cands):
    """Sans IA : premiers articles de chaque source, avec l'extrait RSS comme résumé."""
    return {c["id"]: [(i, clean(i["desc"], 220)) for i in cands[c["id"]][:c["picks"]]]
            for c in categories}


SCHEMA = {
    "type": "object",
    "properties": {
        "essentiel": {"type": "array", "items": {"type": "string"}},
        "picks": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "resume": {"type": "string"}},
            "required": ["id", "resume"]}},
    },
    "required": ["essentiel", "picks"],
}


def call_gemini(key: str, model: str, prompt: str) -> dict:
    body = json.dumps({
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA},
    }).encode()
    req = urllib.request.Request(API_URL.format(model=model), data=body, headers={
        "x-goog-api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    parts = data["candidates"][0]["content"]["parts"]
    digest = json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))
    usage = data.get("usageMetadata", {})
    print(f"  IA ({model}) : {usage.get('promptTokenCount')} tokens lus, "
          f"{usage.get('totalTokenCount')} au total")
    return digest


def ask_gemini(key: str, prompt: str) -> dict | None:
    """Essaie chaque modèle de MODELS dans l'ordre, deux fois chacun. None si tout échoue."""
    for model in MODELS:
        for attempt in range(2):
            try:
                return call_gemini(key, model, prompt)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                print(f"  ! IA ({model}) : HTTP {exc.code} {detail}", file=sys.stderr)
                if exc.code in (401, 403):
                    return None  # clé invalide : aucun modèle ne marchera
                if exc.code != 429 and exc.code < 500:
                    break        # requête refusée par ce modèle : on passe au suivant
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError) as exc:
                print(f"  ! IA ({model}) : {type(exc).__name__} {exc}", file=sys.stderr)
            if attempt == 0:
                time.sleep(30)
    return None


def summarize(categories, cands):
    """Renvoie (essentiel, picks) ou None si l'IA est indisponible."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("  GEMINI_API_KEY absente : digest sans résumés IA")
        return None
    parts = []
    for cat in categories:
        parts.append(f"## {cat['name']} : retenir {cat['picks']} articles")
        parts += [f"[{i['id']}] {i['source']} | {i['title']} | {i['desc'] or '(pas d extrait)'}"
                  for i in cands[cat["id"]]]
    digest = ask_gemini(key, "\n".join(parts))
    if not isinstance(digest, dict) or not isinstance(digest.get("picks"), list):
        return None

    by_id = {i["id"]: (cid, i) for cid, items in cands.items() for i in items}
    picks = {c["id"]: [] for c in categories}
    limits = {c["id"]: c["picks"] for c in categories}
    for pick in digest["picks"]:
        if not isinstance(pick, dict) or pick.get("id") not in by_id:
            continue  # identifiant inventé : ignoré
        cid, item = by_id.pop(pick["id"])
        if len(picks[cid]) < limits[cid]:
            picks[cid].append((item, clean(str(pick.get("resume", "")))))
    backup = fallback_picks(categories, cands)
    for cid, chosen in picks.items():  # catégorie oubliée par l'IA : on la remplit quand même
        if not chosen:
            picks[cid] = backup[cid]
    essentiel = [clean(str(s)) for s in digest.get("essentiel") or []]
    return essentiel, picks


# ---------- rendu ----------

CSS = """
:root{--bg:#f7f6f3;--card:#fff;--fg:#1c1b19;--muted:#6b6862;--line:#e4e1da;--accent:#b4451f}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1d1b;--fg:#ecebe7;--muted:#9b978f;--line:#2e2c29;--accent:#e8825c}}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:56px}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main,header,footer{max-width:720px;margin:0 auto;padding:0 16px}
header{padding-top:28px}
h1{font:700 1.9rem/1.15 Georgia,"Times New Roman",serif;margin:0}
.date{color:var(--muted);margin:4px 0 0}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:1}
nav div{max-width:720px;margin:0 auto;padding:10px 16px;display:flex;gap:8px;overflow-x:auto;scrollbar-width:none}
nav a{flex:none;padding:4px 12px;border:1px solid var(--line);border-radius:999px;color:var(--fg);text-decoration:none;font-size:.85rem;background:var(--card)}
.box{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:8px;padding:14px 18px;margin:20px 0}
.box h2{font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin:0 0 6px}
.box ul{margin:0;padding-left:18px}.box li{margin:6px 0}
section h2{font:700 1.3rem Georgia,serif;margin:32px 0 4px;padding-bottom:6px;border-bottom:2px solid var(--fg)}
article{padding:14px 0;border-bottom:1px solid var(--line)}
article a{color:var(--fg);text-decoration:none;font-weight:650;font-size:1.05rem;line-height:1.3;display:block}
article a:hover{color:var(--accent);text-decoration:underline}
article p{margin:5px 0 0;color:var(--fg);opacity:.85}
.src{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
footer{color:var(--muted);font-size:.85rem;padding:28px 16px 40px}
footer a{color:var(--accent)}
.days{list-style:none;padding:0}.days li{padding:10px 0;border-bottom:1px solid var(--line)}
.days a{color:var(--fg);font-weight:600;text-decoration:none}
"""

ICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><rect width="512" height="512" '
        'rx="96" fill="#b4451f"/><text x="256" y="350" font-family="Georgia,serif" font-size="300" '
        'font-weight="700" text-anchor="middle" fill="#fff">A</text></svg>')

MANIFEST = {"name": "Actu du jour", "short_name": "Actu", "start_url": ".", "display": "standalone",
            "background_color": "#f7f6f3", "theme_color": "#b4451f",
            "icons": [{"src": "icon.svg", "sizes": "any", "type": "image/svg+xml"}]}

esc = html.escape


def date_fr(iso: str) -> str:
    d = datetime.strptime(iso, "%Y-%m-%d")
    return f"{JOURS[d.weekday()]} {d.day} {MOIS[d.month - 1]} {d.year}"


def page(title: str, body: str, nav: str = "") -> str:
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#b4451f">
<title>{esc(title)}</title>
<link rel="manifest" href="manifest.webmanifest"><link rel="icon" href="icon.svg">
<link rel="apple-touch-icon" href="icon.svg">
<style>{CSS}</style></head><body>
{nav}{body}
</body></html>"""


def render_day(day: dict, prev: str | None) -> str:
    cats = [c for c in day["categories"] if c["items"]]
    nav = "<nav><div>" + "".join(
        f'<a href="#{c["id"]}">{esc(c["name"])}</a>' for c in cats) + "</div></nav>"
    body = [f'<header><h1>Actu du jour</h1><p class="date">{date_fr(day["date"])}</p></header><main>']
    if day["essentiel"]:
        body.append('<div class="box"><h2>L\'essentiel</h2><ul>'
                    + "".join(f"<li>{esc(s)}</li>" for s in day["essentiel"]) + "</ul></div>")
    for cat in cats:
        body.append(f'<section id="{cat["id"]}"><h2>{esc(cat["name"])}</h2>')
        for item in cat["items"]:
            resume = f"<p>{esc(item['resume'])}</p>" if item["resume"] else ""
            body.append(f'<article><div class="src">{esc(item["source"])}</div>'
                        f'<a href="{esc(item["link"])}" target="_blank" rel="noopener">'
                        f'{esc(item["title"])}</a>{resume}</article>')
        body.append("</section>")
    note = ("Sélection et résumés générés par IA à partir des titres et extraits des flux RSS : "
            "ouvrez l'article pour le détail." if day["ai"]
            else "Résumés IA indisponibles ce jour : extraits bruts des flux RSS.")
    links = (f'<a href="{prev}.html">← Jour précédent</a> · ' if prev else "") \
        + '<a href="archives.html">Archives</a>'
    body.append(f'</main><footer><p>{note}</p><p>{links} · Mis à jour à {day["generated_at"]}</p></footer>')
    return page(f"Actu du {date_fr(day['date'])}", "".join(body), nav)


def render_site() -> None:
    OUT.mkdir(exist_ok=True)
    days = [json.loads(p.read_text()) for p in sorted(DATA.glob("*.json"))]
    for idx, day in enumerate(days):
        prev = days[idx - 1]["date"] if idx else None
        doc = render_day(day, prev)
        (OUT / f"{day['date']}.html").write_text(doc)
        if idx == len(days) - 1:
            (OUT / "index.html").write_text(doc)
    rows = "".join(f'<li><a href="{d["date"]}.html">{date_fr(d["date"])}</a></li>'
                   for d in reversed(days))
    (OUT / "archives.html").write_text(page(
        "Archives", f'<header><h1>Archives</h1><p class="date"><a href="index.html">'
                    f'← Aujourd\'hui</a></p></header><main><ul class="days">{rows}</ul></main>'))
    (OUT / "icon.svg").write_text(ICON)
    (OUT / "manifest.webmanifest").write_text(json.dumps(MANIFEST))


# ---------- main ----------

def main() -> None:
    categories = tomllib.loads((ROOT / "sources.toml").read_text())["category"]
    now = datetime.now(TZ)
    today = now.strftime("%Y-%m-%d")
    DATA.mkdir(exist_ok=True)

    print("Collecte des flux…")
    cands = collect(categories, today)
    if sum(len(v) for v in cands.values()) == 0:
        sys.exit("Aucun article collecté (réseau coupé ?) : on ne publie rien.")

    print("Sélection et résumés…")
    result = summarize(categories, cands)
    essentiel, picks = result if result else ([], fallback_picks(categories, cands))

    day = {"date": today, "generated_at": now.strftime("%H:%M"), "ai": bool(result),
           "essentiel": essentiel,
           "categories": [{"id": c["id"], "name": c["name"],
                           "items": [{"title": i["title"], "link": i["link"],
                                      "source": i["source"], "resume": r}
                                     for i, r in picks[c["id"]]]} for c in categories]}
    (DATA / f"{today}.json").write_text(json.dumps(day, ensure_ascii=False, indent=1))
    render_site()
    print(f"OK : {sum(len(c['items']) for c in day['categories'])} articles -> {OUT}/index.html")


if __name__ == "__main__":
    main()
