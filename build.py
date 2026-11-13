#!/usr/bin/env python3
"""Digest quotidien : collecte RSS -> sélection IA -> lecture des articles -> rédaction IA
-> site statique dans _site/."""
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
from html.parser import HTMLParser
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


# ---------- appel à Gemini ----------

def call_gemini(key: str, model: str, system: str, prompt: str, schema: dict) -> dict:
    body = json.dumps({
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema},
    }).encode()
    req = urllib.request.Request(API_URL.format(model=model), data=body, headers={
        "x-goog-api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    parts = data["candidates"][0]["content"]["parts"]
    result = json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))
    if not isinstance(result, dict):
        raise ValueError("réponse JSON inattendue")
    usage = data.get("usageMetadata", {})
    print(f"  IA ({model}) : {usage.get('promptTokenCount')} tokens lus, "
          f"{usage.get('totalTokenCount')} au total")
    return result


def ask_gemini(key: str, system: str, prompt: str, schema: dict) -> dict | None:
    """Essaie chaque modèle de MODELS dans l'ordre, deux fois chacun. None si tout échoue."""
    for model in MODELS:
        for attempt in range(2):
            try:
                return call_gemini(key, model, system, prompt, schema)
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


# ---------- étape 1 : sélection ----------

SOURCE_CAP = 2        # articles max d'une même source dans une catégorie
UNE_TOPICS = 3        # sujets « à la une »
UNE_SOURCES = 4       # articles max rattachés à un sujet à la une

SYSTEM_SELECT = f"""Tu es le rédacteur en chef d'une veille d'actualité quotidienne pour un
ingénieur francophone. On te fournit des articles candidats (identifiant, source, titre, extrait
du flux RSS) classés par catégorie.

1. « picks » : pour chaque catégorie, les identifiants des articles à publier, du plus important
   au moins important. Donnes-en 3 de plus que le nombre demandé (ils servent de réserve).
   - Privilégie ce qui est important et instructif ; écarte promotions, bons plans, tests de
     produits mineurs et contenus sans information.
   - Un seul article par sujet : si plusieurs sources couvrent le même événement, garde le plus
     complet.
   - Varie les sources : pas plus de {SOURCE_CAP} articles d'une même source par catégorie.
   - Un même événement ne doit apparaître que dans une seule catégorie, la plus pertinente,
     sauf si un autre article l'aborde sous un angle réellement différent.
2. « une » : les {UNE_TOPICS} sujets majeurs du jour, toutes catégories confondues, en
   privilégiant tech, IA et cybersécurité quand l'actualité s'y prête. Un sujet couvert par
   plusieurs sources est un signal d'importance. Pour chaque sujet, donne dans « ids » les
   identifiants des articles candidats qui en parlent (1 à {UNE_SOURCES}, de sources différentes,
   toutes catégories confondues, qu'ils soient ou non dans « picks »)."""

SCHEMA_SELECT = {
    "type": "object",
    "properties": {
        "picks": {"type": "array", "items": {"type": "integer"}},
        "une": {"type": "array", "items": {
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
            "required": ["ids"]}},
    },
    "required": ["picks", "une"],
}


def cap_sources(items: list[dict], limit: int) -> list[dict]:
    """Garde l'ordre, mais au plus SOURCE_CAP articles par source."""
    kept, count = [], {}
    for item in items:
        if count.get(item["source"], 0) < SOURCE_CAP and len(kept) < limit:
            kept.append(item)
            count[item["source"]] = count.get(item["source"], 0) + 1
    return kept


def fallback_picks(categories, cands):
    """Sans IA : premiers articles en alternant les sources."""
    return {c["id"]: cap_sources(cands[c["id"]], c["picks"]) for c in categories}


def select(key, categories, cands):
    """Renvoie (picks par catégorie, groupes à la une) ou None si l'IA est indisponible."""
    parts = []
    for cat in categories:
        parts.append(f"## {cat['name']} : publier {cat['picks']} articles")
        parts += [f"[{i['id']}] {i['source']} | {i['title']} | {i['desc'] or '(pas d extrait)'}"
                  for i in cands[cat["id"]]]
    answer = ask_gemini(key, SYSTEM_SELECT, "\n".join(parts), SCHEMA_SELECT)
    if not answer or not isinstance(answer.get("picks"), list):
        return None
    by_id = {i["id"]: (cid, i) for cid, items in cands.items() for i in items}
    ranked, used = {c["id"]: [] for c in categories}, set()
    for pid in answer["picks"]:
        if pid in by_id and pid not in used:  # identifiant inventé ou en double : ignoré
            used.add(pid)
            ranked[by_id[pid][0]].append(by_id[pid][1])
    backup = fallback_picks(categories, cands)
    picks = {c["id"]: cap_sources(ranked[c["id"]], c["picks"]) or backup[c["id"]]
             for c in categories}
    groups = []
    for topic in answer.get("une") or []:
        ids = topic.get("ids") if isinstance(topic, dict) else None
        items, sources = [], set()
        for pid in ids if isinstance(ids, list) else []:
            if pid in by_id and by_id[pid][1]["source"] not in sources:
                items.append(by_id[pid][1])
                sources.add(by_id[pid][1]["source"])
        if items and len(groups) < UNE_TOPICS:
            groups.append(items[:UNE_SOURCES])
    return picks, groups


# ---------- étape 2 : lecture des articles ----------

ARTICLE_CHARS = 4000  # longueur max du texte d'article envoyé à l'IA
BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0 Safari/537.36")


class _Paragraphs(HTMLParser):
    """Récupère le texte des <p>, de préférence ceux situés dans <article>."""
    SKIP = {"script", "style", "nav", "footer", "aside", "form", "figure", "noscript", "svg",
            "button", "select"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = self.in_article = 0
        self.buf = None
        self.everywhere, self.in_articles = [], []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "article":
            self.in_article += 1
        elif tag == "p":
            self.buf = []

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "article":
            self.in_article = max(0, self.in_article - 1)
        elif tag == "p" and self.buf is not None:
            text = " ".join("".join(self.buf).split())
            self.buf = None
            if len(text) >= 50:
                self.everywhere.append(text)
                if self.in_article:
                    self.in_articles.append(text)

    def handle_data(self, data):
        if self.buf is not None and not self.skip:
            self.buf.append(data)


def fetch_text(item: dict) -> str:
    """Texte principal de l'article, ou "" s'il est inaccessible (paywall, blocage, erreur)."""
    try:
        req = urllib.request.Request(item["link"], headers={
            "User-Agent": BROWSER_UA, "Accept": "text/html", "Accept-Language": "fr,en;q=0.8"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            if "html" not in resp.headers.get_content_type():
                return ""
            raw = resp.read(2_000_000)
            charset = resp.headers.get_content_charset() or "utf-8"
        parser = _Paragraphs()
        parser.feed(raw.decode(charset, errors="replace"))
    except Exception:  # réseau, HTTP, encodage, HTML cassé : on se rabat sur l'extrait RSS
        return ""
    in_article = "\n".join(parser.in_articles)
    text = in_article if len(in_article) >= 600 else "\n".join(parser.everywhere)
    if len(text) < 400:
        return ""
    return text[:ARTICLE_CHARS]


def read_articles(items: list[dict]) -> None:
    with ThreadPoolExecutor(max_workers=10) as pool:
        for item, text in zip(items, pool.map(fetch_text, items)):
            item["text"] = text
    full = sum(1 for i in items if i["text"])
    print(f"  {full}/{len(items)} articles lus en entier (les autres : extrait RSS)")


# ---------- étape 3 : rédaction ----------

SYSTEM_WRITE = """Tu es le rédacteur d'une veille d'actualité quotidienne pour un ingénieur
francophone. On te fournit les articles retenus aujourd'hui : identifiant, source, titre original,
puis le texte de l'article (ou seulement l'extrait RSS quand le texte est inaccessible), ainsi
que les sujets « à la une » avec les articles qui s'y rapportent.

1. « articles » : pour CHAQUE article fourni, une entrée avec :
   - « titre » : le titre en français. S'il est en anglais, traduis-le fidèlement, sans
     l'embellir ni le dramatiser ; s'il est déjà en français, reprends-le tel quel. Garde les
     noms propres, noms de produits et sigles dans leur forme d'origine.
   - « resume » : 2 à 3 phrases en français, 60 mots maximum : les faits essentiels (qui, quoi,
     chiffres clés), puis ce que cela change. Ne répète pas le titre.
2. « une » : pour chaque sujet à la une, « sujet » (son numéro), « titre » (titre court en
   français) et « resume » (3 à 4 phrases, 80 mots maximum) : une synthèse de ce que rapportent
   les articles rattachés, en signalant ce qui reste incertain ou ce sur quoi ils divergent.
3. « concept » : choisis UN terme ou mécanisme technique (informatique, IA, cybersécurité ou
   finance) qui apparaît dans les articles du jour et qu'un ingénieur gagne à comprendre. Évite
   les termes de la liste « déjà expliqués ». Donne « terme », « id » (l'article où il apparaît)
   et « explication » : 4 à 6 phrases en français, soit ce que c'est, comment ça fonctionne, et
   pourquoi on en parle aujourd'hui.

Règles de fiabilité :
- Titres, résumés et synthèses s'appuient uniquement sur le texte fourni : n'ajoute aucun fait,
  chiffre, nom ou citation qui n'y figure pas. Si seul un extrait est disponible, reste bref.
- Écris les résumés directement, sans formule comme « selon cet extrait » ou « l'article
  indique » ; ne commente jamais le fait que le texte fourni est partiel.
- Pour l'explication du concept, tu peux t'appuyer sur tes connaissances générales, mais
  n'affirme que ce dont tu es sûr ; en cas de doute sur un détail, omets-le."""

SCHEMA_WRITE = {
    "type": "object",
    "properties": {
        "articles": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "titre": {"type": "string"},
                           "resume": {"type": "string"}},
            "required": ["id", "titre", "resume"]}},
        "une": {"type": "array", "items": {
            "type": "object",
            "properties": {"sujet": {"type": "integer"}, "titre": {"type": "string"},
                           "resume": {"type": "string"}},
            "required": ["sujet", "titre", "resume"]}},
        "concept": {
            "type": "object",
            "properties": {"terme": {"type": "string"}, "explication": {"type": "string"},
                           "id": {"type": "integer"}},
            "required": ["terme", "explication", "id"]},
    },
    "required": ["articles", "une", "concept"],
}


def recent_concepts(today: str, days: int = 21) -> list[str]:
    previous = [p for p in sorted(DATA.glob("*.json")) if p.stem < today][-days:]
    terms = [(json.loads(p.read_text()).get("concept") or {}).get("terme") for p in previous]
    return [t for t in terms if t]


def write(key, items: list[dict], groups: list[list[dict]], already: list[str]):
    """Renvoie (textes par id, textes à la une, concept) ou None si l'IA est indisponible."""
    parts = ["### SUJETS À LA UNE"]
    parts += [f"Sujet {n} : articles " + ", ".join(f"[{i['id']}]" for i in group)
              for n, group in enumerate(groups, 1)]
    parts.append("\n### CONCEPTS DÉJÀ EXPLIQUÉS\n" + (", ".join(already) or "(aucun)"))
    parts.append("\n### ARTICLES")
    for item in items:
        body = (f"Texte : {item['text']}" if item["text"]
                else f"Extrait seulement : {item['desc'] or '(aucun)'}")
        parts.append(f"\n[{item['id']}] {item['source']} | {item['title']}\n{body}")
    answer = ask_gemini(key, SYSTEM_WRITE, "\n".join(parts), SCHEMA_WRITE)
    if not answer or not isinstance(answer.get("articles"), list):
        return None
    by_id = {i["id"]: i for i in items}
    texts = {}
    for entry in answer["articles"]:
        if isinstance(entry, dict) and entry.get("id") in by_id and entry.get("resume"):
            texts[entry["id"]] = (clean(str(entry.get("titre") or "")), clean(str(entry["resume"])))
    une = {}
    for entry in answer.get("une") or []:
        if isinstance(entry, dict) and entry.get("titre") and entry.get("resume"):
            une[entry.get("sujet")] = (clean(str(entry["titre"])), clean(str(entry["resume"])))
    concept, raw = None, answer.get("concept")
    if isinstance(raw, dict) and raw.get("terme") and raw.get("explication"):
        linked = by_id.get(raw.get("id"))
        concept = {"terme": clean(str(raw["terme"])), "explication": clean(str(raw["explication"])),
                   "link": linked["link"] if linked else "",
                   "source": linked["source"] if linked else ""}
    return texts, une, concept


# ---------- rendu ----------

CSS = """
:root{--bg:#f7f6f3;--card:#fff;--fg:#1c1b19;--muted:#6b6862;--line:#e4e1da;--accent:#b4451f;--soft:#f1ede6}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1d1b;--fg:#ecebe7;--muted:#9b978f;--line:#2e2c29;--accent:#e8825c;--soft:#23211e}}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:56px}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main,header,footer{max-width:720px;margin:0 auto;padding:0 16px}
header{padding-top:28px}
h1{font:700 1.9rem/1.15 Georgia,"Times New Roman",serif;margin:0}
.date{color:var(--muted);margin:4px 0 0}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:1}
nav div{max-width:720px;margin:0 auto;padding:10px 16px;display:flex;gap:8px;overflow-x:auto;scrollbar-width:none}
nav a{flex:none;padding:4px 12px;border:1px solid var(--line);border-radius:999px;color:var(--fg);text-decoration:none;font-size:.85rem;background:var(--card)}
.label{font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);font-weight:700;margin:0 0 6px}
.box{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:8px;padding:14px 18px;margin:20px 0}
.box ul{margin:0;padding-left:18px}.box li{margin:6px 0}
.box h3{font:700 1.15rem/1.3 Georgia,serif;margin:0 0 6px}.box p{margin:0}
.une{margin-top:22px}
.une article{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 18px;margin:10px 0}
.une h3{font:700 1.2rem/1.3 Georgia,serif;margin:0 0 6px}
.une p{margin:0}
.refs{font-size:.85rem;color:var(--muted);margin-top:8px}
.refs a{color:var(--accent);text-decoration:none;font-weight:600}
section h2{font:700 1.3rem Georgia,serif;margin:32px 0 4px;padding-bottom:6px;border-bottom:2px solid var(--fg)}
.list article{padding:14px 0;border-bottom:1px solid var(--line)}
.list article>a{color:var(--fg);text-decoration:none;font-weight:650;font-size:1.05rem;line-height:1.3;display:block}
.list article>a:hover{color:var(--accent);text-decoration:underline}
.list article p{margin:5px 0 0;opacity:.88}
.src{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
.src span{text-transform:none;letter-spacing:0;font-style:italic}
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


def footnote(day: dict) -> str:
    if not day["ai"]:
        return "Résumés IA indisponibles ce jour : extraits bruts des flux RSS."
    if "une" not in day:
        return ("Sélection et résumés générés par IA à partir des titres et extraits des flux RSS : "
                "ouvrez l'article pour le détail.")
    return ("Sélection, titres français et résumés générés par IA à partir du texte des articles ; "
            "« d'après l'extrait » signale un article qui n'a pas pu être lu en entier. "
            "Le lien mène toujours à l'article original.")


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


def link(href: str, text: str) -> str:
    return f'<a href="{esc(href)}" target="_blank" rel="noopener">{esc(text)}</a>'


def render_day(day: dict, prev: str | None) -> str:
    cats = [c for c in day["categories"] if c["items"]]
    une, concept = day.get("une") or [], day.get("concept")
    chips = ([("une", "À la une")] if une else []) + ([("concept", "Concept")] if concept else []) \
        + [(c["id"], c["name"]) for c in cats]
    nav = "<nav><div>" + "".join(f'<a href="#{i}">{esc(n)}</a>' for i, n in chips) + "</div></nav>"
    body = [f'<header><h1>Actu du jour</h1><p class="date">{date_fr(day["date"])}</p></header><main>']
    if une:
        body.append('<section class="une" id="une"><p class="label">À la une</p>')
        for topic in une:
            refs = " · ".join(link(s["link"], s["source"]) for s in topic["sources"])
            body.append(f'<article><h3>{esc(topic["titre"])}</h3><p>{esc(topic["resume"])}</p>'
                        f'<div class="refs">Sources : {refs}</div></article>')
        body.append("</section>")
    if day.get("essentiel"):  # ancien format, conservé pour les archives
        body.append('<div class="box"><p class="label">L\'essentiel</p><ul>'
                    + "".join(f"<li>{esc(s)}</li>" for s in day["essentiel"]) + "</ul></div>")
    if concept:
        ref = (f'<div class="refs">Vu dans : {link(concept["link"], concept["source"])}</div>'
               if concept.get("link") else "")
        body.append(f'<div class="box" id="concept"><p class="label">Concept du jour</p>'
                    f'<h3>{esc(concept["terme"])}</h3><p>{esc(concept["explication"])}</p>{ref}</div>')
    for cat in cats:
        body.append(f'<section class="list" id="{cat["id"]}"><h2>{esc(cat["name"])}</h2>')
        for item in cat["items"]:
            resume = f"<p>{esc(item['resume'])}</p>" if item["resume"] else ""
            partial = " <span>· d'après l'extrait</span>" if item.get("full") is False else ""
            body.append(f'<article><div class="src">{esc(item["source"])}{partial}</div>'
                        f'{link(item["link"], item.get("titre") or item["title"])}{resume}</article>')
        body.append("</section>")
    links = (f'<a href="{prev}.html">← Jour précédent</a> · ' if prev else "") \
        + '<a href="archives.html">Archives</a>'
    body.append(f'</main><footer><p>{footnote(day)}</p>'
                f'<p>{links} · Mis à jour à {day["generated_at"]}</p></footer>')
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

    key = os.environ.get("GEMINI_API_KEY")
    picks, groups, written = fallback_picks(categories, cands), [], None
    if not key:
        print("GEMINI_API_KEY absente : digest sans IA")
    else:
        print("Sélection…")
        selection = select(key, categories, cands)
        if selection:
            picks, groups = selection
            wanted = {i["id"]: i for items in picks.values() for i in items}
            wanted.update({i["id"]: i for group in groups for i in group})
            print("Lecture des articles…")
            read_articles(list(wanted.values()))
            print("Rédaction…")
            written = write(key, list(wanted.values()), groups, recent_concepts(today))
    texts, une_texts, concept = written or ({}, {}, None)

    def entry(item: dict) -> dict:
        titre, resume = texts.get(item["id"], ("", ""))
        return {"title": item["title"], "titre": titre or item["title"], "link": item["link"],
                "source": item["source"], "resume": resume or clean(item["desc"], 220),
                "full": bool(item.get("text")) if written else None}

    day = {"date": today, "generated_at": now.strftime("%H:%M"), "ai": bool(written),
           "une": [{"titre": une_texts[n][0], "resume": une_texts[n][1],
                    "sources": [{"source": i["source"], "link": i["link"]} for i in group]}
                   for n, group in enumerate(groups, 1) if n in une_texts],
           "concept": concept,
           "categories": [{"id": c["id"], "name": c["name"],
                           "items": [entry(i) for i in picks[c["id"]]]} for c in categories]}
    (DATA / f"{today}.json").write_text(json.dumps(day, ensure_ascii=False, indent=1))
    render_site()
    print(f"OK : {sum(len(c['items']) for c in day['categories'])} articles, "
          f"{len(day['une'])} sujets à la une -> {OUT}/index.html")


if __name__ == "__main__":
    main()
