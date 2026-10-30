"""Online tools: Wikimedia projects, YouTube, Instagram, GitHub, Hugging Face and a table of free public APIs.

Everything here uses official, documented APIs with a descriptive User-Agent — no scraping of logged-in sites.
Tools that need an account read their key from the Connections panel (or the environment); posting tools have risk "post"
and are switched off until the user allows them.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse

import httpx

from .registry import Context, ToolError, reg

UA = "AI-Room/0.3 (+https://github.com/tharunmaks/ai-room; educational use)"
N = "web"
NET = dict(ctx=True, risk="network")


def client(ctx: Context) -> httpx.Client:
    return ctx.http or httpx.Client(timeout=25, headers={"User-Agent": UA, "Accept": "application/json"}, follow_redirects=True)


_last: dict[str, float] = {}


def get(ctx: Context, url: str, params: dict | None = None, headers: dict | None = None, as_text: bool = False, host_gap: float = 0.3):
    host = urllib.parse.urlparse(url).netloc
    wait = host_gap - (time.time() - _last.get(host, 0))
    if wait > 0 and ctx.http is None:
        time.sleep(wait)
    _last[host] = time.time()
    if urllib.parse.urlparse(url).scheme != "https":
        raise ToolError("only https:// addresses are allowed")
    try:
        r = client(ctx).get(url, params=params, headers={"User-Agent": UA, "Accept": "application/json", **(headers or {})})
    except httpx.HTTPError as e:
        raise ToolError(f"network error: {e.__class__.__name__}: {e}")
    if r.status_code == 429:
        raise ToolError(f"{host} says 'too many requests' — wait a minute and try again")
    if r.status_code in (401, 403):
        raise ToolError(f"{host} refused the request (HTTP {r.status_code}); " + _short(r))
    if r.status_code >= 400:
        raise ToolError(f"{host} returned HTTP {r.status_code}: {_short(r)}")
    if as_text:
        return r.text
    try:
        return r.json()
    except ValueError:
        return r.text


def _short(r) -> str:
    try:
        j = r.json()
        return json.dumps(j.get("error", j), ensure_ascii=False)[:200]
    except Exception:
        return r.text[:200].strip()


def post(ctx: Context, url: str, data=None, json_=None, headers=None):
    if urllib.parse.urlparse(url).scheme != "https":
        raise ToolError("only https:// addresses are allowed")
    try:
        r = client(ctx).post(url, data=data, json=json_, headers={"User-Agent": UA, **(headers or {})})
    except httpx.HTTPError as e:
        raise ToolError(f"network error: {e.__class__.__name__}: {e}")
    if r.status_code >= 400:
        raise ToolError(f"HTTP {r.status_code}: {_short(r)}")
    return r.json()


# ------------------------------------------------------------------ Wikimedia family
WIKIS = {"wikipedia": "en.wikipedia.org", "simple_wikipedia": "simple.wikipedia.org", "wiktionary": "en.wiktionary.org", "wikibooks": "en.wikibooks.org", "wikiquote": "en.wikiquote.org", "wikisource": "en.wikisource.org",
         "wikiversity": "en.wikiversity.org", "wikivoyage": "en.wikivoyage.org", "wikinews": "en.wikinews.org", "wikispecies": "species.wikimedia.org", "commons": "commons.wikimedia.org", "meta_wiki": "meta.wikimedia.org", "mediawiki": "www.mediawiki.org"}


def _api(ctx, host, **params):
    j = get(ctx, f"https://{host}/w/api.php", {"format": "json", "formatversion": 2, **params})
    if isinstance(j, dict) and "error" in j:
        raise ToolError(f"{host}: {j['error'].get('info', j['error'])}")
    return j


def _search(ctx, host, query, limit):
    j = _api(ctx, host, action="query", list="search", srsearch=query, srlimit=min(limit, 20))
    return [{"title": h["title"], "snippet": re.sub(r"<[^>]+>", "", h.get("snippet", "")), "url": f"https://{host}/wiki/{urllib.parse.quote(h['title'].replace(' ', '_'))}"} for h in j["query"]["search"]]


def _extract(ctx, host, title, sentences):
    j = _api(ctx, host, action="query", prop="extracts", explaintext=1, exintro=0 if sentences == 0 else 1, redirects=1, titles=title, **({"exsentences": sentences} if sentences else {}))
    pages = j["query"]["pages"]
    pg = pages[0]
    if pg.get("missing"):
        raise ToolError(f"no page called '{title}' on {host}. Try the search tool first.")
    return {"title": pg["title"], "text": (pg.get("extract") or "")[:5000], "url": f"https://{host}/wiki/{urllib.parse.quote(pg['title'].replace(' ', '_'))}", "license": "CC BY-SA 4.0 — cite the URL"}


for _k, _h in WIKIS.items():
    reg(f"{_k}_search", f"Search {_h} for pages matching a query.", N, "query:str, limit:int=5", (lambda h: lambda ctx, query, limit: _search(ctx, h, query, limit))(_h), risk="network", ctx=True, tags=("wikimedia", "research", "encyclopedia"))
    reg(f"{_k}_page", f"Read the plain-text content of a page on {_h} (first part; sentences=0 for the whole page up to 5000 characters).", N, "title:str, sentences:int=8", (lambda h: lambda ctx, title, sentences: _extract(ctx, h, title, sentences))(_h), risk="network", ctx=True, tags=("wikimedia", "research"))
reg("wikipedia_summary", "Short summary of a Wikipedia article (REST API), with thumbnail URL.", N, "title:str, language:str=en", lambda ctx, title, language: (lambda j: {"title": j.get("title"), "description": j.get("description"), "summary": j.get("extract"), "thumbnail": (j.get("thumbnail") or {}).get("source"), "url": j.get("content_urls", {}).get("desktop", {}).get("page")})(get(ctx, f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}")), risk="network", ctx=True, tags=("wikimedia",))
reg("wikipedia_random", "A random Wikipedia article summary.", N, "language:str=en", lambda ctx, language: (lambda j: {"title": j.get("title"), "summary": j.get("extract"), "url": j.get("content_urls", {}).get("desktop", {}).get("page")})(get(ctx, f"https://{language}.wikipedia.org/api/rest_v1/page/random/summary")), risk="network", ctx=True)
reg("wikipedia_on_this_day", "Historical events on a given month/day.", N, "month:int, day:int, kind:str=events", lambda ctx, month, day, kind: [{"year": e.get("year"), "text": e.get("text")} for e in get(ctx, f"https://api.wikimedia.org/feed/v1/wikipedia/en/onthisday/{kind}/{month:02d}/{day:02d}").get(kind, [])[:15]], risk="network", ctx=True)
reg("wikipedia_links", "Pages a Wikipedia article links to.", N, "title:str, limit:int=30", lambda ctx, title, limit: [l["title"] for l in _api(ctx, "en.wikipedia.org", action="query", prop="links", pllimit=min(limit, 100), titles=title, redirects=1)["query"]["pages"][0].get("links", [])], risk="network", ctx=True)
reg("wikipedia_categories", "Categories of a Wikipedia article.", N, "title:str", lambda ctx, title: [c["title"] for c in _api(ctx, "en.wikipedia.org", action="query", prop="categories", cllimit=50, clshow="!hidden", titles=title, redirects=1)["query"]["pages"][0].get("categories", [])], risk="network", ctx=True)
reg("wikipedia_sections", "Section headings of an article.", N, "title:str", lambda ctx, title: [f"{'  ' * (int(s['toclevel']) - 1)}{s['line']}" for s in _api(ctx, "en.wikipedia.org", action="parse", prop="sections", page=title, redirects=1)["parse"]["sections"]], risk="network", ctx=True)
reg("wikipedia_geosearch", "Wikipedia articles near a latitude/longitude.", N, "lat:float, lon:float, radius_m:int=5000", lambda ctx, lat, lon, radius_m: [{"title": g["title"], "distance_m": g["dist"]} for g in _api(ctx, "en.wikipedia.org", action="query", list="geosearch", gscoord=f"{lat}|{lon}", gsradius=min(radius_m, 10000), gslimit=15)["query"]["geosearch"]], risk="network", ctx=True)
reg("wikipedia_pageviews", "Daily page-view counts for an article over the last N days.", N, "title:str, days:int=30", lambda ctx, title, days: _views(ctx, title, days), risk="network", ctx=True)
reg("wiktionary_define", "Dictionary entry (definitions) for an English word from Wiktionary.", N, "word:str", lambda ctx, word: _define(ctx, word), risk="network", ctx=True, tags=("dictionary",))
reg("commons_image_search", "Search Wikimedia Commons for freely licensed images.", N, "query:str, limit:int=6", lambda ctx, query, limit: _commons(ctx, query, limit), risk="network", ctx=True, tags=("image", "media"))
reg("commons_download", "Download an image from Wikimedia Commons into the workspace (use a URL from commons_image_search).", N, "url:str, path:str=images/commons.jpg", lambda ctx, url, path: _download(ctx, url, path, ("upload.wikimedia.org",)), risk="network", ctx=True, produces="image")
reg("wikidata_search", "Find Wikidata entities (people, places, things) by name.", N, "query:str, limit:int=5", lambda ctx, query, limit: [{"id": e["id"], "label": e.get("label"), "description": e.get("description")} for e in _api(ctx, "www.wikidata.org", action="wbsearchentities", search=query, language="en", limit=min(limit, 20))["search"]], risk="network", ctx=True)
reg("wikidata_entity", "Labels, description and key facts of a Wikidata entity (like Q42).", N, "id:str", lambda ctx, id: _wikidata(ctx, id), risk="network", ctx=True)
reg("wikidata_sparql", "Run a SPARQL query on Wikidata (read-only).", N, "query:str", lambda ctx, query: get(ctx, "https://query.wikidata.org/sparql", {"query": query, "format": "json"}, {"Accept": "application/sparql-results+json"})["results"]["bindings"][:50], risk="network", ctx=True)
reg("wikisource_text", "Read a text from Wikisource.", N, "title:str", lambda ctx, title: _extract(ctx, "en.wikisource.org", title, 0), risk="network", ctx=True)
reg("wikiquote_quotes", "Quotes from a Wikiquote page.", N, "title:str", lambda ctx, title: _extract(ctx, "en.wikiquote.org", title, 0), risk="network", ctx=True)


def _views(ctx, title, days):
    from datetime import date, timedelta
    end, start = date.today() - timedelta(days=1), date.today() - timedelta(days=days)
    j = get(ctx, f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{urllib.parse.quote(title.replace(' ', '_'), safe='')}/daily/{start:%Y%m%d}/{end:%Y%m%d}")
    return {"total": sum(i["views"] for i in j["items"]), "daily": [(i["timestamp"][:8], i["views"]) for i in j["items"]]}


def _define(ctx, word):
    j = get(ctx, f"https://en.wiktionary.org/api/rest_v1/page/definition/{urllib.parse.quote(word)}")
    out = []
    for lang, entries in (j.items() if isinstance(j, dict) else []):
        if lang != "en":
            continue
        for e in entries[:6]:
            out.append({"part_of_speech": e.get("partOfSpeech"), "definitions": [re.sub(r"<[^>]+>", "", d["definition"]) for d in e["definitions"][:4]]})
    if not out:
        raise ToolError(f"no English definition found for '{word}'")
    return out


def _commons(ctx, query, limit):
    j = _api(ctx, "commons.wikimedia.org", action="query", generator="search", gsrsearch=f"filetype:bitmap {query}", gsrnamespace=6, gsrlimit=min(limit, 15), prop="imageinfo", iiprop="url|extmetadata|size", iiurlwidth=800)
    out = []
    for p in j.get("query", {}).get("pages", []):
        ii = p["imageinfo"][0]
        md = ii.get("extmetadata", {})
        out.append({"title": p["title"], "thumb_url": ii.get("thumburl"), "url": ii["url"], "size": [ii.get("width"), ii.get("height")], "license": md.get("LicenseShortName", {}).get("value"), "author": re.sub(r"<[^>]+>", "", md.get("Artist", {}).get("value", ""))[:80]})
    return out


def _download(ctx, url, path, hosts):
    u = urllib.parse.urlparse(url)
    if u.scheme != "https" or (hosts and not any(u.netloc == h or u.netloc.endswith("." + h) for h in hosts)):
        raise ToolError(f"downloads are only allowed from: {', '.join(hosts)}")
    try:
        r = client(ctx).get(url, headers={"User-Agent": UA, "Accept": "*/*"})
    except httpx.HTTPError as e:
        raise ToolError(f"network error: {e}")
    if r.status_code != 200:
        raise ToolError(f"download failed: HTTP {r.status_code}")
    if len(r.content) > 40_000_000:
        raise ToolError("file is larger than 40 MB")
    p = ctx.path(path)
    p.write_bytes(r.content)
    return {"saved": ctx.produced(p, "image" if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg") else "file", "downloaded"), "bytes": len(r.content)}


def _wikidata(ctx, qid):
    j = get(ctx, f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json")
    e = j["entities"][qid]
    claims = {}
    for pid, cl in list(e.get("claims", {}).items())[:25]:
        v = cl[0]["mainsnak"].get("datavalue", {}).get("value")
        claims[pid] = v if isinstance(v, (str, int, float)) else (v.get("id") or v.get("amount") or v.get("time") or v.get("text")) if isinstance(v, dict) else None
    return {"id": qid, "label": e["labels"].get("en", {}).get("value"), "description": e["descriptions"].get("en", {}).get("value"), "aliases": [a["value"] for a in e.get("aliases", {}).get("en", [])][:6], "claims": claims}


# ------------------------------------------------------------------ YouTube (official Data API v3)
YT = "youtube"
YK = ("env:YOUTUBE_API_KEY",)
_yt = lambda ctx, path, **p: get(ctx, f"https://www.googleapis.com/youtube/v3/{path}", {**p, "key": ctx.secret("YOUTUBE_API_KEY")})  # noqa: E731
_vid = lambda v: (re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", v) or [None, v])[1]  # noqa: E731
reg("youtube_search_videos", "Search YouTube videos (needs a YouTube Data API key).", YT, "query:str, limit:int=5, order:str=relevance", lambda ctx, query, limit, order: [{"id": i["id"]["videoId"], "title": i["snippet"]["title"], "channel": i["snippet"]["channelTitle"], "published": i["snippet"]["publishedAt"][:10], "url": f"https://youtu.be/{i['id']['videoId']}"} for i in _yt(ctx, "search", part="snippet", q=query, type="video", maxResults=min(limit, 25), order=order)["items"]], risk="network", ctx=True, needs=YK, tags=("yt", "video"))
reg("youtube_video_info", "Title, channel, views, likes, duration of a video (id or URL).", YT, "video:str", lambda ctx, video: (lambda i: {"title": i["snippet"]["title"], "channel": i["snippet"]["channelTitle"], "published": i["snippet"]["publishedAt"][:10], "description": i["snippet"]["description"][:600], "views": i["statistics"].get("viewCount"), "likes": i["statistics"].get("likeCount"), "comments": i["statistics"].get("commentCount"), "duration": i["contentDetails"]["duration"], "tags": i["snippet"].get("tags", [])[:12]})(_yt(ctx, "videos", part="snippet,statistics,contentDetails", id=_vid(video))["items"][0]), risk="network", ctx=True, needs=YK)
reg("youtube_channel_info", "Subscribers, views and video count of a channel (id like UC…, or @handle).", YT, "channel:str", lambda ctx, channel: (lambda i: {"title": i["snippet"]["title"], "description": i["snippet"]["description"][:400], "subscribers": i["statistics"].get("subscriberCount"), "views": i["statistics"].get("viewCount"), "videos": i["statistics"].get("videoCount"), "uploads_playlist": i["contentDetails"]["relatedPlaylists"]["uploads"]})(_yt(ctx, "channels", part="snippet,statistics,contentDetails", **({"forHandle": channel} if channel.startswith("@") else {"id": channel}))["items"][0]), risk="network", ctx=True, needs=YK)
reg("youtube_channel_videos", "Latest videos of a channel (give its uploads playlist id from youtube_channel_info).", YT, "playlist:str, limit:int=10", lambda ctx, playlist, limit: [{"id": i["contentDetails"]["videoId"], "title": i["snippet"]["title"], "published": i["snippet"]["publishedAt"][:10]} for i in _yt(ctx, "playlistItems", part="snippet,contentDetails", playlistId=playlist, maxResults=min(limit, 50))["items"]], risk="network", ctx=True, needs=YK)
reg("youtube_video_comments", "Top comments on a video.", YT, "video:str, limit:int=10", lambda ctx, video, limit: [{"author": c["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"], "text": c["snippet"]["topLevelComment"]["snippet"]["textDisplay"][:300], "likes": c["snippet"]["topLevelComment"]["snippet"]["likeCount"]} for c in _yt(ctx, "commentThreads", part="snippet", videoId=_vid(video), maxResults=min(limit, 50), order="relevance")["items"]], risk="network", ctx=True, needs=YK)
reg("youtube_search_playlists", "Search YouTube playlists.", YT, "query:str, limit:int=5", lambda ctx, query, limit: [{"id": i["id"]["playlistId"], "title": i["snippet"]["title"], "channel": i["snippet"]["channelTitle"]} for i in _yt(ctx, "search", part="snippet", q=query, type="playlist", maxResults=min(limit, 25))["items"]], risk="network", ctx=True, needs=YK)
reg("youtube_search_channels", "Search YouTube channels.", YT, "query:str, limit:int=5", lambda ctx, query, limit: [{"id": i["id"]["channelId"], "title": i["snippet"]["title"], "description": i["snippet"]["description"][:120]} for i in _yt(ctx, "search", part="snippet", q=query, type="channel", maxResults=min(limit, 25))["items"]], risk="network", ctx=True, needs=YK)
reg("youtube_trending", "Most popular videos in a region (ISO country code).", YT, "region:str=US, limit:int=10", lambda ctx, region, limit: [{"id": i["id"], "title": i["snippet"]["title"], "channel": i["snippet"]["channelTitle"], "views": i["statistics"].get("viewCount")} for i in _yt(ctx, "videos", part="snippet,statistics", chart="mostPopular", regionCode=region, maxResults=min(limit, 50))["items"]], risk="network", ctx=True, needs=YK)
reg("youtube_categories", "YouTube video categories in a region.", YT, "region:str=US", lambda ctx, region: {i["id"]: i["snippet"]["title"] for i in _yt(ctx, "videoCategories", part="snippet", regionCode=region)["items"]}, risk="network", ctx=True, needs=YK)
reg("youtube_transcript", "Public captions (transcript) of a video via the timedtext endpoint (works only when captions are public).", YT, "video:str, language:str=en", lambda ctx, video, language: _transcript(ctx, _vid(video), language), risk="network", ctx=True, tags=("subtitles",))
reg("youtube_thumbnail_url", "Thumbnail image URLs of a video (no API key needed).", YT, "video:str", lambda video: {q: f"https://i.ytimg.com/vi/{_vid(video)}/{q}.jpg" for q in ("maxresdefault", "hqdefault", "mqdefault")})
reg("youtube_thumbnail_download", "Download a video's thumbnail into the workspace.", YT, "video:str, path:str=images/thumbnail.jpg", lambda ctx, video, path: _download(ctx, f"https://i.ytimg.com/vi/{_vid(video)}/hqdefault.jpg", path, ("i.ytimg.com",)), risk="network", ctx=True, produces="image")
reg("youtube_embed_code", "HTML embed code for a video.", YT, "video:str, width:int=560, height:int=315", lambda video, width, height: f'<iframe width="{width}" height="{height}" src="https://www.youtube.com/embed/{_vid(video)}" title="YouTube video" frameborder="0" allowfullscreen></iframe>')
reg("youtube_title_ideas", "Generate title ideas for a video topic from proven templates (no AI, no network).", YT, "topic:str", lambda topic: [t.format(t=topic, T=topic.title()) for t in ("How {t} Actually Works (Explained Simply)", "I Tried {t} for 30 Days — Here's What Happened", "{T}: The Complete Beginner's Guide", "Stop Doing {t} Wrong", "{T} in 10 Minutes", "5 {t} Mistakes Everyone Makes", "The Truth About {t}", "{T} — What Nobody Tells You", "Learn {t} From Scratch (Full Course)", "{T} vs the Alternatives: Which Is Better?")])
reg("youtube_description_template", "A YouTube description with chapters and hashtags.", YT, "title:str, chapters:list?, hashtags:str?", lambda title, chapters, hashtags: f"{title}\n\nIn this video:\n" + "\n".join(f"{c[0]} {c[1]}" if isinstance(c, (list, tuple)) else str(c) for c in (chapters or [])) + "\n\nSubscribe for more!\n\n" + (hashtags or ""))
reg("youtube_chapters_from_seconds", "Format chapter timestamps (list of [seconds, title]) for a description.", YT, "chapters:list", lambda chapters: "\n".join(f"{int(s) // 60}:{int(s) % 60:02d} {t}" for s, t in chapters))
reg("youtube_download_instructions", "How to save a video you are allowed to download (yt-dlp), as a ready command.", YT, "url:str, audio_only:bool=false", lambda url, audio_only: ("yt-dlp -x --audio-format mp3 " if audio_only else "yt-dlp -f 'bv*[height<=1080]+ba/b' ") + f"'{url}'  # only for content you own or that is licensed for download")
reg("yt_dlp_download", "Download a video/audio you are permitted to download, using yt-dlp (must be installed).", YT, "url:str, audio_only:bool=false, folder:str=downloads", lambda ctx, url, audio_only, folder: _ytdlp(ctx, url, audio_only, folder), risk="exec", ctx=True, needs=("bin:yt-dlp",), timeout=600.0, produces="video")
reg("yt_dlp_info", "Metadata of a video without downloading (yt-dlp).", YT, "url:str", lambda ctx, url: _ytdlp_info(ctx, url), risk="exec", ctx=True, needs=("bin:yt-dlp",))
reg("youtube_upload_video", "Upload a video to YouTube (needs OAuth: YOUTUBE_ACCESS_TOKEN). Posts publicly only if privacy=public.", YT, "path:str, title:str, description:str?, privacy:str=private", lambda ctx, path, title, description, privacy: _yt_upload(ctx, path, title, description, privacy), risk="post", ctx=True, needs=("env:YOUTUBE_ACCESS_TOKEN",), timeout=900.0)


def _transcript(ctx, vid, lang):
    xml = get(ctx, "https://www.youtube.com/api/timedtext", {"v": vid, "lang": lang, "fmt": "srv3"}, as_text=True)
    if not xml.strip():
        raise ToolError("no public captions for this video/language")
    import html as h
    return h.unescape(" ".join(re.findall(r"<s[^>]*>([^<]*)</s>|<p[^>]*>([^<]+)</p>", xml) and [a or b for a, b in re.findall(r"<s[^>]*>([^<]*)</s>|<p[^>]*>([^<]+)</p>", xml)]))[:6000]


def _ytdlp(ctx, url, audio, folder):
    import subprocess
    if not url.startswith("https://"):
        raise ToolError("give a full https:// link")
    d = ctx.path(folder + "/.keep").parent
    args = ["yt-dlp", "--no-playlist", "-o", str(d / "%(title).80s.%(ext)s"), "--restrict-filenames"] + (["-x", "--audio-format", "mp3"] if audio else ["-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4"]) + [url]
    r = subprocess.run(args, capture_output=True, text=True, timeout=600, cwd=ctx.workspace, stdin=subprocess.DEVNULL)
    if r.returncode:
        raise ToolError("yt-dlp failed: " + r.stderr.strip().splitlines()[-1][:300])
    for p in sorted(d.iterdir(), key=lambda p: -p.stat().st_mtime)[:1]:
        return {"saved": ctx.produced(p, "video" if p.suffix == ".mp4" else "audio", "downloaded with yt-dlp")}
    raise ToolError("nothing was downloaded")


def _ytdlp_info(ctx, url):
    import subprocess
    r = subprocess.run(["yt-dlp", "--no-playlist", "-J", url], capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    if r.returncode:
        raise ToolError("yt-dlp failed: " + r.stderr.strip().splitlines()[-1][:300])
    j = json.loads(r.stdout)
    return {k: j.get(k) for k in ("title", "uploader", "duration", "view_count", "upload_date", "description")}


def _yt_upload(ctx, path, title, desc, privacy):
    p = ctx.path(path, True)
    tok = ctx.secret("YOUTUBE_ACCESS_TOKEN")
    meta = {"snippet": {"title": title, "description": desc or ""}, "status": {"privacyStatus": privacy}}
    c = client(ctx)
    r = c.post("https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status", json=meta, headers={"Authorization": f"Bearer {tok}", "X-Upload-Content-Type": "video/*"})
    if r.status_code >= 400:
        raise ToolError(f"YouTube refused the upload: {_short(r)}")
    up = c.put(r.headers["Location"], content=p.read_bytes(), headers={"Content-Type": "video/*"}, timeout=800)
    if up.status_code >= 400:
        raise ToolError(f"upload failed: {_short(up)}")
    return {"video_id": up.json().get("id"), "privacy": privacy}


# ------------------------------------------------------------------ Instagram (official Graph API for Business/Creator accounts)
IG = "instagram"
GRAPH = "https://graph.facebook.com/v21.0"
IGK = ("env:INSTAGRAM_ACCESS_TOKEN", "env:INSTAGRAM_USER_ID")
_ig = lambda ctx, path, **p: get(ctx, f"{GRAPH}/{path}", {**p, "access_token": ctx.secret("INSTAGRAM_ACCESS_TOKEN")})  # noqa: E731
reg("instagram_account_info", "Your Instagram professional account: username, followers, media count.", IG, "", lambda ctx: _ig(ctx, ctx.secret("INSTAGRAM_USER_ID"), fields="username,name,biography,followers_count,follows_count,media_count"), risk="network", ctx=True, needs=IGK, tags=("insta",))
reg("instagram_recent_posts", "Your latest posts with captions, likes and comment counts.", IG, "limit:int=10", lambda ctx, limit: _ig(ctx, f"{ctx.secret('INSTAGRAM_USER_ID')}/media", fields="id,caption,media_type,permalink,timestamp,like_count,comments_count", limit=min(limit, 50))["data"], risk="network", ctx=True, needs=IGK)
reg("instagram_post_insights", "Reach, impressions and saves of one of your posts.", IG, "media_id:str", lambda ctx, media_id: {m["name"]: m["values"][0]["value"] for m in _ig(ctx, f"{media_id}/insights", metric="reach,saved,likes,comments,shares")["data"]}, risk="network", ctx=True, needs=IGK)
reg("instagram_account_insights", "Account-level reach and profile views for the last N days (max 30).", IG, "days:int=7", lambda ctx, days: _ig(ctx, f"{ctx.secret('INSTAGRAM_USER_ID')}/insights", metric="reach,profile_views", period="day", since=int(time.time()) - 86400 * min(days, 30), until=int(time.time()))["data"], risk="network", ctx=True, needs=IGK)
reg("instagram_hashtag_search", "Look up a hashtag id, then its top media.", IG, "hashtag:str, limit:int=9", lambda ctx, hashtag, limit: (lambda hid: _ig(ctx, f"{hid}/top_media", user_id=ctx.secret("INSTAGRAM_USER_ID"), fields="id,caption,like_count,comments_count,permalink", limit=min(limit, 30))["data"])(_ig(ctx, "ig_hashtag_search", user_id=ctx.secret("INSTAGRAM_USER_ID"), q=hashtag.lstrip("#"))["data"][0]["id"]), risk="network", ctx=True, needs=IGK)
reg("instagram_post_image", "Publish a photo (public https image URL) with a caption to your Instagram account.", IG, "image_url:str, caption:str", lambda ctx, image_url, caption: _ig_publish(ctx, {"image_url": image_url, "caption": caption}), risk="post", ctx=True, needs=IGK, tags=("insta", "publish"), timeout=120.0)
reg("instagram_post_reel", "Publish a Reel (public https video URL) with a caption.", IG, "video_url:str, caption:str", lambda ctx, video_url, caption: _ig_publish(ctx, {"media_type": "REELS", "video_url": video_url, "caption": caption}, True), risk="post", ctx=True, needs=IGK, timeout=300.0)
reg("instagram_post_story", "Publish a Story (image or video https URL).", IG, "media_url:str, is_video:bool=false", lambda ctx, media_url, is_video: _ig_publish(ctx, {"media_type": "STORIES", ("video_url" if is_video else "image_url"): media_url}, is_video), risk="post", ctx=True, needs=IGK, timeout=300.0)
reg("instagram_reply_to_comment", "Reply to a comment on your post.", IG, "comment_id:str, message:str", lambda ctx, comment_id, message: post(ctx, f"{GRAPH}/{comment_id}/replies", data={"message": message, "access_token": ctx.secret("INSTAGRAM_ACCESS_TOKEN")}), risk="post", ctx=True, needs=IGK)
reg("instagram_caption_template", "Draft a caption with hook, body, call-to-action and hashtags (rule-based, no AI).", IG, "topic:str, tone:str=friendly, hashtags:int=8", lambda topic, tone, hashtags: _caption(topic, tone, hashtags), tags=("insta",))
reg("instagram_hashtag_ideas", "Hashtag ideas from a topic.", IG, "topic:str, count:int=15", lambda topic, count: _hashtags(topic, count))
reg("instagram_best_times", "General best posting windows (study averages, not your audience data).", IG, "", lambda: {"weekday_windows_local": ["Tue–Fri 10:00–15:00", "Mon–Fri 18:00–21:00"], "note": "Check your own Insights → Followers → Most active times."})
reg("instagram_image_specs", "Image and video size rules for posts, stories, reels.", IG, "", lambda: {"post_square": "1080×1080", "post_portrait": "1080×1350 (4:5)", "story_reel": "1080×1920 (9:16)", "max_reel_length_s": 900, "caption_limit": 2200, "hashtag_limit": 30, "carousel_max": 20})


def _ig_publish(ctx, params, video=False):
    uid, tok = ctx.secret("INSTAGRAM_USER_ID"), ctx.secret("INSTAGRAM_ACCESS_TOKEN")
    c = post(ctx, f"{GRAPH}/{uid}/media", data={**params, "access_token": tok})
    cid = c["id"]
    for _ in range(40 if video else 3):
        st = get(ctx, f"{GRAPH}/{cid}", {"fields": "status_code", "access_token": tok}).get("status_code")
        if st in ("FINISHED", None):
            break
        if st == "ERROR":
            raise ToolError("Instagram could not process the media")
        time.sleep(3)
    return {"published_id": post(ctx, f"{GRAPH}/{uid}/media_publish", data={"creation_id": cid, "access_token": tok}).get("id")}


def _hashtags(topic, n):
    words = [w for w in re.findall(r"[a-z0-9]+", topic.lower()) if len(w) > 2]
    base = ["".join(words)] + words + [w + s for w in words[:3] for s in ("tips", "life", "daily", "community", "love", "goals")] + ["instagood", "explore", "reels", "viral", "trending", "learn", "howto", "creator"]
    return list(dict.fromkeys("#" + b for b in base if b))[:n]


def _caption(topic, tone, n):
    hooks = {"friendly": f"Quick question: have you tried {topic} yet? 👀", "professional": f"Three things to know about {topic}.", "funny": f"Me explaining {topic} to everyone who didn't ask 😅", "inspiring": f"Start {topic} today — future you will say thanks. ✨"}
    return hooks.get(tone, hooks["friendly"]) + f"\n\nHere's what I've learned about {topic} so far — save this for later! 📌\n\nWhat would you add? Tell me below 👇\n\n" + " ".join(_hashtags(topic, n))


# ------------------------------------------------------------------ GitHub (public REST API; token optional)
GH = "github"
_gh = lambda ctx, path, **p: get(ctx, f"https://api.github.com/{path}", p, {"Accept": "application/vnd.github+json", "User-Agent": UA, **({"Authorization": "Bearer " + (ctx.secrets.get("GITHUB_TOKEN") or "")} if ctx.secrets.get("GITHUB_TOKEN") else {})})  # noqa: E731
reg("github_search_repos", "Search public GitHub repositories.", GH, "query:str, limit:int=5, sort:str=stars", lambda ctx, query, limit, sort: [{"name": r["full_name"], "stars": r["stargazers_count"], "language": r["language"], "description": r["description"], "url": r["html_url"]} for r in _gh(ctx, "search/repositories", q=query, sort=sort, per_page=min(limit, 20))["items"]], risk="network", ctx=True)
reg("github_repo_info", "Details of a repository (owner/name).", GH, "repo:str", lambda ctx, repo: (lambda r: {k: r.get(k) for k in ("full_name", "description", "stargazers_count", "forks_count", "open_issues_count", "language", "license", "pushed_at", "html_url", "topics")})(_gh(ctx, f"repos/{repo}")), risk="network", ctx=True)
reg("github_readme", "The README of a repository (text).", GH, "repo:str", lambda ctx, repo: get(ctx, f"https://api.github.com/repos/{repo}/readme", None, {"Accept": "application/vnd.github.raw+json", "User-Agent": UA}, as_text=True)[:6000], risk="network", ctx=True)
reg("github_list_files", "List files in a repository folder.", GH, "repo:str, path:str=", lambda ctx, repo, path: [f"{i['type']}: {i['path']}" for i in _gh(ctx, f"repos/{repo}/contents/{path}")][:100], risk="network", ctx=True)
reg("github_read_file", "Read a text file from a public repository.", GH, "repo:str, path:str", lambda ctx, repo, path: get(ctx, f"https://api.github.com/repos/{repo}/contents/{path}", None, {"Accept": "application/vnd.github.raw+json", "User-Agent": UA}, as_text=True)[:8000], risk="network", ctx=True)
reg("github_search_code", "Search code on GitHub (needs a GITHUB_TOKEN in Connections).", GH, "query:str, limit:int=5", lambda ctx, query, limit: [{"repo": i["repository"]["full_name"], "path": i["path"], "url": i["html_url"]} for i in _gh(ctx, "search/code", q=query, per_page=min(limit, 20))["items"]], risk="network", ctx=True, needs=("env:GITHUB_TOKEN",))
reg("github_issues", "Open issues of a repository.", GH, "repo:str, limit:int=10", lambda ctx, repo, limit: [{"number": i["number"], "title": i["title"], "by": i["user"]["login"], "url": i["html_url"]} for i in _gh(ctx, f"repos/{repo}/issues", state="open", per_page=min(limit, 30)) if "pull_request" not in i], risk="network", ctx=True)
reg("github_latest_release", "Latest release of a repository.", GH, "repo:str", lambda ctx, repo: (lambda r: {"tag": r["tag_name"], "name": r["name"], "published": r["published_at"], "url": r["html_url"], "notes": (r.get("body") or "")[:800]})(_gh(ctx, f"repos/{repo}/releases/latest")), risk="network", ctx=True)
reg("github_user", "Public profile of a GitHub user.", GH, "username:str", lambda ctx, username: (lambda u: {k: u.get(k) for k in ("login", "name", "bio", "public_repos", "followers", "html_url")})(_gh(ctx, f"users/{username}")), risk="network", ctx=True)
reg("github_trending_like", "Recently created popular repos for a topic (stars sorted).", GH, "topic:str, days:int=30, limit:int=8", lambda ctx, topic, days, limit: [{"name": r["full_name"], "stars": r["stargazers_count"], "url": r["html_url"]} for r in _gh(ctx, "search/repositories", q=f"topic:{topic} created:>{time.strftime('%Y-%m-%d', time.gmtime(time.time() - 86400 * days))}", sort="stars", per_page=min(limit, 20))["items"]], risk="network", ctx=True)
reg("github_download_file", "Download a public repository file into the workspace.", GH, "repo:str, path:str, dest:str?", lambda ctx, repo, path, dest: _download(ctx, f"https://raw.githubusercontent.com/{repo}/HEAD/{path}", dest or f"downloads/{path.rsplit('/', 1)[-1]}", ("raw.githubusercontent.com",)), risk="network", ctx=True)

# ------------------------------------------------------------------ Hugging Face (public hub API)
HF = "huggingface"
reg("hf_search_models", "Search models on the Hugging Face Hub.", HF, "query:str, limit:int=8, task:str?", lambda ctx, query, limit, task: [{"id": m["id"], "downloads": m.get("downloads"), "likes": m.get("likes"), "task": m.get("pipeline_tag")} for m in get(ctx, "https://huggingface.co/api/models", {"search": query, "limit": min(limit, 30), "sort": "downloads", "direction": -1, **({"pipeline_tag": task} if task else {})})], risk="network", ctx=True, tags=("models", "ai"))
reg("hf_search_datasets", "Search datasets on the Hugging Face Hub.", HF, "query:str, limit:int=8", lambda ctx, query, limit: [{"id": m["id"], "downloads": m.get("downloads"), "likes": m.get("likes")} for m in get(ctx, "https://huggingface.co/api/datasets", {"search": query, "limit": min(limit, 30), "sort": "downloads", "direction": -1})], risk="network", ctx=True)
reg("hf_search_spaces", "Search Spaces (demo apps) on the Hub.", HF, "query:str, limit:int=8", lambda ctx, query, limit: [{"id": m["id"], "likes": m.get("likes"), "sdk": m.get("sdk")} for m in get(ctx, "https://huggingface.co/api/spaces", {"search": query, "limit": min(limit, 30), "sort": "likes", "direction": -1})], risk="network", ctx=True)
reg("hf_model_card", "Model details: size, tags, licence and the start of its README.", HF, "model:str", lambda ctx, model: (lambda j, r: {"id": j["id"], "task": j.get("pipeline_tag"), "library": j.get("library_name"), "tags": j.get("tags", [])[:15], "downloads": j.get("downloads"), "safetensors_params": (j.get("safetensors") or {}).get("total"), "readme": r[:1500]})(get(ctx, f"https://huggingface.co/api/models/{model}"), get(ctx, f"https://huggingface.co/{model}/raw/main/README.md", as_text=True)), risk="network", ctx=True)
reg("hf_dataset_card", "Dataset details and the start of its README.", HF, "dataset:str", lambda ctx, dataset: (lambda j, r: {"id": j["id"], "downloads": j.get("downloads"), "tags": j.get("tags", [])[:15], "readme": r[:1500]})(get(ctx, f"https://huggingface.co/api/datasets/{dataset}"), get(ctx, f"https://huggingface.co/datasets/{dataset}/raw/main/README.md", as_text=True)), risk="network", ctx=True)
reg("hf_dataset_rows", "Preview rows of a Hub dataset (datasets-server).", HF, "dataset:str, config:str=default, split:str=train, rows:int=5", lambda ctx, dataset, config, split, rows: [r["row"] for r in get(ctx, "https://datasets-server.huggingface.co/rows", {"dataset": dataset, "config": config, "split": split, "offset": 0, "length": min(rows, 20)})["rows"]], risk="network", ctx=True)
reg("hf_inference", "Run a hosted model through the HF Inference API (needs HF_TOKEN). Example: model='facebook/bart-large-cnn', inputs='long text'.", HF, "model:str, inputs:str", lambda ctx, model, inputs: post(ctx, f"https://api-inference.huggingface.co/models/{model}", json_={"inputs": inputs}, headers={"Authorization": "Bearer " + ctx.secret("HF_TOKEN")}), risk="network", ctx=True, needs=("env:HF_TOKEN",), timeout=120.0)
reg("hf_model_size_check", "Is a Hub model within the classroom's 27B-parameter limit? (reads the safetensors metadata)", HF, "model:str", lambda ctx, model: (lambda t: {"parameters": t, "within_27b": (t or 0) <= 27.5e9} if t else {"parameters": None, "note": "size not published — check the model card"})((get(ctx, f"https://huggingface.co/api/models/{model}").get("safetensors") or {}).get("total")), risk="network", ctx=True)

# ------------------------------------------------------------------ free public APIs (no key) — one table, one tool each
APIS = {
    "country_info": ("Facts about a country: capital, population, languages, currency, flag.", "https://restcountries.com/v3.1/name/{name}", "name:str", lambda j: [{"name": c["name"]["common"], "capital": c.get("capital"), "region": c.get("region"), "population": c.get("population"), "area": c.get("area"), "languages": list((c.get("languages") or {}).values()), "currencies": list((c.get("currencies") or {}).keys()), "flag": c.get("flag")} for c in j[:3]]),
    "weather_now": ("Current weather at a latitude/longitude (Open-Meteo).", "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code", "lat:float, lon:float", lambda j: j["current"]),
    "weather_forecast": ("7-day forecast at a latitude/longitude (Open-Meteo).", "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto", "lat:float, lon:float", lambda j: j["daily"]),
    "geocode_place": ("Latitude/longitude of a place name (Open-Meteo geocoding).", "https://geocoding-api.open-meteo.com/v1/search?name={name}&count=3", "name:str", lambda j: [{"name": r["name"], "country": r.get("country"), "lat": r["latitude"], "lon": r["longitude"], "population": r.get("population")} for r in j.get("results", [])]),
    "air_quality": ("Air quality (PM2.5, PM10, ozone) at a location.", "https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm10,pm2_5,ozone,us_aqi", "lat:float, lon:float", lambda j: j["current"]),
    "earthquakes_recent": ("Significant earthquakes in the past week (USGS).", "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.geojson", "", lambda j: [{"place": f["properties"]["place"], "magnitude": f["properties"]["mag"], "url": f["properties"]["url"]} for f in j["features"][:10]]),
    "iss_position": ("Where the International Space Station is right now.", "https://api.wheretheiss.at/v1/satellites/25544", "", lambda j: {k: j[k] for k in ("latitude", "longitude", "altitude", "velocity")}),
    "nasa_apod": ("NASA Astronomy Picture of the Day (demo key).", "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY", "", lambda j: {k: j.get(k) for k in ("title", "date", "explanation", "url", "media_type")}),
    "exchange_rates": ("Currency rates for a base currency (Frankfurter, ECB data).", "https://api.frankfurter.app/latest?from={base}", "base:str=USD", lambda j: j),
    "convert_currency": ("Convert an amount between currencies (ECB rates).", "https://api.frankfurter.app/latest?amount={amount}&from={frm}&to={to}", "amount:float=1, frm:str=USD, to:str=EUR", lambda j: j),
    "crypto_price": ("Price of a coin in USD/EUR (CoinGecko), e.g. bitcoin.", "https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd,eur&include_24hr_change=true", "coin:str=bitcoin", lambda j: j),
    "random_fact": ("A random useless fact.", "https://uselessfacts.jsph.pl/api/v2/facts/random?language=en", "", lambda j: j.get("text")),
    "random_joke": ("A clean random joke.", "https://v2.jokeapi.dev/joke/Any?safe-mode&type=single", "", lambda j: j.get("joke")),
    "random_advice": ("A piece of random advice.", "https://api.adviceslip.com/advice", "", lambda j: (json.loads(j) if isinstance(j, str) else j)["slip"]["advice"]),
    "dictionary_define": ("English dictionary definitions, phonetics and examples.", "https://api.dictionaryapi.dev/api/v2/entries/en/{word}", "word:str", lambda j: [{"word": e["word"], "phonetic": e.get("phonetic"), "meanings": [{"part": m["partOfSpeech"], "definitions": [d["definition"] for d in m["definitions"][:3]]} for m in e["meanings"][:4]]} for e in j[:1]]),
    "datamuse_rhymes": ("Words that rhyme with a word.", "https://api.datamuse.com/words?rel_rhy={word}&max=20", "word:str", lambda j: [w["word"] for w in j]),
    "datamuse_synonyms": ("Synonyms of a word.", "https://api.datamuse.com/words?rel_syn={word}&max=20", "word:str", lambda j: [w["word"] for w in j]),
    "datamuse_related": ("Words with a similar meaning to a phrase.", "https://api.datamuse.com/words?ml={word}&max=20", "word:str", lambda j: [w["word"] for w in j]),
    "pokemon_info": ("Stats of a Pokémon.", "https://pokeapi.co/api/v2/pokemon/{name}", "name:str", lambda j: {"name": j["name"], "height": j["height"], "weight": j["weight"], "types": [t["type"]["name"] for t in j["types"]], "stats": {s["stat"]["name"]: s["base_stat"] for s in j["stats"]}}),
    "public_holidays": ("Public holidays of a country (ISO code) in a year.", "https://date.nager.at/api/v3/PublicHolidays/{year}/{country}", "year:int=2026, country:str=US", lambda j: [(h["date"], h["name"]) for h in j]),
    "book_search": ("Search books (Open Library).", "https://openlibrary.org/search.json?q={query}&limit=6", "query:str", lambda j: [{"title": d["title"], "author": d.get("author_name", ["?"])[0], "year": d.get("first_publish_year"), "key": d["key"]} for d in j["docs"]]),
    "university_search": ("Universities by name (Hipolabs).", "https://universities.hipolabs.com/search?name={name}", "name:str", lambda j: [{"name": u["name"], "country": u["country"], "web": (u.get("web_pages") or [None])[0]} for u in j[:10]]),
    "ip_info": ("Location and network of an IP address (ipapi.co).", "https://ipapi.co/{ip}/json/", "ip:str", lambda j: {k: j.get(k) for k in ("ip", "city", "region", "country_name", "org", "timezone")}),
    "arxiv_search": ("Search arXiv papers.", "https://export.arxiv.org/api/query?search_query=all:{query}&max_results=5", "query:str", None),
    "pubmed_search": ("Search PubMed article ids and titles.", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=8&term={query}", "query:str", lambda j: j["esearchresult"]["idlist"]),
    "crossref_search": ("Search scholarly works (Crossref).", "https://api.crossref.org/works?query={query}&rows=5", "query:str", lambda j: [{"title": (w.get("title") or ["?"])[0], "doi": w["DOI"], "year": (w.get("issued", {}).get("date-parts") or [[None]])[0][0]} for w in j["message"]["items"]]),
    "openalex_search": ("Search research works (OpenAlex).", "https://api.openalex.org/works?search={query}&per-page=5", "query:str", lambda j: [{"title": w["title"], "year": w["publication_year"], "cited_by": w["cited_by_count"], "doi": w.get("doi")} for w in j["results"]]),
    "periodic_element": ("Element data from the PubChem periodic table.", "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/MolecularFormula,MolecularWeight,IUPACName/JSON", "name:str", lambda j: j["PropertyTable"]["Properties"][0]),
    "recipe_search": ("Search recipes (TheMealDB).", "https://www.themealdb.com/api/json/v1/1/search.php?s={query}", "query:str", lambda j: [{"meal": m["strMeal"], "category": m["strCategory"], "area": m["strArea"], "instructions": m["strInstructions"][:400]} for m in (j.get("meals") or [])[:3]]),
    "cocktail_search": ("Search cocktails (TheCocktailDB).", "https://www.thecocktaildb.com/api/json/v1/1/search.php?s={query}", "query:str", lambda j: [{"drink": d["strDrink"], "glass": d["strGlass"], "instructions": d["strInstructions"][:300]} for d in (j.get("drinks") or [])[:3]]),
    "art_search": ("Search artworks at the Art Institute of Chicago.", "https://api.artic.edu/api/v1/artworks/search?q={query}&limit=5&fields=title,artist_display,date_display,image_id", "query:str", lambda j: j["data"]),
    "met_museum_search": ("Search the Metropolitan Museum of Art collection.", "https://collectionapi.metmuseum.org/public/collection/v1/search?q={query}&hasImages=true", "query:str", lambda j: {"total": j["total"], "object_ids": (j.get("objectIDs") or [])[:8]}),
    "space_launches": ("Upcoming rocket launches.", "https://ll.thespacedevs.com/2.2.0/launch/upcoming/?limit=5&mode=list", "", lambda j: [{"name": l["name"], "net": l["net"], "status": l["status"]["name"]} for l in j["results"]]),
    "spacex_latest": ("SpaceX's latest launch.", "https://api.spacexdata.com/v4/launches/latest", "", lambda j: {k: j.get(k) for k in ("name", "date_utc", "success", "details")}),
    "tv_search": ("Search TV shows (TVmaze).", "https://api.tvmaze.com/search/shows?q={query}", "query:str", lambda j: [{"name": s["show"]["name"], "premiered": s["show"].get("premiered"), "rating": s["show"]["rating"]["average"], "summary": re.sub(r"<[^>]+>", "", s["show"].get("summary") or "")[:250]} for s in j[:4]]),
    "npm_package": ("Info about an npm package.", "https://registry.npmjs.org/{name}/latest", "name:str", lambda j: {k: j.get(k) for k in ("name", "version", "description", "license", "homepage")}),
    "pypi_package": ("Info about a PyPI package.", "https://pypi.org/pypi/{name}/json", "name:str", lambda j: {k: j["info"].get(k) for k in ("name", "version", "summary", "license", "home_page", "requires_python")}),
    "crates_package": ("Info about a Rust crate.", "https://crates.io/api/v1/crates/{name}", "name:str", lambda j: {k: j["crate"].get(k) for k in ("name", "max_version", "description", "downloads")}),
    "stack_overflow_search": ("Search Stack Overflow questions.", "https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=votes&q={query}&site=stackoverflow&pagesize=5", "query:str", lambda j: [{"title": q["title"], "score": q["score"], "answered": q["is_answered"], "link": q["link"]} for q in j["items"]]),
    "hacker_news_top": ("Top Hacker News stories.", "https://hacker-news.firebaseio.com/v0/topstories.json", "", lambda j: j[:10]),
    "color_name_lookup": ("Name of a hex colour (thecolorapi).", "https://www.thecolorapi.com/id?hex={hex}", "hex:str", lambda j: {"name": j["name"]["value"], "hex": j["hex"]["value"], "rgb": j["rgb"]["value"]}),
    "qr_code_url": ("A URL that renders a QR code image for some text.", None, "text:str", lambda text: "https://api.qrserver.com/v1/create-qr-code/?size=300x300&data=" + urllib.parse.quote(text)),
    "worldbank_indicator": ("A World Bank indicator for a country, e.g. SP.POP.TOTL (population) for IN.", "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=8", "country:str=IN, indicator:str=SP.POP.TOTL", lambda j: [(r["date"], r["value"]) for r in j[1]]),
    "sunrise_sunset": ("Sunrise and sunset times at a location.", "https://api.sunrise-sunset.org/json?lat={lat}&lng={lon}&formatted=0", "lat:float, lon:float", lambda j: j["results"]),
    "timezone_now": ("Current time in a time zone like Asia/Kolkata (WorldTimeAPI).", "https://worldtimeapi.org/api/timezone/{zone}", "zone:str=Asia/Kolkata", lambda j: {k: j.get(k) for k in ("datetime", "timezone", "utc_offset", "day_of_week")}),
}


def _api_tool(name, desc, url, sig, post_):
    def run(ctx, **kw):
        if url is None:
            return post_(**kw)
        vals = {k: (urllib.parse.quote(str(v), safe="") if isinstance(v, str) else v) for k, v in kw.items()}
        j = get(ctx, url.format(**vals), as_text=name == "arxiv_search")
        if name == "arxiv_search":
            return [{"title": re.sub(r"\s+", " ", t).strip(), "link": l} for t, l in re.findall(r"<entry>.*?<title>(.*?)</title>.*?<id>(.*?)</id>", j, re.S)]
        return post_(j)
    reg(name, desc, "public_api", sig, run, risk="network", ctx=True, tags=("api", "data", "internet"))


for _n, (_d, _u, _s, _p) in APIS.items():
    _api_tool(_n, _d, _u, _s, _p)

reg("fetch_web_page_text", "Fetch an https web page and return its readable text (no logins, no scripts run).", N, "url:str, max_chars:int=6000", lambda ctx, url, max_chars: _page_text(ctx, url, max_chars), risk="network", ctx=True, tags=("browse", "research", "scrape"))
reg("fetch_json", "GET a public JSON API (https only).", N, "url:str, params:dict?", lambda ctx, url, params: get(ctx, url, params), risk="network", ctx=True, tags=("api",))
reg("download_file", "Download a file from an https URL into the workspace (max 40 MB).", N, "url:str, path:str", lambda ctx, url, path: _download(ctx, url, path, ()), risk="network", ctx=True)
reg("web_search_duckduckgo", "Instant-answer search (DuckDuckGo): summary and related topics, not a full results page.", N, "query:str", lambda ctx, query: (lambda j: {"abstract": j.get("AbstractText"), "source": j.get("AbstractURL"), "related": [t.get("Text") for t in j.get("RelatedTopics", []) if isinstance(t, dict) and t.get("Text")][:6]})(json.loads(get(ctx, "https://api.duckduckgo.com/", {"q": query, "format": "json", "no_html": 1}, as_text=True) or "{}")), risk="network", ctx=True, tags=("google", "search"))


def _page_text(ctx, url, n):
    u = urllib.parse.urlparse(url)
    if u.scheme != "https" or not u.hostname or re.fullmatch(r"(localhost|127\..*|10\..*|192\.168\..*|169\.254\..*|0\.0\.0\.0|\[.*\])", u.hostname) or u.hostname.startswith("172.") or u.hostname.endswith(".local"):
        raise ToolError("only public https:// pages are allowed")
    html = get(ctx, url, as_text=True, headers={"Accept": "text/html,text/plain"})
    html = re.sub(r"(?is)<(script|style|nav|footer|header|svg|form).*?</\1>", " ", html)
    t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    import html as h
    return h.unescape(t).strip()[:n]
