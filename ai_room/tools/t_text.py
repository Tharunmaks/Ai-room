"""Text, encoding, hashing, date/time, colour, and CSV/JSON data tools (all offline, pure Python)."""

from __future__ import annotations

import base64
import binascii
import calendar
import colorsys
import csv
import datetime as dt
import difflib
import hashlib
import html
import io
import json
import math
import re
import string
import textwrap
import unicodedata
import urllib.parse
import uuid
import zlib
from collections import Counter

from .registry import ToolError, reg

T = "text"
reg("word_count", "Count characters, words, lines and sentences in a text.", T, "text:str",
    lambda text: {"characters": len(text), "words": len(text.split()), "lines": len(text.splitlines()), "sentences": len(re.findall(r"[.!?]+(?:\s|$)", text))})
reg("reading_time", "Estimated reading time in minutes (230 wpm).", T, "text:str", lambda text: round(len(text.split()) / 230, 2))
reg("to_upper", "Upper-case a text.", T, "text:str", lambda text: text.upper())
reg("to_lower", "Lower-case a text.", T, "text:str", lambda text: text.lower())
reg("to_title", "Title-case a text.", T, "text:str", lambda text: string.capwords(text))
reg("snake_case", "Convert text to snake_case.", T, "text:str", lambda text: re.sub(r"[\W_]+", "_", re.sub(r"([a-z])([A-Z])", r"\1_\2", text)).strip("_").lower())
reg("camel_case", "Convert text to camelCase.", T, "text:str", lambda text: (lambda w: w[0].lower() + "".join(x.capitalize() for x in w[1:]) if w else "")(re.findall(r"[A-Za-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text))))
reg("kebab_case", "Convert text to kebab-case.", T, "text:str", lambda text: re.sub(r"[\W_]+", "-", re.sub(r"([a-z])([A-Z])", r"\1-\2", text)).strip("-").lower())
reg("slugify", "URL-friendly slug.", T, "text:str", lambda text: re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()).strip("-"))
reg("reverse_text", "Reverse a text.", T, "text:str", lambda text: text[::-1])
reg("reverse_words", "Reverse the order of words.", T, "text:str", lambda text: " ".join(reversed(text.split())))
reg("sort_lines", "Sort lines alphabetically.", T, "text:str, reverse:bool=false", lambda text, reverse: "\n".join(sorted(text.splitlines(), key=str.lower, reverse=reverse)))
reg("unique_lines", "Remove duplicate lines, keeping order.", T, "text:str", lambda text: "\n".join(dict.fromkeys(text.splitlines())))
reg("remove_extra_spaces", "Collapse repeated whitespace.", T, "text:str", lambda text: re.sub(r"\s+", " ", text).strip())
reg("strip_accents", "Remove accents from letters.", T, "text:str", lambda text: "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)))
reg("find_replace", "Replace text (optionally regex).", T, "text:str, find:str, replace:str, regex:bool=false", lambda text, find, replace, regex: re.sub(find, replace, text) if regex else text.replace(find, replace))
reg("regex_find_all", "All matches of a regular expression.", T, "text:str, pattern:str", lambda text, pattern: re.findall(pattern, text)[:200])
reg("regex_test", "Does the pattern match anywhere?", T, "text:str, pattern:str", lambda text, pattern: bool(re.search(pattern, text)))
reg("extract_emails", "Email addresses in a text.", T, "text:str", lambda text: re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text))
reg("extract_urls", "URLs in a text.", T, "text:str", lambda text: re.findall(r"https?://[^\s<>\"')]+", text))
reg("extract_numbers", "Numbers in a text.", T, "text:str", lambda text: [float(x) if "." in x else int(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)])
reg("extract_hashtags", "Hashtags in a text.", T, "text:str", lambda text: re.findall(r"#\w+", text))
reg("extract_mentions", "@mentions in a text.", T, "text:str", lambda text: re.findall(r"@\w+", text))
reg("word_frequency", "Most common words.", T, "text:str, top:int=10", lambda text, top: Counter(re.findall(r"[^\W\d_]+", text.lower())).most_common(top))
reg("character_frequency", "Most common characters.", T, "text:str, top:int=10", lambda text, top: Counter(c for c in text if not c.isspace()).most_common(top))
reg("longest_word", "Longest word.", T, "text:str", lambda text: max(re.findall(r"\w+", text), key=len, default=""))
reg("is_palindrome", "Is the text a palindrome (ignoring case and punctuation)?", T, "text:str", lambda text: (lambda s: s == s[::-1])(re.sub(r"[\W_]", "", text.lower())))
reg("is_anagram", "Are the two texts anagrams?", T, "a:str, b:str", lambda a, b: sorted(re.sub(r"\W", "", a.lower())) == sorted(re.sub(r"\W", "", b.lower())))
reg("caesar_cipher", "Caesar shift (negative to decode).", "crypto", "text:str, shift:int=3", lambda text, shift: "".join(chr((ord(c) - (65 if c.isupper() else 97) + shift) % 26 + (65 if c.isupper() else 97)) if c.isascii() and c.isalpha() else c for c in text))
reg("rot13", "ROT13.", "crypto", "text:str", lambda text: __import__("codecs").encode(text, "rot13"))
reg("vigenere", "Vigenère cipher; decode=true to decrypt.", "crypto", "text:str, key:str, decode:bool=false",
    lambda text, key, decode: _vig(text, key, decode))
reg("atbash", "Atbash cipher.", "crypto", "text:str", lambda text: "".join(chr(155 - ord(c)) if c.isascii() and c.isupper() else chr(219 - ord(c)) if c.isascii() and c.islower() else c for c in text))
_MORSE = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", ".- -... -.-. -.. . ..-. --. .... .. .--- -.- .-.. -- -. --- .--. --.- .-. ... - ..- ...- .-- -..- -.-- --.. -----  .---- ..--- ...-- ....- ..... -.... --... ---.. ----.".replace("  ", " ").split()))
reg("to_morse", "Text to Morse code.", "crypto", "text:str", lambda text: " / ".join(" ".join(_MORSE.get(c, "?") for c in w) for w in text.upper().split()))
reg("from_morse", "Morse code to text (words separated by ' / ').", "crypto", "morse:str", lambda morse: " ".join("".join({v: k for k, v in _MORSE.items()}.get(c, "?") for c in w.split()) for w in morse.split("/")))
reg("text_to_binary", "Text to 8-bit binary.", "crypto", "text:str", lambda text: " ".join(f"{b:08b}" for b in text.encode()))
reg("binary_to_text", "8-bit binary to text.", "crypto", "binary:str", lambda binary: bytes(int(b, 2) for b in binary.split()).decode(errors="replace"))
reg("text_to_hex", "Text to hex.", "crypto", "text:str", lambda text: text.encode().hex())
reg("hex_to_text", "Hex to text.", "crypto", "hex:str", lambda hex: bytes.fromhex(hex.replace(" ", "")).decode(errors="replace"))
reg("base64_encode", "Base64 encode.", "crypto", "text:str", lambda text: base64.b64encode(text.encode()).decode())
reg("base64_decode", "Base64 decode.", "crypto", "data:str", lambda data: base64.b64decode(data + "=" * (-len(data) % 4)).decode(errors="replace"))
reg("base32_encode", "Base32 encode.", "crypto", "text:str", lambda text: base64.b32encode(text.encode()).decode())
reg("base85_encode", "Base85 encode.", "crypto", "text:str", lambda text: base64.b85encode(text.encode()).decode())
reg("url_encode", "Percent-encode text for URLs.", "crypto", "text:str", lambda text: urllib.parse.quote(text, safe=""))
reg("url_decode", "Decode percent-encoded text.", "crypto", "text:str", lambda text: urllib.parse.unquote(text))
reg("url_parse", "Split a URL into parts.", "crypto", "url:str", lambda url: (lambda u: {"scheme": u.scheme, "host": u.hostname, "port": u.port, "path": u.path, "query": urllib.parse.parse_qs(u.query), "fragment": u.fragment})(urllib.parse.urlparse(url)))
reg("html_escape", "Escape HTML characters.", "crypto", "text:str", lambda text: html.escape(text))
reg("html_unescape", "Unescape HTML entities.", "crypto", "text:str", lambda text: html.unescape(text))
reg("strip_html_tags", "Remove HTML tags.", T, "html:str", lambda html: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<(script|style).*?</\1>", "", html))).strip())
for _alg in sorted(a for a in hashlib.algorithms_guaranteed if not a.startswith("shake")):
    reg(f"hash_{_alg}", f"{_alg.upper()} hash of a text.", "crypto", "text:str", (lambda a: lambda text: hashlib.new(a, text.encode()).hexdigest())(_alg))
reg("crc32", "CRC-32 checksum of a text.", "crypto", "text:str", lambda text: f"{zlib.crc32(text.encode()):08x}")
reg("adler32", "Adler-32 checksum of a text.", "crypto", "text:str", lambda text: f"{zlib.adler32(text.encode()):08x}")
reg("hmac_sha256", "HMAC-SHA256 of a message with a key.", "crypto", "message:str, key:str", lambda message, key: __import__("hmac").new(key.encode(), message.encode(), "sha256").hexdigest())
reg("generate_uuid", "A random UUID4.", "crypto", "", lambda: str(uuid.uuid4()))
reg("uuid_from_name", "Deterministic UUID5 from a name.", "crypto", "name:str", lambda name: str(uuid.uuid5(uuid.NAMESPACE_DNS, name)))
reg("generate_password", "Random strong password.", "crypto", "length:int=16, symbols:bool=true", lambda length, symbols: "".join(__import__("secrets").choice(string.ascii_letters + string.digits + ("!@#$%^&*-_" if symbols else "")) for _ in range(min(max(length, 4), 128))))
reg("password_strength", "Rough password strength estimate in bits.", "crypto", "password:str", lambda password: {"bits": round(len(password) * math.log2(max(2, sum(n for n, ok in ((26, re.search("[a-z]", password)), (26, re.search("[A-Z]", password)), (10, re.search(r"\d", password)), (32, re.search(r"[^\w]", password))) if ok))), 1)})
reg("lorem_ipsum", "Placeholder text of N words.", T, "words:int=50", lambda words: " ".join(("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua " * 40).split()[:min(words, 800)]).capitalize() + ".")
reg("text_diff", "Unified diff of two texts.", T, "a:str, b:str", lambda a, b: "\n".join(difflib.unified_diff(a.splitlines(), b.splitlines(), "a", "b", lineterm="")))
reg("text_similarity", "Similarity (0–1) of two texts.", T, "a:str, b:str", lambda a, b: round(difflib.SequenceMatcher(None, a, b).ratio(), 4))
reg("edit_distance", "Levenshtein distance between two texts.", T, "a:str, b:str", lambda a, b: _lev(a, b))
reg("wrap_text", "Wrap text to a width.", T, "text:str, width:int=70", lambda text, width: textwrap.fill(text, width))
reg("indent_text", "Indent every line.", T, "text:str, spaces:int=4", lambda text, spaces: textwrap.indent(text, " " * spaces))
reg("truncate_text", "Cut a text to N characters with an ellipsis.", T, "text:str, length:int=100", lambda text, length: text if len(text) <= length else text[:length - 1].rstrip() + "…")
reg("count_syllables", "Estimate syllables in English text.", T, "text:str", lambda text: sum(_syl(w) for w in re.findall(r"[A-Za-z]+", text)))
reg("flesch_reading_ease", "Flesch reading-ease score (higher is easier).", T, "text:str",
    lambda text: (lambda w, s, y: round(206.835 - 1.015 * w / s - 84.6 * y / w, 1))(max(1, len(re.findall(r"[A-Za-z']+", text))), max(1, len(re.findall(r"[.!?]+", text))), sum(_syl(x) for x in re.findall(r"[A-Za-z]+", text)) or 1))
reg("detect_script", "Which writing systems appear in the text.", T, "text:str", lambda text: dict(Counter(unicodedata.name(c, "?").split()[0] for c in text if c.isalpha())))
reg("unicode_info", "Name and code point of each character.", T, "text:str", lambda text: [{"char": c, "code": f"U+{ord(c):04X}", "name": unicodedata.name(c, "?")} for c in text[:50]])
reg("numbers_to_text_table", "Pretty-print rows as an aligned text table.", T, "rows:list", lambda rows: "\n".join("  ".join(str(c).ljust(max(len(str(r[i])) for r in rows)) for i, c in enumerate(row)) for row in rows))
reg("markdown_table", "Markdown table from rows (first row is the header).", T, "rows:list", lambda rows: "\n".join(["| " + " | ".join(map(str, rows[0])) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(map(str, r)) + " |" for r in rows[1:]]))
reg("make_acronym", "Acronym from a phrase.", T, "text:str", lambda text: "".join(w[0] for w in text.split() if w).upper())
reg("pig_latin", "Translate to Pig Latin.", T, "text:str", lambda text: " ".join((w + "way") if w[:1].lower() in "aeiou" else (re.sub(r"^([^aeiou]+)(.*)$", r"\2\1ay", w, flags=re.I)) for w in text.split()))
reg("number_lines", "Prefix lines with numbers.", T, "text:str", lambda text: "\n".join(f"{i}: {l}" for i, l in enumerate(text.splitlines(), 1)))
reg("sentence_split", "Split text into sentences.", T, "text:str", lambda text: [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()])
reg("keyword_extract", "Top keywords by frequency (stop-words removed).", T, "text:str, top:int=8",
    lambda text, top: [w for w, _ in Counter(w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP).most_common(top)])
reg("summarize_extractive", "Pick the most informative sentences (frequency scoring) — no AI.", T, "text:str, sentences:int=3", lambda text, sentences: _summ(text, sentences))
reg("json_pretty", "Pretty-print JSON.", "data", "text:str", lambda text: json.dumps(json.loads(text), indent=2, ensure_ascii=False))
reg("json_minify", "Minify JSON.", "data", "text:str", lambda text: json.dumps(json.loads(text), separators=(",", ":"), ensure_ascii=False))
reg("json_validate", "Is this valid JSON?", "data", "text:str", lambda text: _jv(text))
reg("json_get", "Read a value from JSON by dotted path like a.b.0.c.", "data", "text:str, path:str", lambda text, path: _jget(json.loads(text), path))
reg("json_keys", "Keys of a JSON object (recursive paths).", "data", "text:str", lambda text: _paths(json.loads(text))[:200])
reg("json_to_csv", "Convert a JSON array of objects to CSV.", "data", "text:str", lambda text: _j2c(json.loads(text)))
reg("csv_to_json", "Convert CSV text to a JSON array of objects.", "data", "text:str", lambda text: json.dumps(list(csv.DictReader(io.StringIO(text))), ensure_ascii=False, indent=1))
reg("csv_summary", "Rows, columns and per-column numeric summary of CSV text.", "data", "text:str", lambda text: _csum(text))
reg("csv_column", "Values of one CSV column.", "data", "text:str, column:str", lambda text, column: [r[column] for r in csv.DictReader(io.StringIO(text))][:500])
reg("csv_filter", "CSV rows where column equals a value.", "data", "text:str, column:str, value:str", lambda text, column, value: [r for r in csv.DictReader(io.StringIO(text)) if r.get(column) == value][:200])
reg("csv_sort", "Sort CSV rows by a column.", "data", "text:str, column:str, descending:bool=false", lambda text, column, descending: _csort(text, column, descending))
reg("csv_group_sum", "Group CSV rows by a column and sum another.", "data", "text:str, group:str, value:str", lambda text, group, value: _cgroup(text, group, value))
reg("csv_stats_column", "Mean/min/max of a numeric CSV column.", "data", "text:str, column:str", lambda text, column: _ccol(text, column))
reg("transpose_rows", "Transpose a list of rows.", "data", "rows:list", lambda rows: [list(r) for r in zip(*rows)])
reg("dedupe_list", "Remove duplicates from a list.", "data", "items:list", lambda items: list(dict.fromkeys(map(str, items))))
reg("sort_list", "Sort a list of numbers or text.", "data", "items:list, reverse:bool=false", lambda items, reverse: sorted(items, reverse=reverse))
reg("flatten_list", "Flatten a nested list.", "data", "items:list", lambda items: _flat(items))
reg("chunk_list", "Split a list into chunks.", "data", "items:list, size:int", lambda items, size: [items[i:i + size] for i in range(0, len(items), size)])
reg("list_intersection", "Items in both lists.", "data", "a:list, b:list", lambda a, b: [x for x in a if x in b])
reg("list_difference", "Items in a but not b.", "data", "a:list, b:list", lambda a, b: [x for x in a if x not in b])
reg("count_items", "Count occurrences of each item.", "data", "items:list", lambda items: dict(Counter(map(str, items))))
reg("range_list", "Numbers from start to stop by step.", "data", "start:int, stop:int, step:int=1", lambda start, stop, step: list(range(start, stop, step))[:2000])
reg("yaml_like_to_json", "Parse simple 'key: value' lines into JSON.", "data", "text:str", lambda text: json.dumps({k.strip(): v.strip() for k, _, v in (l.partition(":") for l in text.splitlines() if ":" in l)}, indent=1))
reg("ini_to_json", "Parse INI text into JSON.", "data", "text:str", lambda text: (lambda cp: json.dumps({s: dict(cp[s]) for s in cp.sections()}, indent=1))((lambda c: (c.read_string(text), c)[1])(__import__("configparser").ConfigParser())))
reg("query_string_build", "Build a URL query string from a dict.", "data", "params:dict", lambda params: urllib.parse.urlencode(params))

# ---- date and time
D = "datetime"
_now = lambda: dt.datetime.now(dt.timezone.utc)  # noqa: E731
reg("current_time_utc", "The current UTC date and time.", D, "", lambda: _now().isoformat(timespec="seconds"))
reg("current_date", "Today's date (UTC).", D, "", lambda: _now().date().isoformat())
reg("unix_timestamp", "Current Unix timestamp.", D, "", lambda: int(_now().timestamp()))
reg("timestamp_to_date", "Unix timestamp to an ISO date-time (UTC).", D, "timestamp:float", lambda timestamp: dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat())
reg("date_to_timestamp", "ISO date to a Unix timestamp (UTC).", D, "date:str", lambda date: int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()))
reg("day_of_week", "Weekday name of a date (YYYY-MM-DD).", D, "date:str", lambda date: dt.date.fromisoformat(date).strftime("%A"))
reg("days_between", "Days between two dates.", D, "start:str, end:str", lambda start, end: (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days)
reg("add_days", "Date after adding days.", D, "date:str, days:int", lambda date, days: (dt.date.fromisoformat(date) + dt.timedelta(days=days)).isoformat())
reg("add_months", "Date after adding months.", D, "date:str, months:int", lambda date, months: _addm(date, months))
reg("age_from_birthdate", "Age in years, months and days.", D, "birthdate:str, on:str?", lambda birthdate, on: _age(birthdate, on))
reg("is_leap_year", "Is the year a leap year?", D, "year:int", lambda year: calendar.isleap(year))
reg("days_in_month", "Days in a month.", D, "year:int, month:int", lambda year, month: calendar.monthrange(year, month)[1])
reg("month_calendar", "Text calendar for a month.", D, "year:int, month:int", lambda year, month: calendar.month(year, month))
reg("week_number", "ISO week number of a date.", D, "date:str", lambda date: dt.date.fromisoformat(date).isocalendar()[1])
reg("day_of_year", "Day number within the year.", D, "date:str", lambda date: dt.date.fromisoformat(date).timetuple().tm_yday)
reg("business_days_between", "Weekdays between two dates.", D, "start:str, end:str", lambda start, end: sum(1 for i in range((dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days) if (dt.date.fromisoformat(start) + dt.timedelta(days=i)).weekday() < 5))
reg("convert_timezone", "Convert a time between UTC offsets, e.g. 2025-01-01T09:00 from +0 to +5.5.", D, "datetime:str, from_offset_hours:float, to_offset_hours:float",
    lambda datetime, from_offset_hours, to_offset_hours: (dt.datetime.fromisoformat(datetime) + dt.timedelta(hours=to_offset_hours - from_offset_hours)).isoformat())
reg("seconds_to_hms", "Seconds to H:MM:SS.", D, "seconds:float", lambda seconds: str(dt.timedelta(seconds=int(seconds))))
reg("hms_to_seconds", "H:MM:SS to seconds.", D, "hms:str", lambda hms: sum(float(p) * 60 ** i for i, p in enumerate(reversed(hms.split(":")))))
reg("easter_date", "Date of Easter Sunday (Gregorian).", D, "year:int", lambda year: _easter(year))
reg("zodiac_sign", "Western zodiac sign for a month and day.", D, "month:int, day:int", lambda month, day: _zodiac(month, day))
reg("countdown_days", "Days from today until a date.", D, "date:str", lambda date: (dt.date.fromisoformat(date) - _now().date()).days)
reg("julian_day", "Julian day number of a date.", D, "date:str", lambda date: dt.date.fromisoformat(date).toordinal() + 1721424.5)
reg("moon_phase", "Approximate moon phase (0 new … 0.5 full) for a date.", D, "date:str", lambda date: round(((dt.date.fromisoformat(date).toordinal() + 1721424.5 - 2451550.1) / 29.530588853) % 1, 3))

# ---- colour
K = "colour"
reg("hex_to_rgb", "#RRGGBB to RGB.", K, "hex:str", lambda hex: _hex2rgb(hex))
reg("rgb_to_hex", "RGB to #RRGGBB.", K, "r:int, g:int, b:int", lambda r, g, b: f"#{r:02x}{g:02x}{b:02x}")
reg("rgb_to_hsl", "RGB to HSL.", K, "r:int, g:int, b:int", lambda r, g, b: (lambda h, l, s: {"h": round(h * 360), "s": round(s * 100), "l": round(l * 100)})(*colorsys.rgb_to_hls(r / 255, g / 255, b / 255)))
reg("hsl_to_rgb", "HSL to RGB.", K, "h:float, s:float, l:float", lambda h, s, l: [round(x * 255) for x in colorsys.hls_to_rgb(h / 360, l / 100, s / 100)])
reg("rgb_to_hsv", "RGB to HSV.", K, "r:int, g:int, b:int", lambda r, g, b: (lambda h, s, v: {"h": round(h * 360), "s": round(s * 100), "v": round(v * 100)})(*colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)))
reg("rgb_to_cmyk", "RGB to CMYK.", K, "r:int, g:int, b:int", lambda r, g, b: (lambda k: {"c": 0, "m": 0, "y": 0, "k": 100} if k == 1 else {"c": round((1 - r / 255 - k) / (1 - k) * 100), "m": round((1 - g / 255 - k) / (1 - k) * 100), "y": round((1 - b / 255 - k) / (1 - k) * 100), "k": round(k * 100)})(1 - max(r, g, b) / 255))
reg("contrast_ratio", "WCAG contrast ratio between two hex colours.", K, "foreground:str, background:str", lambda foreground, background: _contrast(foreground, background))
reg("lighten_color", "Lighten a hex colour by a percentage.", K, "hex:str, percent:float=20", lambda hex, percent: _shade(hex, percent / 100))
reg("darken_color", "Darken a hex colour by a percentage.", K, "hex:str, percent:float=20", lambda hex, percent: _shade(hex, -percent / 100))
reg("mix_colors", "Mix two hex colours.", K, "a:str, b:str, ratio:float=0.5", lambda a, b, ratio: "#" + "".join(f"{round(x * (1 - ratio) + y * ratio):02x}" for x, y in zip(_hex2rgb(a), _hex2rgb(b))))
reg("complementary_color", "The complementary colour.", K, "hex:str", lambda hex: "#" + "".join(f"{255 - c:02x}" for c in _hex2rgb(hex)))
reg("color_palette", "A harmonious palette from a base colour (analogous, triadic, complementary).", K, "hex:str", lambda hex: _palette(hex))
reg("gradient_colors", "N colours between two hex colours.", K, "a:str, b:str, steps:int=5", lambda a, b, steps: ["#" + "".join(f"{round(x + (y - x) * i / max(1, steps - 1)):02x}" for x, y in zip(_hex2rgb(a), _hex2rgb(b))) for i in range(steps)])
reg("color_name", "Nearest common colour name.", K, "hex:str", lambda hex: min(_NAMED, key=lambda n: sum((a - b) ** 2 for a, b in zip(_NAMED[n], _hex2rgb(hex)))))
reg("random_color", "A random hex colour.", K, "seed:int?", lambda seed: "#%06x" % __import__("random").Random(seed).randrange(1 << 24))
reg("kelvin_to_rgb", "Approximate RGB of a blackbody colour temperature (1000–40000 K).", K, "kelvin:float", lambda kelvin: _kelvin(kelvin))

_STOP = set("the a an and or but of to in on at for with from by is are was were be been it this that these those as not no you your we our they their he she his her its have has had do does did will would can could should may might about into over than then there which who whom what when where why how also more most such some any".split())
_NAMED = {"black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0), "green": (0, 128, 0), "lime": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
          "gray": (128, 128, 128), "silver": (192, 192, 192), "maroon": (128, 0, 0), "olive": (128, 128, 0), "navy": (0, 0, 128), "purple": (128, 0, 128), "teal": (0, 128, 128), "orange": (255, 165, 0),
          "pink": (255, 192, 203), "brown": (165, 42, 42), "gold": (255, 215, 0), "indigo": (75, 0, 130), "violet": (238, 130, 238), "coral": (255, 127, 80), "salmon": (250, 128, 114), "turquoise": (64, 224, 208), "beige": (245, 245, 220)}


def _vig(text, key, decode):
    k = [ord(c.lower()) - 97 for c in key if c.isalpha()]
    if not k:
        raise ToolError("key must contain letters")
    out, i = [], 0
    for c in text:
        if c.isascii() and c.isalpha():
            base = 65 if c.isupper() else 97
            out.append(chr((ord(c) - base + (-k[i % len(k)] if decode else k[i % len(k)])) % 26 + base))
            i += 1
        else:
            out.append(c)
    return "".join(out)


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _syl(w):
    w = w.lower()
    n = len(re.findall(r"[aeiouy]+", w)) - (1 if w.endswith("e") and not w.endswith("le") and len(w) > 2 else 0)
    return max(1, n)


def _summ(text, k):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 3]
    freq = Counter(w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP)
    score = {s: sum(freq[w] for w in re.findall(r"[a-z]{3,}", s.lower())) / (len(s.split()) ** 0.7) for s in sents}
    keep = set(sorted(score, key=score.get, reverse=True)[:k])
    return " ".join(s for s in sents if s in keep)


def _jv(t):
    try:
        json.loads(t)
        return {"valid": True}
    except json.JSONDecodeError as e:
        return {"valid": False, "error": str(e)}


def _jget(o, path):
    for p in [x for x in path.split(".") if x]:
        try:
            o = o[int(p)] if isinstance(o, list) else o[p]
        except (KeyError, IndexError, ValueError, TypeError):
            raise ToolError(f"path not found at '{p}'")
    return o


def _paths(o, pre=""):
    if isinstance(o, dict):
        return [p for k, v in o.items() for p in _paths(v, f"{pre}{k}.")] or [pre.rstrip(".")]
    if isinstance(o, list):
        return [p for i, v in enumerate(o[:20]) for p in _paths(v, f"{pre}{i}.")] or [pre.rstrip(".")]
    return [pre.rstrip(".")]


def _j2c(rows):
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise ToolError("expected a JSON array of objects")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out = io.StringIO()
    w = csv.DictWriter(out, cols)
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


def _rows(text):
    r = list(csv.DictReader(io.StringIO(text)))
    if not r:
        raise ToolError("no rows found (the first line must be a header)")
    return r


def _csum(text):
    rows = _rows(text)
    out = {"rows": len(rows), "columns": {}}
    for c in rows[0]:
        vals = [r[c] for r in rows]
        try:
            nums = [float(v) for v in vals if v != ""]
            out["columns"][c] = {"type": "number", "min": min(nums), "max": max(nums), "mean": round(sum(nums) / len(nums), 6)}
        except ValueError:
            out["columns"][c] = {"type": "text", "unique": len(set(vals)), "top": Counter(vals).most_common(3)}
    return out


def _csort(text, col, desc):
    rows = _rows(text)

    def key(r):
        try:
            return (0, float(r[col]))
        except ValueError:
            return (1, r[col])
    rows.sort(key=key, reverse=desc)
    return _j2c(rows)


def _cgroup(text, g, v):
    out: dict = {}
    for r in _rows(text):
        out[r[g]] = out.get(r[g], 0) + float(r[v])
    return {k: round(x, 6) for k, x in out.items()}


def _ccol(text, c):
    nums = [float(r[c]) for r in _rows(text) if r[c] != ""]
    return {"count": len(nums), "mean": sum(nums) / len(nums), "min": min(nums), "max": max(nums), "sum": sum(nums)}


def _flat(x):
    return [y for i in x for y in (_flat(i) if isinstance(i, list) else [i])]


def _addm(d, m):
    d0 = dt.date.fromisoformat(d)
    y, mo = divmod(d0.month - 1 + m, 12)
    year, month = d0.year + y, mo + 1
    return dt.date(year, month, min(d0.day, calendar.monthrange(year, month)[1])).isoformat()


def _age(b, on):
    b0 = dt.date.fromisoformat(b)
    t = dt.date.fromisoformat(on) if on else _now().date()
    y = t.year - b0.year - ((t.month, t.day) < (b0.month, b0.day))
    m = (t.month - b0.month - (t.day < b0.day)) % 12
    d = (t - dt.date(*divmod(t.year * 12 + t.month - 1 - m, 12)[0:1], (t.month - 1 - m) % 12 + 1, 1)).days if False else (t - dt.date.fromisoformat(_addm(b, y * 12 + m))).days
    return {"years": y, "months": m, "days": d, "total_days": (t - b0).days}


def _easter(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mo, da = divmod(h + l - 7 * m + 114, 31)
    return dt.date(y, mo, da + 1).isoformat()


def _zodiac(m, d):
    signs = [(1, 20, "Capricorn"), (2, 19, "Aquarius"), (3, 21, "Pisces"), (4, 20, "Aries"), (5, 21, "Taurus"), (6, 21, "Gemini"), (7, 23, "Cancer"), (8, 23, "Leo"), (9, 23, "Virgo"), (10, 23, "Libra"), (11, 22, "Scorpio"), (12, 22, "Sagittarius")]
    return next((n for mm, dd, n in signs if (m, d) < (mm, dd)), "Capricorn") if (m, d) < (12, 22) else "Capricorn"


def _hex2rgb(h):
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        raise ToolError("colour must look like #1a2b3c")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def _lum(rgb):
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    r, g, b = (f(c / 255) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    la, lb = sorted((_lum(_hex2rgb(a)), _lum(_hex2rgb(b))), reverse=True)
    r = (la + 0.05) / (lb + 0.05)
    return {"ratio": round(r, 2), "AA_normal": r >= 4.5, "AA_large": r >= 3, "AAA_normal": r >= 7}


def _shade(h, amt):
    h_, l, s = colorsys.rgb_to_hls(*[c / 255 for c in _hex2rgb(h)])
    r, g, b = colorsys.hls_to_rgb(h_, min(1, max(0, l + amt)), s)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def _palette(h):
    hh, l, s = colorsys.rgb_to_hls(*[c / 255 for c in _hex2rgb(h)])

    def rot(deg):
        r, g, b = colorsys.hls_to_rgb((hh + deg / 360) % 1, l, s)
        return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"
    return {"base": h, "complementary": rot(180), "analogous": [rot(-30), rot(30)], "triadic": [rot(120), rot(240)], "split_complementary": [rot(150), rot(210)], "tetradic": [rot(90), rot(180), rot(270)]}


def _kelvin(k):
    t = min(max(k, 1000), 40000) / 100
    r = 255 if t <= 66 else min(255, max(0, 329.698727446 * (t - 60) ** -0.1332047592))
    g = min(255, max(0, 99.4708025861 * math.log(t) - 161.1195681661)) if t <= 66 else min(255, max(0, 288.1221695283 * (t - 60) ** -0.0755148492))
    b = 255 if t >= 66 else 0 if t <= 19 else min(255, max(0, 138.5177312231 * math.log(t - 10) - 305.0447927307))
    return "#%02x%02x%02x" % (round(r), round(g), round(b))
