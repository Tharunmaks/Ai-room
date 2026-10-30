"""Foreign-language families: vocabulary tables and rule-based regular grammar.

All answers come from the tables below or from regular conjugation rules, so
they are exact. Only regular verbs are used so the rules never need exceptions.
Accents are ignored when grading (see grading.cloze_match).
"""

from __future__ import annotations

from .generators import Q, family, mcq

# (english, target, romanisation-or-empty)
Word = tuple[str, str, str]


def _w(pairs: str) -> list[Word]:
    out = []
    for item in pairs.strip().splitlines():
        parts = [p.strip() for p in item.split("|")]
        out.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
    return out


NUM_EN = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
          "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]

VOCAB: dict[str, dict[str, list[Word]]] = {
    "spanish": {
        "numbers": list(zip(NUM_EN, "uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciséis diecisiete dieciocho diecinueve veinte".split(), [""] * 20)),
        "colors": _w("red|rojo\nblue|azul\ngreen|verde\nyellow|amarillo\nblack|negro\nwhite|blanco\norange|naranja\npink|rosa\ngray|gris\nbrown|marrón"),
        "family": _w("mother|madre\nfather|padre\nbrother|hermano\nsister|hermana\ngrandfather|abuelo\ngrandmother|abuela\nson|hijo\ndaughter|hija\nuncle|tío\naunt|tía\ncousin|primo"),
        "food": _w("bread|pan\nmilk|leche\napple|manzana\ncheese|queso\nchicken|pollo\nrice|arroz\negg|huevo\nfish|pescado"),
        "days": _w("Monday|lunes\nTuesday|martes\nWednesday|miércoles\nThursday|jueves\nFriday|viernes\nSaturday|sábado\nSunday|domingo"),
        "animals": _w("dog|perro\ncat|gato\nhorse|caballo\nbird|pájaro\ncow|vaca"),
        "things": _w("house|casa\nbook|libro\nschool|escuela\ncity|ciudad\nfriend|amigo\nwork|trabajo\ntime|tiempo\nday|día\nnight|noche"),
    },
    "french": {
        "numbers": list(zip(NUM_EN, "un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize dix-sept dix-huit dix-neuf vingt".split(), [""] * 20)),
        "colors": _w("red|rouge\nblue|bleu\ngreen|vert\nyellow|jaune\nblack|noir\nwhite|blanc\norange|orange\npink|rose\ngray|gris\nbrown|marron"),
        "family": _w("mother|mère\nfather|père\nbrother|frère\nsister|sœur\ngrandfather|grand-père\ngrandmother|grand-mère\nson|fils\ndaughter|fille\nuncle|oncle\naunt|tante\ncousin|cousin"),
        "food": _w("bread|pain\nwater|eau\nmilk|lait\napple|pomme\ncheese|fromage\nchicken|poulet\nrice|riz\negg|œuf\nfish|poisson"),
        "days": _w("Monday|lundi\nTuesday|mardi\nWednesday|mercredi\nThursday|jeudi\nFriday|vendredi\nSaturday|samedi\nSunday|dimanche"),
        "animals": _w("dog|chien\ncat|chat\nhorse|cheval\nbird|oiseau\ncow|vache"),
        "things": _w("house|maison\nbook|livre\nschool|école\ncity|ville\nfriend|ami\nwork|travail\ntime|temps\nday|jour\nnight|nuit"),
    },
    "german": {
        "numbers": list(zip(NUM_EN, "eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf dreizehn vierzehn fünfzehn sechzehn siebzehn achtzehn neunzehn zwanzig".split(), [""] * 20)),
        "colors": _w("red|rot\nblue|blau\ngreen|grün\nyellow|gelb\nblack|schwarz\nwhite|weiß\norange|orange\npink|rosa\ngray|grau\nbrown|braun"),
        "family": _w("mother|Mutter\nfather|Vater\nbrother|Bruder\nsister|Schwester\ngrandfather|Großvater\ngrandmother|Großmutter\nson|Sohn\ndaughter|Tochter\nuncle|Onkel\naunt|Tante\ncousin|Cousin"),
        "food": _w("bread|Brot\nwater|Wasser\nmilk|Milch\napple|Apfel\ncheese|Käse\nrice|Reis\negg|Ei\nfish|Fisch"),
        "days": _w("Monday|Montag\nTuesday|Dienstag\nWednesday|Mittwoch\nThursday|Donnerstag\nFriday|Freitag\nSaturday|Samstag\nSunday|Sonntag"),
        "animals": _w("dog|Hund\ncat|Katze\nhorse|Pferd\nbird|Vogel\ncow|Kuh"),
        "things": _w("house|Haus\nbook|Buch\nschool|Schule\ncity|Stadt\nfriend|Freund\nwork|Arbeit\ntime|Zeit\nday|Tag\nnight|Nacht"),
    },
    "mandarin": {
        "numbers": _w("one|一|yī\ntwo|二|èr\nthree|三|sān\nfour|四|sì\nfive|五|wǔ\nsix|六|liù\nseven|七|qī\neight|八|bā\nnine|九|jiǔ\nten|十|shí"),
        "basics": _w("hello|你好|nǐ hǎo\nthank you|谢谢|xièxie\ngoodbye|再见|zàijiàn\nI|我|wǒ\nyou|你|nǐ\nhe|他|tā\ngood|好|hǎo\nnot|不|bù\nbig|大|dà\nsmall|小|xiǎo"),
        "things": _w("water|水|shuǐ\nperson|人|rén\nbook|书|shū\nhome|家|jiā\nschool|学校|xuéxiào\nfriend|朋友|péngyǒu\nteacher|老师|lǎoshī\nstudent|学生|xuésheng\nChina|中国|Zhōngguó\nmountain|山|shān\nfire|火|huǒ"),
        "verbs": _w("eat|吃|chī\ndrink|喝|hē\nlove|爱|ài\nbe|是|shì\nhave|有|yǒu"),
    },
    "hindi": {
        "numbers": _w("one|एक|ek\ntwo|दो|do\nthree|तीन|tīn\nfour|चार|chār\nsix|छह|chhah\nseven|सात|sāt\neight|आठ|āṭh\nnine|नौ|nau\nten|दस|das"),
        "colors": _w("red|लाल|lāl\nblue|नीला|nīlā\ngreen|हरा|harā\nyellow|पीला|pīlā"),
        "things": _w("water|पानी|pānī\nhouse|घर|ghar\nbook|किताब|kitāb\nfriend|दोस्त|dost\nschool|स्कूल|skūl\ncity|शहर|shahar\nday|दिन|din\nnight|रात|rāt\nfood|खाना|khānā"),
        "family": _w("mother|माँ|mā\nfather|पिता|pitā\nbrother|भाई|bhāī\nsister|बहन|bahan\nboy|लड़का|laṛkā\ngirl|लड़की|laṛkī"),
    },
}
NAMES = {"spanish": "Spanish", "french": "French", "german": "German", "mandarin": "Mandarin Chinese", "hindi": "Hindi"}
TIERS = {"spanish": [["numbers", "colors"], ["family", "food", "days", "animals", "things"]], "french": [["numbers", "colors"], ["family", "food", "days", "animals", "things"]],
         "german": [["numbers", "colors"], ["family", "food", "days", "animals", "things"]], "mandarin": [["numbers", "basics"], ["things", "verbs", "numbers"]],
         "hindi": [["numbers", "colors"], ["things", "family"]]}

CARD_VOCAB = ("Learn vocabulary in small groups (numbers, colours, family, food, days, animals, everyday things). Say each word aloud, write it down, and use it in a short sentence. "
              "Spaced repetition works: review new words after a day, a few days and a week. Accents are checked leniently here, but practise them — they change meaning.")


def _vocab_question(rng, lang, t):
    tier = TIERS[lang][0 if t < .5 else 1]
    cat = rng.choice(tier)
    words = VOCAB[lang][cat]
    en, tgt, rom = rng.choice(words)
    name = NAMES[lang]
    asian = lang in ("mandarin", "hindi")
    forms = ["to"] if not asian else ["to", "to_rom", "from_rom"]
    form = rng.choice(forms + ["from"])
    if form == "to":
        q = Q("cloze", f"How do you say '{en}' in {name}{' (write it in ' + ('Chinese characters' if lang == 'mandarin' else 'Devanagari') + ')' if asian else ''}?", tgt, f"{en} = {tgt}" + (f" ({rom})" if rom else ""), exact=asian)
        return q
    if form == "to_rom":
        return Q("cloze", f"Give the {'pinyin' if lang == 'mandarin' else 'romanisation'} of {tgt}.", rom, f"{tgt} = {rom}")
    if form == "from_rom":
        return Q("cloze", f"What does '{rom}' mean in English? (one word or short phrase)", en, f"{rom} = {en}")
    if rng.random() < .5 and t < .6:
        pool = [w[0] for c in VOCAB[lang].values() for w in c if w[0] != en]
        return mcq(rng, f"What does '{tgt}' mean?", en, rng.sample(pool, 3), f"{tgt} = {en}")
    return Q("cloze", f"What does '{tgt}' mean in English? (one word)", en, f"{tgt} = {en}")


def _register_vocab(lang):
    @family(f"lang.vocab.{lang}", f"{NAMES[lang]} vocabulary", {lang: (1, 12)}, CARD_VOCAB, "Mixing up similar-looking words; forgetting accents and tones.")
    def make(rng, t, lang=lang):
        return _vocab_question(rng, lang, t)


for _lang in VOCAB:
    _register_vocab(_lang)


# ----------------------------------------------------------------- numbers arithmetic in the language

NUM_WORDS = {
    "spanish": dict(zip(range(1, 21), VOCAB["spanish"]["numbers"])), "french": dict(zip(range(1, 21), VOCAB["french"]["numbers"])),
    "german": dict(zip(range(1, 21), VOCAB["german"]["numbers"])), "mandarin": dict(zip(range(1, 11), VOCAB["mandarin"]["numbers"])),
    "hindi": dict(zip([1, 2, 3, 4, 6, 7, 8, 9, 10], VOCAB["hindi"]["numbers"])),
}
PLUS = {"spanish": "más", "french": "plus", "german": "plus", "mandarin": "加", "hindi": "और"}


def _register_numbers(lang):
    nums = NUM_WORDS[lang]

    @family(f"lang.numbers.{lang}", f"{NAMES[lang]} numbers and sums", {lang: (3, 12)},
            "Practise reading number words and doing small sums entirely in the target language. Answer with the number WORD (or character) in that language.", "Guessing from similar-looking numbers (six / seven).")
    def make(rng, t, nums=nums, lang=lang):
        keys = sorted(nums)
        for _ in range(100):
            a, b = rng.choice(keys), rng.choice(keys)
            if a + b in nums:
                break
        else:
            a, b = 1, 2
        text = f"{nums[a][1]} {PLUS[lang]} {nums[b][1]}"
        return Q("cloze", f"Do this sum and answer with the number word in {NAMES[lang]}: {text} = ?", nums[a + b][1], f"{a} + {b} = {a + b}", exact=lang in ("mandarin", "hindi"))


for _lang in NUM_WORDS:
    _register_numbers(_lang)


# ----------------------------------------------------------------- regular conjugation

ES_VERBS = {"ar": "hablar caminar estudiar trabajar bailar cantar comprar cocinar necesitar".split(), "er": "comer beber aprender correr leer vender".split(),
            "ir": "vivir escribir abrir".split()}
ES_PRON = ["yo", "tú", "él / ella", "nosotros", "vosotros", "ellos / ellas"]
ES_PRES = {"ar": ["o", "as", "a", "amos", "áis", "an"], "er": ["o", "es", "e", "emos", "éis", "en"], "ir": ["o", "es", "e", "imos", "ís", "en"]}
ES_PRET = {"ar": ["é", "aste", "ó", "amos", "asteis", "aron"], "er": ["í", "iste", "ió", "imos", "isteis", "ieron"], "ir": ["í", "iste", "ió", "imos", "isteis", "ieron"]}
ES_SUBJ = {"ar": ["e", "es", "e", "emos", "éis", "en"], "er": ["a", "as", "a", "amos", "áis", "an"], "ir": ["a", "as", "a", "amos", "áis", "an"]}
ES_FUT = ["é", "ás", "á", "emos", "éis", "án"]

FR_VERBS = {"er": "parler habiter marcher chanter danser travailler étudier aimer regarder écouter donner".split(), "ir": "finir choisir réussir grandir".split(),
            "re": "vendre attendre entendre répondre perdre".split()}
FR_PRON = ["je", "tu", "il / elle", "nous", "vous", "ils / elles"]
FR_PRES = {"er": ["e", "es", "e", "ons", "ez", "ent"], "ir": ["is", "is", "it", "issons", "issez", "issent"], "re": ["s", "s", "", "ons", "ez", "ent"]}
FR_FUT = ["ai", "as", "a", "ons", "ez", "ont"]
FR_PART = {"er": "é", "ir": "i", "re": "u"}

DE_VERBS = "spielen machen lernen wohnen kaufen arbeiten lieben hören sagen tanzen kochen fragen glauben brauchen lachen".split()
DE_PRON = ["ich", "du", "er / sie", "wir", "ihr", "sie (plural)"]


def _de_present(verb):
    stem = verb[:-2] if verb.endswith("en") else verb[:-1]
    e = "e" if stem.endswith(("t", "d")) else ""
    du = "t" if stem.endswith(("s", "ß", "z", "x")) else f"{e}st"
    return [stem + "e", stem + du, stem + e + "t", stem + "en", stem + e + "t", stem + "en"]


def _de_participle(verb):
    stem = verb[:-2]
    return "ge" + stem + ("et" if stem.endswith(("t", "d")) else "t")


CARD_CONJ = ("A verb ending tells you WHO does the action and WHEN. Take the infinitive, remove its ending (-ar/-er/-ir in Spanish, -er/-ir/-re in French, -en in German) to get the stem, "
             "then add the ending for the person and tense. Regular verbs follow the pattern without exceptions.")


@family("lang.conj.spanish", "Spanish verbs", {"spanish": (4, 12)}, CARD_CONJ, "Using the wrong ending family (-ar vs -er vs -ir).")
def conj_es(rng, t):
    kind = rng.choice(["ar", "er", "ir"])
    verb = rng.choice(ES_VERBS[kind])
    i = rng.randrange(6)
    tense = "present" if t < .35 else rng.choice(["present", "preterite", "future"] + (["subjunctive"] if t > .7 else []))
    if tense == "preterite" and verb == "leer":
        verb = "comer"
    stem = verb[:-2]
    if tense == "present":
        form, label = stem + ES_PRES[kind][i], "present tense"
    elif tense == "preterite":
        form, label = stem + ES_PRET[kind][i], "preterite (simple past)"
    elif tense == "future":
        form, label = verb + ES_FUT[i], "simple future"
    else:
        form, label = stem + ES_SUBJ[kind][i], "present subjunctive (use the stem of the 'yo' present form: hablo → habl-)"
    return Q("cloze", f"Conjugate the Spanish verb '{verb}' for '{ES_PRON[i]}' in the {label}.", form, f"{verb}: {stem}- + {form[len(stem):] if form.startswith(stem) else form}")


@family("lang.conj.french", "French verbs", {"french": (4, 12)}, CARD_CONJ, "Forgetting that -ir verbs like finir add -iss- in the plural.")
def conj_fr(rng, t):
    kind = rng.choice(["er", "ir", "re"])
    verb = rng.choice(FR_VERBS[kind])
    i = rng.randrange(6)
    tense = "present" if t < .4 else rng.choice(["present", "future", "participle"])
    stem = verb[:-2] if kind in ("er", "ir") else verb[:-2]
    if tense == "present":
        form = stem + FR_PRES[kind][i]
        shown = FR_PRON[i]
        if i == 0 and verb[0] in "aeiouéhè":
            shown = "j'"
        return Q("cloze", f"Conjugate the French verb '{verb}' for '{shown}' in the present tense (give the verb form only).", form, f"{stem}- + {FR_PRES[kind][i] or '(nothing)'}")
    if tense == "future":
        base = verb[:-1] if kind == "re" else verb
        return Q("cloze", f"Conjugate the French verb '{verb}' for '{FR_PRON[i]}' in the simple future (verb form only).", base + FR_FUT[i], f"infinitive{' without final e' if kind == 're' else ''} + {FR_FUT[i]}")
    return Q("cloze", f"What is the past participle of the French verb '{verb}' (used in the passé composé with avoir)?", stem + FR_PART[kind], f"{stem}- + {FR_PART[kind]}")


@family("lang.conj.german", "German verbs", {"german": (4, 12)}, CARD_CONJ + " German stems ending in -t or -d insert an extra e (du arbeitest); stems ending in s, ß, z or x take just -t after du (du tanzt).",
        "Forgetting the extra e in arbeiten.")
def conj_de(rng, t):
    verb = rng.choice(DE_VERBS)
    i = rng.randrange(6)
    if t < .5 or rng.random() < .6:
        return Q("cloze", f"Conjugate the German verb '{verb}' for '{DE_PRON[i]}' in the present tense (verb form only).", _de_present(verb)[i], "stem + ending")
    return Q("cloze", f"What is the past participle (Perfekt) of the regular German verb '{verb}'? (ge- + stem + -t)", _de_participle(verb), "ge + stem + (e)t")


GENDER = {
    "spanish": {"el": "libro perro gato caballo pájaro pan queso pollo arroz huevo pescado día trabajo tiempo", "la": "casa escuela ciudad leche manzana vaca noche"},
    "french": {"le": "livre chien chat cheval oiseau pain lait fromage poulet riz poisson jour travail temps", "la": "maison école ville vache pomme nuit"},
    "german": {"der": "Freund Tag Hund Vogel Apfel Käse Reis Fisch Vater Bruder Sohn Onkel", "die": "Schule Stadt Arbeit Zeit Nacht Katze Kuh Milch Mutter Schwester Tochter Tante", "das": "Haus Buch Pferd Brot Wasser Ei"},
}


def _register_gender(lang):
    table = {art: words.split() for art, words in GENDER[lang].items()}
    arts = list(table)

    @family(f"lang.gender.{lang}", f"{NAMES[lang]} noun genders", {lang: (2, 9)},
            "Every noun has a grammatical gender that decides its article (Spanish el/la, French le/la, German der/die/das). Learn each new noun together with its article, not on its own.",
            "Guessing gender from meaning instead of memorising the article.")
    def make(rng, t, table=table, arts=arts, lang=lang):
        art = rng.choice(arts)
        word = rng.choice(table[art])
        return mcq(rng, f"Which definite article goes with the {NAMES[lang]} noun '{word}'?", art, [a for a in arts if a != art] + ["los" if lang == "spanish" else "les" if lang == "french" else "den"], f"{art} {word}")


for _lang in GENDER:
    _register_gender(_lang)
