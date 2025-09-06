from flask import Flask, request, jsonify

app = Flask(__name__)

# -------------------- Roman --------------------
_ROM = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
import re
_ROM_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")

def roman_to_int(s: str) -> int:
    if not s or not _ROM_RE.match(s):
        raise ValueError("not roman")
    total = 0
    for i, ch in enumerate(s):
        v = _ROM[ch]
        if i + 1 < len(s) and _ROM[s[i+1]] > v:
            total -= v
        else:
            total += v
    return total

# -------------------- English --------------------
_UNITS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
    "seventeen":17,"eighteen":18,"nineteen":19
}
_TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
_SCALES = {"hundred":100,"thousand":1000,"million":1_000_000,"billion":1_000_000_000}

def english_to_int(s: str) -> int:
    t = s.lower().replace("-", " ").split()
    if not t: raise ValueError("not english")
    total, cur = 0, 0
    for w in t:
        if w == "and":  # optional
            continue
        if w in _UNITS:
            cur += _UNITS[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred":
            if cur == 0: cur = 1
            cur *= 100
        elif w in ("thousand", "million", "billion"):
            if cur == 0: cur = 1
            total += cur * _SCALES[w]
            cur = 0
        else:
            raise ValueError("not english")
    return total + cur

# -------------------- Chinese (Trad/Simp) --------------------
_CH_DIG = {"零":0,"〇":0,"一":1,"二":2,"兩":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
_SMALL_UNITS = {"十":10,"百":100,"千":1000}
_BIG_UNITS = {"萬":10_000,"万":10_000,"億":100_000_000,"亿":100_000_000,"兆":1_000_000_000_000}

def is_chinese_num(s: str) -> bool:
    return any(c in _CH_DIG or c in _SMALL_UNITS or c in _BIG_UNITS for c in s)

def chinese_to_int(s: str) -> int:
    total = 0
    section = 0
    number = 0
    for ch in s:
        if ch in _CH_DIG:
            number = _CH_DIG[ch]
        elif ch in _SMALL_UNITS:
            unit = _SMALL_UNITS[ch]
            section += (number if number != 0 else 1) * unit
            number = 0
        elif ch in _BIG_UNITS:
            unit = _BIG_UNITS[ch]
            section += number
            total += section * unit
            section = 0
            number = 0
        else:
            raise ValueError("not chinese")
    section += number
    return total + section

def chinese_variant(s: str) -> str:
    # default ambiguous forms to Traditional (higher tie priority)
    if any(c in ("萬","億","兩") for c in s): return "trad"
    if any(c in ("万","亿","两") for c in s): return "simp"
    return "trad"

# -------------------- German --------------------
def _norm_de(s: str) -> str:
    s = s.lower().replace(" ", "").replace("-", "")
    s = s.replace("ä","ae").replace("ö","oe").replace("ü","ue").replace("ß","ss")
    return s

_DE_UNITS = {
    "null":0,"eins":1,"ein":1,"eine":1,"zwei":2,"drei":3,"vier":4,"fuenf":5,"funf":5,
    "sechs":6,"sieben":7,"acht":8,"neun":9,"zehn":10,"elf":11,"zwoelf":12,"zwoelfe":12,
    "dreizehn":13,"vierzehn":14,"fuenfzehn":15,"funfzehn":15,"sechzehn":16,"siebzehn":17,
    "achtzehn":18,"neunzehn":19
}
_DE_TENS = {
    "zwanzig":20,"dreissig":30,"dreiszig":30,"dreißig":30,  # allow both
    "vierzig":40,"fuenfzig":50,"funfzig":50,"sechzig":60,"siebzig":70,"achtzig":80,"neunzig":90
}

def german_to_int(s: str) -> int:
    s = _norm_de(s)
    if not s: raise ValueError("not german")
    # big scales (tausend)
    if "tausend" in s:
        i = s.index("tausend")
        left, right = s[:i], s[i+7:]
        left_val = german_to_int(left) if left else 1
        return left_val * 1000 + (german_to_int(right) if right else 0)
    if "hundert" in s:
        i = s.index("hundert")
        left, right = s[:i], s[i+7:]
        left_val = german_to_int(left) if left else 1
        return left_val * 100 + (german_to_int(right) if right else 0)
    if s in _DE_UNITS:
        return _DE_UNITS[s]
    if s in _DE_TENS:
        return _DE_TENS[s]
    if "und" in s:
        # unit + 'und' + tens (einundzwanzig, siebenundachtzig)
        i = s.rfind("und")
        units = s[:i]
        tens = s[i+3:]
        u = german_to_int(units)
        t = _DE_TENS.get(tens, None)
        if t is None:
            t = german_to_int(tens)
        return u + t
    raise ValueError("not german")

# -------------------- Detection & Parsing --------------------
LANG_ORDER = {
    "roman":0, "english":1, "trad_chinese":2, "simp_chinese":3, "german":4, "arabic":5
}

def detect_and_parse(s: str):
    s_stripped = s.strip()
    # Arabic numerals
    if s_stripped.isdigit():
        return int(s_stripped), "arabic"
    # Roman (strict upper case form)
    if re.fullmatch(r"[IVXLCDM]+", s_stripped):
        try:
            return roman_to_int(s_stripped), "roman"
        except Exception:
            pass
    # Chinese
    if is_chinese_num(s_stripped):
        val = chinese_to_int(s_stripped)
        variant = chinese_variant(s_stripped)
        return val, ("trad_chinese" if variant=="trad" else "simp_chinese")
    # English
    try:
        val = english_to_int(s_stripped)
        return val, "english"
    except Exception:
        pass
    # German
    try:
        val = german_to_int(s_stripped)
        return val, "german"
    except Exception:
        pass
    raise ValueError(f"Unrecognized numeral: {s}")

# -------------------- API --------------------
@app.get("/healthz")
def healthz():
    return "ok", 200

@app.post("/duolingo-sort")
def duolingo_sort():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    part = str(data.get("part", "")).upper()
    arr = (data.get("challengeInput") or {}).get("unsortedList", [])
    if not isinstance(arr, list):
        return jsonify({"error": "challengeInput.unsortedList must be a list"}), 400

    if part == "ONE":
        try:
            vals = [int(x) if str(x).strip().isdigit() else roman_to_int(str(x).strip()) for x in arr]
        except Exception as e:
            return jsonify({"error": f"Part ONE parse error: {e}"}), 400
        return jsonify({"sortedList": [str(v) for v in sorted(vals)]}), 200

    if part == "TWO":
        items = []
        for idx, s in enumerate(arr):
            s = str(s).strip()
            try:
                v, lang = detect_and_parse(s)       # <-- if anything fails, we catch it
                items.append((v, LANG_ORDER[lang], idx, s))
            except Exception as e:
                return jsonify({"error": f"Cannot parse '{s}': {e}"}), 400
        items.sort(key=lambda t: (t[0], t[1], t[2]))
        return jsonify({"sortedList": [t[3] for t in items]}), 200

    return jsonify({"error": "part must be 'ONE' or 'TWO'"}), 400

@app.get("/")
def home():
    return "UBS GCC duolingo-sort service. Try GET /healthz or POST /duolingo-sort.", 200

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
