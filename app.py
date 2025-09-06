# app.py
from flask import Flask, request, jsonify
import re
import math
import os

app = Flask(__name__)
# keep Unicode characters (Chinese, German umlauts) in JSON responses
app.config["JSON_AS_ASCII"] = False


# =========================
#        DUOLINGO SORT
# =========================
# ---- Roman numerals ----
_ROM = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROM_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def roman_to_int(s: str) -> int:
    if not s:
        raise ValueError("not roman")
    s = s.strip().upper()
    if not _ROM_RE.match(s):
        raise ValueError("not roman")
    total = 0
    for i, ch in enumerate(s):
        v = _ROM[ch]
        if i + 1 < len(s) and _ROM[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


# ---- English words ----
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}
# only scales needed by the challenge
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}


def english_to_int(s: str) -> int:
    if not s:
        raise ValueError("not english")
    t = s.lower().replace("-", " ").split()
    total, cur = 0, 0
    for w in t:
        if w == "and":
            continue
        if w in _UNITS:
            cur += _UNITS[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred":
            if cur == 0:
                cur = 1
            cur *= 100
        elif w in ("thousand", "million", "billion"):
            if cur == 0:
                cur = 1
            total += cur * _SCALES[w]
            cur = 0
        else:
            raise ValueError("not english")
    return total + cur


# ---- Chinese numerals (Trad/Simp) ----
_CH_DIG = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4,
           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_BIG_UNITS = {"萬": 10_000, "万": 10_000, "億": 100_000_000, "亿": 100_000_000, "兆": 1_000_000_000_000}


def is_chinese_num(s: str) -> bool:
    return any(c in _CH_DIG or c in _SMALL_UNITS or c in _BIG_UNITS for c in s)


def chinese_to_int(s: str) -> int:
    if not s:
        raise ValueError("not chinese")
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
    # default ambiguous forms to Traditional for the tie-break order
    if any(c in ("萬", "億", "兩") for c in s):
        return "trad_chinese"
    if any(c in ("万", "亿", "两") for c in s):
        return "simp_chinese"
    return "trad_chinese"


# ---- German words ----
def _norm_de(s: str) -> str:
    s = s.lower().replace(" ", "").replace("-", "")
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    return s


_DE_UNITS = {
    "null": 0, "eins": 1, "ein": 1, "eine": 1, "zwei": 2, "drei": 3, "vier": 4,
    "fuenf": 5, "funf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9,
    "zehn": 10, "elf": 11, "zwoelf": 12, "zwoelfe": 12, "zwoelfer": 12,
    "dreizehn": 13, "vierzehn": 14, "fuenfzehn": 15, "funfzehn": 15,
    "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19
}
_DE_TENS = {
    "zwanzig": 20, "dreissig": 30, "dreiszig": 30, "dreißig": 30,
    "vierzig": 40, "fuenfzig": 50, "funfzig": 50, "sechzig": 60,
    "siebzig": 70, "achtzig": 80, "neunzig": 90
}


def german_to_int(s: str) -> int:
    if not s:
        raise ValueError("not german")
    s = _norm_de(s)
    # big scales
    if "tausend" in s:
        i = s.index("tausend")
        left, right = s[:i], s[i + 7:]
        left_val = german_to_int(left) if left else 1
        return left_val * 1000 + (german_to_int(right) if right else 0)
    if "hundert" in s:
        i = s.index("hundert")
        left, right = s[:i], s[i + 7:]
        left_val = german_to_int(left) if left else 1
        return left_val * 100 + (german_to_int(right) if right else 0)
    if s in _DE_UNITS:
        return _DE_UNITS[s]
    if s in _DE_TENS:
        return _DE_TENS[s]
    # einundzwanzig pattern: units + 'und' + tens
    if "und" in s:
        i = s.rfind("und")
        units = s[:i]
        tens = s[i + 3:]
        if tens in _DE_TENS:
            u = german_to_int(units)
            return u + _DE_TENS[tens]
    raise ValueError("not german")


# ---- Detection & parsing ----
LANG_ORDER = {
    "roman": 0,
    "english": 1,
    "trad_chinese": 2,
    "simp_chinese": 3,
    "german": 4,
    "arabic": 5,
}


def detect_and_parse(s: str):
    if s is None:
        raise ValueError("empty")
    s_stripped = str(s).strip()
    # Arabic digits
    if s_stripped.isdigit():
        return int(s_stripped), "arabic"

    # Roman (accept lowercase -> uppercase for validation)
    if re.fullmatch(r"[ivxlcdmIVXLCDM]+", s_stripped):
        return roman_to_int(s_stripped), "roman"

    # Chinese
    if is_chinese_num(s_stripped):
        val = chinese_to_int(s_stripped)
        return val, chinese_variant(s_stripped)

    # English
    try:
        return english_to_int(s_stripped), "english"
    except Exception:
        pass

    # German
    try:
        return german_to_int(s_stripped), "german"
    except Exception:
        pass

    raise ValueError(f"Unrecognized numeral: {s_stripped}")


# ---- Health & home ----
@app.get("/healthz")
def healthz():
    return "ok", 200


@app.get("/")
def home():
    return "UBS GCC service. Endpoints: /duolingo-sort, /trading-bot, /healthz", 200


# ---- Main Duolingo API ----
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
        vals = []
        try:
            for s in arr:
                s = str(s).strip()
                if s.isdigit():
                    vals.append(int(s))
                else:
                    vals.append(roman_to_int(s))
        except Exception as e:
            return jsonify({"error": f"Part ONE parse error: {e}"}), 400
        vals.sort()
        return jsonify({"sortedList": [str(v) for v in vals]}), 200

    if part == "TWO":
        items = []
        for idx, s in enumerate(arr):
            s0 = str(s).strip()
            try:
                v, lang = detect_and_parse(s0)
            except Exception as e:
                return jsonify({"error": f"Cannot parse '{s0}': {e}"}), 400
            items.append((v, LANG_ORDER[lang], idx, s0))
        items.sort(key=lambda t: (t[0], t[1], t[2]))
        return jsonify({"sortedList": [t[3] for t in items]}), 200

    return jsonify({"error": "part must be 'ONE' or 'TWO'"}), 400


# =========================
#        TRADING BOT
# =========================
def _to_float(x, default=None):
    try:
        f = float(x)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return default


def _sort_by_ts(lst):
    return sorted(lst or [], key=lambda c: int(c.get("timestamp", 0)))


def _atr(candles):
    """Average True Range over provided candles; protects against missing data."""
    cs = _sort_by_ts(candles)
    if not cs:
        return 1e-9
    prev_close = _to_float(cs[0].get("close"), _to_float(cs[0].get("open"), 0.0))
    trs = []
    for c in cs:
        h = _to_float(c.get("high"), prev_close)
        l = _to_float(c.get("low"), prev_close)
        cl = _to_float(c.get("close"), prev_close)
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        tr = tr if math.isfinite(tr) else 0.0
        trs.append(tr)
        prev_close = cl
    atr = sum(trs) / max(len(trs), 1)
    return atr if atr > 1e-9 else 1e-9


def _entry_price(obs):
    """Entry is the close of the first observation candle."""
    if not obs:
        return None
    first = _sort_by_ts(obs)[0]
    return _to_float(first.get("close"))


def _obs_extremes(obs):
    """Return (maxHigh, minLow, lastClose) from observation candles."""
    if not obs:
        return (None, None, None)
    cs = _sort_by_ts(obs)
    highs = [_to_float(c.get("high")) for c in cs]
    lows = [_to_float(c.get("low")) for c in cs]
    highs = [h for h in highs if h is not None]
    lows = [l for l in lows if l is not None]
    last_close = _to_float(cs[-1].get("close"))
    return (max(highs) if highs else None,
            min(lows) if lows else None,
            last_close)


def _last_prev_close(prevs):
    if not prevs:
        return None
    cs = _sort_by_ts(prevs)
    return _to_float(cs[-1].get("close") or cs[-1].get("open"))


def _signal_for_event(ev):
    """Vol-normalized score using immediate post-entry behavior."""
    prevs = ev.get("previous_candles") or []
    obs = ev.get("observation_candles") or []
    entry = _entry_price(obs)
    if entry is None:
        return (0.0, False)
    atr = _atr(prevs)
    maxH, minL, lastClose = _obs_extremes(obs)
    lastPrevClose = _last_prev_close(prevs)

    m1 = 0.0 if lastClose is None else (lastClose - entry) / atr  # momentum after entry
    m2 = 0.0
    if maxH is not None and minL is not None:
        m2 = ((maxH - entry) - (entry - minL)) / atr              # skew of extremes
    m3 = 0.0 if lastPrevClose is None else (entry - lastPrevClose) / atr  # gap

    s = 0.6 * m1 + 0.3 * m2 + 0.1 * m3
    s = s if math.isfinite(s) else 0.0
    return (s, True)


def _decide_from_score(s, entry, maxH, minL, ev_id):
    if s > 0:
        return "LONG"
    if s < 0:
        return "SHORT"
    # tie-break with skew; finally id parity deterministic
    skew = 0.0
    if entry is not None and maxH is not None and minL is not None:
        skew = (maxH - entry) - (entry - minL)
    if skew > 0:
        return "LONG"
    if skew < 0:
        return "SHORT"
    return "LONG" if (int(ev_id) % 2 == 0) else "SHORT"


@app.post("/trading-bot")
def trading_bot():
    """
    Input: JSON array of events. Each event has 'id', 'previous_candles', 'observation_candles'.
    Output: EXACTLY 50 items: [{"id": <id>, "decision": "LONG"|"SHORT"}, ...]
    """
    data = request.get_json(silent=True)
    if not isinstance(data, list):
        return jsonify({"error": "Body must be a JSON array"}), 400

    scored = []
    for ev in data:
        try:
            ev_id = ev.get("id")
            s, ok = _signal_for_event(ev)
            if not ok or ev_id is None:
                continue
            entry = _entry_price(ev.get("observation_candles") or [])
            maxH, minL, _ = _obs_extremes(ev.get("observation_candles") or [])
            scored.append({
                "id": ev_id,
                "score": float(s),
                "abs_score": float(abs(s)),
                "entry": entry, "maxH": maxH, "minL": minL
            })
        except Exception:
            # skip malformed records
            continue

    # pick strongest 50 by absolute score (then by id for determinism)
    scored.sort(key=lambda r: (r["abs_score"], r["id"]), reverse=True)
    top = scored[:50]

    # backfill deterministically if not enough valid events
    if len(top) < 50:
        seen = {r["id"] for r in top}
        for ev in data:
            if len(top) >= 50:
                break
            ev_id = ev.get("id")
            if ev_id is None or ev_id in seen:
                continue
            top.append({"id": ev_id, "score": 0.0, "abs_score": 0.0,
                        "entry": None, "maxH": None, "minL": None})
            seen.add(ev_id)

    out = []
    for r in top:
        decision = _decide_from_score(r["score"], r["entry"], r["maxH"], r["minL"], r["id"])
        out.append({"id": r["id"], "decision": decision})

    # ensure deterministic order: strongest first, then id
    abs_map = {r["id"]: r["abs_score"] for r in top}
    out.sort(key=lambda x: (abs_map.get(x["id"], 0.0), x["id"]), reverse=True)
    return jsonify(out), 200


# =========================
#        ENTRY POINT
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))

