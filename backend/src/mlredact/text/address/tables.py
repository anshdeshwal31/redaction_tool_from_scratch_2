"""Lexicons and tables of the address grammar: street types, designators, building and property words, box kinds,
states, occupation and organisation words, address labels, connecting words and the structure roles."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

from mlredact.text.address.model import Role

# ----------------------------------------------------------------------------------------------------- states
SPELLED_STATES: Final[Mapping[tuple[str, ...], str]] = {
    ("new", "south", "wales"): "NSW",
    ("australian", "capital", "territory"): "ACT",
    ("south", "australia"): "SA",
    ("western", "australia"): "WA",
    ("northern", "territory"): "NT",
    ("victoria",): "VIC",
    ("queensland",): "QLD",
    ("tasmania",): "TAS",
}
TITLE_ABBR_STATES: Final[Mapping[str, str]] = {"Vic": "VIC", "Qld": "QLD", "Tas": "TAS"}
DOTTED_STATES: Final[Mapping[str, str]] = {
    "N.S.W": "NSW",
    "S.A": "SA",
    "W.A": "WA",
    "N.T": "NT",
    "A.C.T": "ACT",
    "V.I.C": "VIC",
    "Q.L.D": "QLD",
    "T.A.S": "TAS",
}
# OCR reads of a state in upper case ('N5W'): digit-like characters mapped back to letters.
OCR_STATE_MAP: Final = str.maketrans({"5": "S", "0": "O", "1": "I", "8": "B", "4": "A", "$": "S"})

COUNTRY: Final = frozenset({"australia"})
OTHER_COUNTRIES: Final = frozenset(
    {
        ("new", "zealand"),
        ("united", "kingdom"),
        ("england",),
        ("scotland",),
        ("ireland",),
        ("wales",),
        ("india",),
        ("china",),
        ("vietnam",),
        ("philippines",),
        ("fiji",),
        ("lebanon",),
        ("greece",),
        ("italy",),
        ("germany",),
        ("canada",),
        ("usa",),
        ("united", "states"),
        ("south", "africa"),
        ("malaysia",),
        ("singapore",),
        ("indonesia",),
        ("japan",),
        ("korea",),
        ("sri", "lanka"),
        ("pakistan",),
        ("iraq",),
        ("iran",),
        ("afghanistan",),
        ("sudan",),
        ("egypt",),
        ("turkey",),
        ("croatia",),
        ("serbia",),
        ("poland",),
        ("netherlands",),
        ("france",),
        ("spain",),
        ("portugal",),
        ("nepal",),
        ("bangladesh",),
        ("thailand",),
        ("cambodia",),
        ("samoa",),
        ("tonga",),
        ("chile",),
        ("brazil",),
        ("argentina",),
        ("mexico",),
        ("russia",),
        ("ukraine",),
        ("hong", "kong"),
        ("taiwan",),
    }
)

# ----------------------------------------------------------------------------------------------------- street types
# Street types of AS 4590 and AS/NZS 4819: the canonical type (the full spelling, upper case) and its written forms,
# the full spelling first.  Plural forms ('Streets') serve corner addresses.
_TYPE_FORMS: Final[Mapping[str, str]] = {
    "STREET": "street st str",
    "ROAD": "road rd",
    "AVENUE": "avenue ave av avn",
    "DRIVE": "drive dr drv",
    "COURT": "court ct crt",
    "PLACE": "place pl",
    "CRESCENT": "crescent cres cr crs",
    "LANE": "lane ln",
    "WAY": "way",
    "PARADE": "parade pde",
    "TERRACE": "terrace tce terr",
    "BOULEVARD": "boulevard boulevarde blvd bvd",
    "HIGHWAY": "highway hwy",
    "CLOSE": "close cl",
    "CIRCUIT": "circuit cct",
    "GROVE": "grove gr gve",
    "SQUARE": "square sq",
    "ESPLANADE": "esplanade esp",
    "TRACK": "track trk",
    "TRAIL": "trail trl",
    "RISE": "rise",
    "ROW": "row",
    "MEWS": "mews",
    "WALK": "walk",
    "LOOP": "loop",
    "LINK": "link",
    "VISTA": "vista vsta",
    "VIEW": "view",
    "GLADE": "glade",
    "GARDENS": "gardens gdns gdn",
    "HEIGHTS": "heights hts",
    "RIDGE": "ridge rdge",
    "PROMENADE": "promenade prom",
    "QUAY": "quay qy",
    "ALLEY": "alley",
    "ARCADE": "arcade",
    "BEND": "bend",
    "BRACE": "brace",
    "CHASE": "chase",
    "CONCOURSE": "concourse",
    "CORSO": "corso",
    "COVE": "cove",
    "CREST": "crest crst",
    "CROSS": "cross crss",
    "DELL": "dell",
    "ENTRANCE": "entrance",
    "GATE": "gate gte",
    "GRANGE": "grange",
    "GREEN": "green grn",
    "HILL": "hill",
    "JUNCTION": "junction jnc jct",
    "LANDING": "landing",
    "MALL": "mall",
    "MEANDER": "meander",
    "NOOK": "nook",
    "OUTLOOK": "outlook",
    "PASS": "pass",
    "PATH": "path",
    "PATHWAY": "pathway pway",
    "POINT": "point pnt",
    "RESERVE": "reserve",
    "RETREAT": "retreat rtt",
    "ROUND": "round rnd",
    "STRIP": "strip",
    "TURN": "turn",
    "VALE": "vale",
    "WHARF": "wharf whrf",
    "CIRCLE": "circle",
    "PLAZA": "plaza plza",
    "PARKWAY": "parkway pkwy pwy",
    "FREEWAY": "freeway fwy",
    "MOTORWAY": "motorway mwy",
    # more AS 4590 types
    "BYPASS": "bypass bypa byp",
    "BYWAY": "byway bywy",
    "CAUSEWAY": "causeway cswy",
    "CIRCUS": "circus crcs ccs",
    "DRIVEWAY": "driveway drwy",
    "EXPRESSWAY": "expressway expy",
    "FAIRWAY": "fairway fawy",
    "GATEWAY": "gateway gwy",
    "GLEN": "glen gln",
    "LANEWAY": "laneway lnwy",
    "QUADRANT": "quadrant qdrt",
    "TOLLWAY": "tollway tlwy",
    "WALKWAY": "walkway wway",
    "WYND": "wynd",
}
TYPE_FORMS: Final[Mapping[str, tuple[str, ...]]] = {canon: tuple(forms.split()) for canon, forms in _TYPE_FORMS.items()}


def _type_canon() -> dict[str, str]:
    out: dict[str, str] = {}
    for canon, forms in TYPE_FORMS.items():
        for f in forms:
            out[f] = canon
    for canon in TYPE_FORMS:
        plural = canon.lower() + ("es" if canon.endswith(("S", "X", "CH")) else "s")
        out.setdefault(plural, canon)  # 'streets', 'roads' (corner forms)
    return out


TYPE_CANON: Final[Mapping[str, str]] = _type_canon()
PLURAL_TYPES: Final = frozenset(k for k, v in TYPE_CANON.items() if k.endswith("s") and k[:-1] in TYPE_CANON)
STREET_TYPES: Final = tuple(sorted(TYPE_FORMS))  # the canonical types, in a stable order


def canonical_street_type(word: str) -> str | None:
    """The canonical street type of a written type ('St', 'Streets', 'CRES' -> 'STREET', 'STREET', 'CRESCENT'), or
    None when the word is not a street type."""
    return TYPE_CANON.get(word.strip(" ,.;:()").lower())


def is_abbreviated_type(word: str) -> bool:
    """True when a written street type is an abbreviation ('St', 'Rd', 'Cres'); false for the full spelling, its plural
    and its variant spellings ('Streets', 'Boulevarde')."""
    canon = canonical_street_type(word)
    return canon is not None and not word.strip(" ,.;:()").lower().startswith(canon.lower())


# Types that are never a street line on their own without a house number ('Local Court', 'Penrith Plaza').
NUMBERLESS_BANNED_TYPES: Final = frozenset({"COURT", "PLAZA", "MALL", "GARDENS"})

# Australia Post's street-type suffixes ('Smith Street North').
SUFFIX_WORDS: Final = frozenset(
    {"north", "south", "east", "west", "nth", "sth", "upper", "lower", "central", "extension"}
)
SUFFIX_CODES: Final = frozenset({"N", "S", "E", "W", "NE", "NW", "SE", "SW"})
SUFFIX_CANON: Final[Mapping[str, str]] = {
    "north": "N",
    "nth": "N",
    "n": "N",
    "south": "S",
    "sth": "S",
    "s": "S",
    "east": "E",
    "e": "E",
    "west": "W",
    "w": "W",
    "ne": "NE",
    "nw": "NW",
    "se": "SE",
    "sw": "SW",
    "upper": "UP",
    "lower": "LR",
    "central": "CN",
    "extension": "EX",
}

# ----------------------------------------------------------------------------------------------------- designators
# Australia Post's flat/unit and floor/level types ('House' stays a building word).  Value: 'unit', 'level', 'floor'.
DESIGNATORS: Final[Mapping[str, str]] = {
    "unit": "unit",
    "u": "unit",
    "apt": "unit",
    "apartment": "unit",
    "flat": "unit",
    "suite": "unit",
    "ste": "unit",
    "shop": "unit",
    "lot": "unit",
    "villa": "unit",
    "office": "unit",
    "townhouse": "unit",
    "tnhs": "unit",
    "factory": "unit",
    "fy": "unit",
    "warehouse": "unit",
    "penthouse": "unit",
    "pths": "unit",
    "studio": "unit",
    "stu": "unit",
    "room": "unit",
    "rm": "unit",
    "hse": "unit",
    "off": "unit",
    "level": "level",
    "lvl": "level",
    "floor": "floor",
    "flr": "floor",
    "fl": "floor",
}
# Floor/level types that need no number: 'Ground Floor', 'Lower Ground Floor', 'Mezzanine', 'Basement'.
NUMBERLESS_LEVELS: Final = frozenset({"ground", "mezzanine", "basement", "lg", "ug"})
LEVEL_WORDS: Final = frozenset({"floor", "flr", "fl", "level"})
# 'Building C', 'Block 2', 'Tower 2', 'Wing B': designator words before a letter or number designator.
BUILDING_DESIG: Final = frozenset({"building", "block", "tower", "wing"})
# A level-type code before a hotel-style unit number ('B02/53'): the code stays.
LEVEL_CODE: Final = re.compile(r"^(LG|UG|B|G|M)(\d{1,4}[A-Za-z]?(?:/\d{1,5}[A-Za-z]?)?)$")
ORDINAL: Final = re.compile(r"^(\d{1,2})(st|nd|rd|th)$", re.IGNORECASE)


def level_key(text: str) -> str:
    """A level number as an address key holds it: upper case, an ordinal's suffix dropped ('4th' and '4' are one
    level)."""
    core = text.strip(" ,.;:()").upper()
    m = ORDINAL.match(core)
    return m.group(1) if m else core


# ----------------------------------------------------------------------------------------------------- buildings
BUILDING_WORDS: Final = frozenset(
    """
    tower towers house chambers centre center building buildings station farm homestead complex arcade plaza
    court estate park square
    """.split()
)
# Property words that can head a 'via' line ('Kilbride Station via Berrima NSW 2577').
PROPERTY_WORDS: Final = frozenset({"station", "farm", "homestead", "estate", "park", "vineyard", "orchard", "downs"})

# ----------------------------------------------------------------------------------------------------- boxes
# Postal delivery types read as kinds in upper case and in an address context only, and those needing no number.
BOX_CODES: Final = frozenset({"rmb", "rsd", "cmb", "pmb", "rms", "ms", "cma", "cpa"})
NUMBERLESS_BOXES: Final = frozenset({("care", "po"), ("cma",), ("cpa",), ("cmb",)})
RURAL_BOXES: Final = frozenset({"rmb", "rsd", "ms", "rms"})  # road lines above them join the unit (Australia Post)
CAPITAL_CITY_PHRASE: Final = ("in", "your", "capital", "city")
BOX_PHRASES: Final[tuple[tuple[str, ...], ...]] = (
    ("post", "office", "box"),
    ("locked", "mail", "bag"),
    ("private", "mail", "bag"),
    ("p", "o", "box"),
    ("po", "box"),
    ("gpo", "box"),
    ("locked", "bag"),
    ("private", "bag"),
    ("reply", "paid"),
    ("care", "po"),
    ("pobox",),
    ("rmb",),
    ("rsd",),
    ("cmb",),
    ("pmb",),
    ("rms",),
    ("ms",),
    ("cma",),
    ("cpa",),
)
# Spellings of one box kind (amendments B and E): one canonical key, so one box gets one fake whatever its spelling.
BOX_KEY_KINDS: Final[Mapping[str, str]] = {
    "p o box": "po box",
    "pobox": "po box",
    "post office box": "po box",
    "locked mail bag": "locked bag",
    "private mail bag": "private bag",
    "pmb": "private bag",
}
BOX_FIRST_WORDS: Final = frozenset(p[0] for p in BOX_PHRASES)

# ----------------------------------------------------------------------------------------------------- word lists
# On a line ABOVE the street line these mark the organisation's name line, not a building.
ORG_WORDS: Final = frozenset(
    """
    pty ltd limited inc incorporated lawyers solicitors barristers legal insurance insurer insurers hospital
    hospitals clinic clinics centre center group associates partners practice services service department dept
    council commission tribunal university school college institute foundation trust society association club
    church union board ministry police corporation company co authority agency government rehabilitation
    physiotherapy psychology radiology imaging pathology orthopaedics orthopedics specialists consulting
    consultants surgery pharmacy dental health medicine healthcare workcover icare
    """.split()
)
ORG_NOT_BUILDING: Final = ORG_WORDS - BUILDING_WORDS
OCCUPATION_WORDS: Final = frozenset(
    """
    surgeon surgeons psychiatrist psychologist physician physiotherapist orthopaedic orthopedic consultant
    specialist practitioner nurse solicitor lawyer barrister radiologist neurologist neurosurgeon rheumatologist
    therapist counsellor chiropractor osteopath gp dentist pharmacist engineer manager director officer assessor
    labourer teacher driver registrar clinician doctor paediatrician anaesthetist cardiologist dermatologist
    endocrinologist gastroenterologist gynaecologist haematologist oncologist ophthalmologist pathologist
    urologist sonographer audiologist podiatrist dietitian coordinator adviser advisor worker secretary
    receptionist clerk
    """.split()
)
PUBLIC_BODY_WORDS: Final = frozenset(
    """
    workcover sira icare comcare centrelink medicare ndis ndia worksafe commission tribunal court police
    department council ncat aat tac ahpra ombudsman parliament government
    """.split()
)
# Words a suburb or building run never crosses (organisation, occupation and public-body words).
OCC_STOP: Final = ORG_WORDS | OCCUPATION_WORDS | PUBLIC_BODY_WORDS
TITLES: Final = frozenset(
    """
    mr mrs ms miss mx mstr master dr doctor prof professor a/prof assoc sir dame rev reverend fr father sr
    sister br brother hon judge justice magistrate commissioner coroner registrar
    """.split()
)
MONTHS: Final = frozenset(
    """
    january february march april may june july august september october november december jan feb mar apr jun
    jul aug sep sept oct nov dec
    """.split()
)
WEEKDAYS: Final = frozenset(
    {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    | {"mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun"}
)
# A number followed by one of these is a measure or a duration, never a house number.
MEASURE_WORDS: Final = frozenset(
    """
    minute minutes min mins second seconds sec secs hour hours hr hrs day days week weeks wk wks month months
    year years yr yrs metre metres meter meters m km kms cm mm kg kgs g mg ml times percent degree degrees point
    points repetitions reps step steps
    """.split()
)
# A clinical or legal-record noun right after a street type vetoes the street ('3 Local Court convictions').
VETO_NOUNS: Final = frozenset(
    """
    test tests conviction convictions order orders evidence report reports assessment assessments score scores
    scale sign signs syndrome scan scans procedure fracture injury injuries questionnaire inventory index
    criteria grade stage manoeuvre reflex view
    """.split()
)
CONNECTORS: Final = frozenset({"of", "the", "de", "la", "du", "van", "von", "des", "le"})
# Words a place kept on its own follows (birth, injury, accident or incident, declaration), and words that mark a
# residence or an address (owner ruling on question 2, option c).
PLACE_KIND_CUES: Final = frozenset(
    {"born", "birth", "birthplace", "injury", "injured", "accident", "incident", "declared", "declaration"}
)
RESIDENCE_CUES: Final = frozenset(
    """
    resides reside residing resided lives live living lived address addresses residence residential postal home
    """.split()
)
PHRASE_CUES: Final = frozenset(
    {"lives", "living", "lived", "resides", "residing", "resided", "reside", "located", "situated", "based"}
)
# Capitalised function words that start sentences, never part of a place or organisation name.
FUNCTION_WORDS: Final = frozenset(
    """
    the a an our your my his her their its this that these those in at on to from for with by of and or but as
    if when we i you he she they it dear re yours near via both all some any each every after before since until
    while also
    """.split()
)
SEPARATORS: Final = frozenset({"|", "||", "•", "·", "–", "—", "-", "¦", "/", "‖"})
# Words whose final full stop marks an abbreviation, not the end of a sentence ('St.', 'Dr.', 'Mt.', 'Cnr.').
DOT_ABBREVIATIONS: Final = frozenset(
    {"mt", "pt", "cnr", "apt", "ste", "lvl", "flr", "fl", "rm", "nth", "sth", "no", "approx", "ph", "tel"}
)
FURNITURE: Final = re.compile(r"^(?:page\s*\d+(?:\s*(?:of|/)\s*\d+)?|-\s*\d+\s*-|\d+)$", re.IGNORECASE)

# ----------------------------------------------------------------------------------------------------- word shapes
HN: Final = re.compile(r"^\d{1,5}[A-Za-z]?(?:[-–]\d{1,5}[A-Za-z]?)?$")  # 12, 12A, 12-14
HN_SLASH: Final = re.compile(r"^(?:LG|UG|B|G|M)?\d{1,5}[A-Za-z]?/\d{1,5}[A-Za-z]?$")  # 5/3, B02/53
BOXNUM: Final = re.compile(r"^\d{1,6}[A-Za-z]?$")
GLUED: Final = re.compile(r"^(NSW|VIC|QLD|SA|WA|TAS|NT|ACT)(\d{4})$")
PC_OCR: Final = re.compile(r"^[0-9OoIlSB]{4}$")
LETTER_ID: Final = re.compile(r"^[A-Z0-9]{1,3}$")
# Overseas postcodes: four to six digits ('1010', '90210'), a UK code ('OX1 4AA') and a Canadian code ('K1A 0B1').
POSTCODE_ONE_WORD: Final = re.compile(r"^\d{4,6}$")
POSTCODE_UK_OUT: Final = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?$")
POSTCODE_UK_IN: Final = re.compile(r"^\d[A-Z]{2}$")
POSTCODE_CA_OUT: Final = re.compile(r"^[A-Z]\d[A-Z]$")
POSTCODE_CA_IN: Final = re.compile(r"^\d[A-Z]\d$")

# ----------------------------------------------------------------------------------------------------- labels
# Form and field labels (compared on normalised words: lower case, ':' and '.' removed, '/' kept).
LABELS: Final[Mapping[tuple[str, ...], str]] = {
    ("address",): "address",
    ("addr",): "address",
    ("residential", "address"): "address",
    ("postal", "address"): "address",
    ("home", "address"): "address",
    ("business", "address"): "address",
    ("street", "address"): "address",
    ("mailing", "address"): "address",
    ("employer", "address"): "address",
    ("employer's", "address"): "address",
    ("employers", "address"): "address",
    ("clinic", "address"): "address",
    ("correspondence", "address"): "address",
    ("current", "address"): "address",
    ("previous", "address"): "address",
    ("work", "address"): "address",
    ("address", "line", "1"): "address",
    ("address", "line", "2"): "address",
    ("practice", "address"): "address",
    ("residential/postal", "address"): "address",
    ("postal/residential", "address"): "address",
    ("home",): "address",
    ("postal",): "address",
    ("street",): "street",
    ("street", "name"): "street_name",
    ("street", "no"): "street_no",
    ("street", "number"): "street_no",
    ("house", "number"): "street_no",
    ("house", "no"): "street_no",
    ("unit",): "unit",
    ("unit", "no"): "unit",
    ("unit/flat", "no"): "unit",
    ("unit/flat",): "unit",
    ("flat", "no"): "unit",
    ("unit", "number"): "unit",
    ("unit/flat", "number"): "unit",
    ("suburb",): "suburb",
    ("town",): "suburb",
    ("suburb/town",): "suburb",
    ("town/suburb",): "suburb",
    ("town", "or", "suburb"): "suburb",
    ("suburb", "or", "town"): "suburb",
    ("city",): "suburb",
    ("city/town",): "suburb",
    ("town/city",): "suburb",
    ("locality",): "suburb",
    ("suburb/locality",): "suburb",
    ("state",): "state",
    ("state/territory",): "state",
    ("state", "or", "territory"): "state",
    ("postcode",): "postcode",
    ("post", "code"): "postcode",
    ("p/code",): "postcode",
    ("pcode",): "postcode",
}
OTHER_LABELS: Final = frozenset(
    {
        ("occupation",),
        ("employer",),
        ("date", "of", "birth"),
        ("dob",),
        ("place", "of", "birth"),
        ("name",),
        ("surname",),
        ("given", "names"),
        ("given", "name"),
        ("first", "name"),
        ("family", "name"),
        ("phone",),
        ("telephone",),
        ("mobile",),
        ("email",),
        ("date",),
        ("sex",),
        ("gender",),
        ("position",),
        ("title",),
        ("country", "of", "birth"),
        ("provider",),
        ("country",),
        ("signature",),
        ("claim", "number"),
    }
)
# Compound labels ('Unit/building no.', 'Suburb/town/locality'): every '/' piece from one family.
LABEL_FAMILY: Final[Mapping[str, str]] = {
    "suburb": "suburb",
    "town": "suburb",
    "locality": "suburb",
    "city": "suburb",
    "unit": "unit",
    "flat": "unit",
    "building": "unit",
    "lot": "unit",
    "apt": "unit",
    "apartment": "unit",
    "shop": "unit",
    "suite": "unit",
    "villa": "unit",
    "level": "unit",
    "state": "state",
    "territory": "state",
    "postcode": "postcode",
    "pcode": "postcode",
    "street": "street",
    "house": "street",
    "address": "address",
    "addr": "address",
}
LABEL_MODS: Final[Mapping[str, str]] = {"no": "no", "number": "no", "num": "no", "#": "no", "nos": "no", "name": "name"}
LABEL_QUALS: Final = frozenset(
    """
    residential postal home business employer employer's employers clinic practice correspondence current
    previous work mailing line 1 2 of or and & the your worker's workers claimant's usual permanent
    """.split()
)
NUMBERING: Final = re.compile(r"^(?:\d{1,2}|[a-z])[.)]?$")

# ----------------------------------------------------------------------------------------------------- roles
STREET_ROLES: Final = frozenset(
    {
        Role.BUILDING,
        Role.DESIGNATOR,
        Role.UNIT_NUMBER,
        Role.LEVEL_NUMBER,
        Role.NUMBER,
        Role.UNIT_NUMBER_AND_NUMBER,
        Role.STREET_NAME,
        Role.STREET_TYPE,
        Role.STREET_SUFFIX,
        Role.CORNER,
        Role.BOX_KIND,
        Role.BOX_NUMBER,
        Role.BOX_EXTRA,
        Role.VIA,
        Role.DX,
        Role.DX_NUMBER,
        Role.DX_EXCHANGE,
    }
)
LOCALITY_ROLES: Final = frozenset(
    {Role.SUBURB, Role.STATE, Role.POSTCODE, Role.STATE_AND_POSTCODE, Role.COUNTRY, Role.CAPITAL_CITY_PHRASE}
)
# The words that stay unchanged (the case table's structure_roles).
STRUCTURE_ROLES: Final = frozenset(
    {
        Role.DESIGNATOR,
        Role.BOX_KIND,
        Role.CORNER,
        Role.VIA,
        Role.DX,
        Role.STATE,
        Role.COUNTRY,
        Role.CAPITAL_CITY_PHRASE,
    }
)
