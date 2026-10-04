"""Shared lexical fragments for rule detectors (Australian conventions)."""

from __future__ import annotations

import re

TITLES = (
    r"(?:Mr|Mrs|Ms|Miss|Mx|Mstr|Master|Dr|Doctor|Prof|Professor|A/Prof|Assoc(?:iate)?\.?[ \t]+Prof(?:essor)?"
    r"|Sir|Dame|Rev|Reverend|Fr|Father|Sr|Sister|Br|Brother|Hon|Judge|Justice|Magistrate|Commissioner"
    r"|Coroner|Registrar|Arbitrator|Senior[ \t]+Member|Conciliator|Mediator)"
)

# One name token: Capitalised, ALL CAPS, Mc/Mac/O' prefixes, hyphenated/apostrophised, or an initial.
# A regex fragment, not a credential (the name trips the hardcoded-password heuristic).
NAME_TOKEN = (
    r"(?:(?:Mc|Mac|O['’]|D['’])[A-Z][a-z]+|[A-Z][a-z]+(?:[-'’][A-Z]?[a-z]+)*"  # noqa: S105
    r"|[A-Z]{2,}(?:[-'’][A-Z]+)*|[A-Z]\.?)"
)
PARTICLE = r"(?:van|von|de|der|den|del|della|di|da|du|dos|das|la|le|bin|binti|ibn|al|el|ter|ten)"

# Capitalised words that are never (part of) a person's name in this context.
NON_NAME_WORDS = frozenset(
    w.lower()
    for w in """
    The And Of In On At To For With From By Re Dear Yours Regards Report Hospital Clinic Medical Centre
    Center Pty Ltd Limited Group Services Service Health Department Court Tribunal Commission Insurance
    Lawyers Solicitors Partners Associates Practice Surgery Specialist Specialists Rooms Street Road
    Avenue Drive Please Thank Thanks Kind Sincerely Faithfully Copy Cc Encl Attachment Page Date Claim
    Policy Reference Ref Number No Patient Claimant Worker Doctor Sir Madam Colleague Team Manager Officer
    Re: Mon Tue Wed Thu Fri Sat Sun Monday Tuesday Wednesday Thursday Friday Saturday Sunday January
    February March April June July August September October November December He She They His Her
    Their It This That These Those I We You Our Your Was Were Is Are Has Have Had Not No Yes Also
    Today Yesterday Tomorrow Medicare Centrelink NSW VIC QLD SA WA TAS NT ACT Australia Australian
    """.split()
)

# Post-nominal qualifications: kept visible, and they terminate a name.
POST_NOMINALS = frozenset(
    w.lower()
    for w in """
    FRACS FRACP FRANZCP FAFOM FAFRM FACSEP FRCS FRCP MBBS MBChB MB BS MD PhD DPhil BSc BMedSc BAppSc
    MPhty MPsych MClinPsych FRANZCR FANZCA FRACGP FACRRM FACEM FRANZCOG FCICM DipMSM OAM AM AO AC QC SC
    KC LLB LLM BA MA MSc FAOrthA FAPS MAPS FAMA GAICD BPhty BPhysio DPT BOccThy BPsych MBA JP RN EN
    """.split()
)

STATES = (
    r"(?:NSW|VIC|QLD|SA|WA|TAS|NT|ACT|N\.S\.W\.?|Vic\.?|Qld\.?|Tas\.?|New[ \t]+South[ \t]+Wales|Victoria"
    r"|Queensland|South[ \t]+Australia|Western[ \t]+Australia|Tasmania|Northern[ \t]+Territory"
    r"|Australian[ \t]+Capital[ \t]+Territory)"
)

# Large cities kept under the Broad profile (low re-identification risk).
CAPITAL_CITIES = frozenset({"sydney", "melbourne", "brisbane", "perth", "adelaide", "hobart", "darwin", "canberra"})

STREET_TYPES = (
    r"(?:Street|St|Road|Rd|Avenue|Ave|Av|Drive|Dr|Court|Ct|Place|Pl|Crescent|Cres|Cr|Lane|Ln|Way|Parade"
    r"|Pde|Terrace|Tce|Boulevard|Boulevarde|Blvd|Bvd|Highway|Hwy|Close|Cl|Circuit|Cct|Grove|Gr|Gve|Square"
    r"|Sq|Esplanade|Esp|Track|Trail|Rise|Row|Mews|Walk|Loop|Link|Vista|View|Glade|Gardens|Gdns|Heights"
    r"|Hts|Ridge|Promenade|Quay|Alley|Arcade|Bend|Brace|Chase|Concourse|Corso|Cove|Crest|Cross|Dell"
    r"|Entrance|Gate|Grange|Green|Hill|Junction|Landing|Mall|Meander|Nook|Outlook|Pass|Path|Pathway"
    r"|Point|Reserve|Retreat|Round|Strip|Turn|Vale|Wharf|Circle|Plaza|Parkway|Freeway|Motorway)"
)
STREET_TYPE_WORDS = frozenset(w.lower() for w in re.findall(r"[A-Za-z]+", STREET_TYPES))

MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)

# Medical abbreviations that look like initials and must never be treated as names.
MEDICAL_ABBREVIATIONS = frozenset(
    w.lower()
    for w in """
    OT PT GP CT MR MRI ED ICU GCS BP HR RR ECG EEG EMG NCS XR US DVT PE MI CVA TIA COPD CKD AF IV IM SC
    PO PR PRN BD TDS QID NBM ROM WPI AMA DASS PTSD MDD GAD OCD ADHD ASD TBI LOC NAD SOB RSI CRPS
    """.split()
)

# Nouns that make a preceding surname-like word clinical content ("Tinel's sign", "Colles fracture").
CLINICAL_NOUNS = frozenset(
    """sign signs test tests fracture fractures disease syndrome manoeuvre maneuver classification score scale
    criteria procedure operation approach lesion neuroma cyst contracture palsy phenomenon reflex view node
    angle index questionnaire inventory grade grading stage staging""".split()
)

# Public bodies, schemes, courts and tribunals: context, not identifying (plan §7.1 "not redacted").
# Compared by ``match_key`` (case-, space- and punctuation-insensitive).
PUBLIC_BODIES = frozenset(
    """
    medicare centrelink servicesaustralia ndis ndia sira icare worksafe worksafevictoria workcover
    workcoverqueensland workcoversa returntoworksa comcare tac transportaccidentcommission dva
    departmentofveteransaffairs ahpra ama racgp racs racp tga pbs mbs ato australiantaxationoffice
    nswhealth queenslandhealth sahealth wahealth departmentofhealth personalinjurycommission
    workerscompensationcommission ncat vcat qcat aat administrativeappealstribunal fairworkcommission
    highcourt federalcourt supremecourt districtcourt countycourt magistratescourt localcourt
    federalcircuitcourt coronerscourt nswpolice victoriapolice queenslandpolice
    """.split()
)

# Dose forms / modifiers that may follow a medicine name ("Panadeine Forte", "Targin 10 mg SR").
DOSE_FORM_WORDS = frozenset(
    """forte plus extra osteo duo mite rapid max sr cr xr er mr cd la odt mg mcg microgram micrograms tablet
    tablets tabs capsule capsules caps patch patches gel cream injection injections syrup liquid drops spray
    inhaler suppository suppositories sachet sachets ampoule""".split()
)
