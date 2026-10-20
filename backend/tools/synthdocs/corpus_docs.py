# ruff: noqa: E501 - document text and markup lines read better unbroken
"""The nine documents of the synthetic corpus (markup described in ``synthdocs.corpus``).

Everyone and everything here is invented.  Mobile numbers are from ACMA's fictional range
0491 570 xxx-0491 579 xxx and landlines from (0x) 5550 xxxx; identifiers carry valid check digits
so the tool sees realistic numbers.
"""

from __future__ import annotations

from mlredact.detect.rules.checksums import abn_valid, acn_valid, ihi_valid, tfn_valid
from synthdocs.corpus import DocSpec, Page, Scan
from synthdocs.generator import medicare_number, provider_number


def _ihi(stem9: str) -> str:
    n = next(f"800360{stem9}{d}" for d in "0123456789" if ihi_valid(f"800360{stem9}{d}"))
    return f"{n[:4]} {n[4:8]} {n[8:12]} {n[12:]}"


def _tfn(stem8: str) -> str:
    n = next(f"{stem8}{d}" for d in "0123456789" if tfn_valid(f"{stem8}{d}"))
    return f"{n[:3]} {n[3:6]} {n[6:]}"


def _abn(stem9: str) -> str:
    n = next(f"{p}{stem9}" for p in range(10, 100) if abn_valid(f"{p}{stem9}"))
    return f"{n[:2]} {n[2:5]} {n[5:8]} {n[8:]}"


def _acn(stem8: str) -> str:
    n = next(f"{stem8}{d}" for d in "0123456789" if acn_valid(f"{stem8}{d}"))
    return f"{n[:3]} {n[3:6]} {n[6:]}"


LOW = Scan(dpi=300, skew=0.3, blur=0.4, noise=4.0, jpeg=80)
MEDIUM = Scan(
    dpi=200, colour=True, skew=1.2, blur=0.8, noise=9.0, jpeg=55, paper=(250, 247, 238), shadow=0.15, speckle=15
)


# ------------------------------------------------------------------------------------------------
# Low: clean straight scans, one column, every identifier labelled
# ------------------------------------------------------------------------------------------------

_mc_rowe = medicare_number("29537015", 1)
_prov_raman = provider_number("231874", "B")

GP_REFERRAL = DocSpec(
    slug="low-01-gp-referral",
    tier="low",
    title="GP referral letter to an orthopaedic surgeon",
    challenges=[
        "Clean 300 dpi scan, straight, one column of typed text.",
        "Every identifier is labelled (Medicare, Provider No, claim).",
        "Names appear in full and as 'Mr Rowe' / 'Dr Whitfield'.",
        "Must keep: medicines, tests named after people (Neer's sign), dates, suburbs and postcodes.",
    ],
    size=10.0,
    scan=LOW,
    seed=11,
    pages=[
        Page(
            [
                "@logo 56,40 | NF",
                "@text 104,44,13,Bold | [Northgate Family Medical Centre](organisation)",
                "@text 104,62,8.5,Regular | [Shop 4, 210 Pacific Highway](street_address), [Hornsby](locality) NSW [2077](postcode)",
                "@text 104,74,8.5,Regular | Ph [(02) 5550 1842](phone)   Fax [(02) 5550 1843](phone)   [reception@northgatefmc.example](email)",
                "@gap 28",
                "@rule",
                "[14 March 2025](date)",
                "",
                "Dr [Andrew Whitfield](person), Orthopaedic Surgeon",
                "[Lakeside Orthopaedics](organisation), [Suite 12, 45 Ocean Street](street_address), [Narrabeen](locality) NSW [2101](postcode)",
                "",
                "@h3 Re: Mr [Michael James Rowe](person)",
                f"Date of birth: [02/07/1981](date_of_birth)      Medicare: [{_mc_rowe}](medicare) (IRN 2)",
                "Address: [17 Banksia Crescent](street_address), [Asquith](locality) NSW [2077](postcode)",
                "Mobile: [0491 570 156](phone)      Workers compensation claim: [WC0425-77811](reference)",
                "",
                "Dear Dr [Whitfield](person),",
                "",
                "Thank you for seeing Mr [Rowe](person), a [43-year-old](age) warehouse supervisor, about his right shoulder. "
                "On [3 February 2025](date) he was lifting a carton onto a high shelf at work at [Kingsford Logistics](organisation) "
                "when he felt a sudden tearing pain in the shoulder. Since then he has had night pain and difficulty reaching overhead.",
                "",
                "On examination there is a painful arc between 70 and 120 degrees of abduction. The <Hawkins-Kennedy test>(eponym) "
                "and <Neer's sign>(eponym) are both positive, and <Jobe's test>(eponym) is weak on the right. An ultrasound at "
                "[Hornsby Imaging](organisation) on [21 February 2025](date) showed a <partial-thickness supraspinatus tear>(clinical) "
                "with <subacromial bursitis>(clinical).",
                "",
                "He has had six sessions of physiotherapy with Ms [Olivia Brandt](person) at [Hornsby Sports Physio](organisation) "
                "with limited improvement. I would value your opinion on further management, including whether surgical repair is "
                "indicated.",
                "",
                "@h3 Current medications",
                "<Celecoxib 200 mg>(medication) once daily; <Panadol Osteo>(medication) two tablets three times daily; "
                "<Metformin XR 1 g>(medication) at night.",
                "",
                "@h3 Past history",
                "<Type 2 diabetes>(clinical), hypertension, left knee arthroscopy in 2016. Allergy: <penicillin>(medication) (rash). "
                "Non-smoker.",
                "",
                "The insurer's case manager is Ms [Tamsin Gale](person) of [Harbourline Insurance](organisation), telephone "
                "[(02) 5550 1977](phone). Please send a copy of your report to the practice and to the insurer.",
                "",
                "Kind regards,",
                "@sig",
                "Dr [Priya Raman](person) MBBS FRACGP",
                f"Provider No: [{_prov_raman}](provider_number)",
                "",
                "@small cc: Mr [Michael Rowe](person)",
            ]
        )
    ],
)

_mc_henderson = medicare_number("41820937", 3)
_prov_nguyen = provider_number("426815", "F")

RADIOLOGY = DocSpec(
    slug="low-02-radiology-report",
    tier="low",
    title="MRI lumbar spine report",
    challenges=[
        "Clean 300 dpi scan with a details table at the top.",
        "Two hospital-style numbers (UR number, accession) beside the Medicare and provider numbers.",
        "Name written 'SURNAME, Given Middle' in the table and 'Ms Henderson' in the text.",
        "Must keep: anatomy, findings and terms named after people (Modic, Schmorl's node).",
    ],
    size=10.0,
    scan=LOW,
    seed=12,
    footer=f"[Coastal Radiology Group](organisation)  |  ABN [{_abn('004085616')}](abn)  |  Page {{page}} of {{pages}}",
    pages=[
        Page(
            [
                "@logo 56,40 | CR",
                "@text 104,44,14,Bold | [Coastal Radiology Group](organisation)",
                "@text 104,63,8.5,Regular | [Level 1, 88 Pittwater Road](street_address), [Manly](locality) NSW [2095](postcode)",
                "@text 104,75,8.5,Regular | Ph [(02) 5550 4410](phone)   Fax [(02) 5550 4411](phone)   [www.coastalradiology.example](url)",
                "@gap 28",
                "@rule",
                "@h1 MRI LUMBAR SPINE",
                "@table 95,150,95,143",
                "Patient | [HENDERSON, Sarah Louise](person) | Date of birth | [19/11/1986](date_of_birth)",
                "UR number | [CRG0081245](clinical_id) | Accession | [ACC-25-0457812](clinical_id)",
                f"Exam date | [08/04/2025](date) | Medicare | [{_mc_henderson}](medicare)",
                f"Referrer | Dr [Thomas Nguyen](person) | Provider No | [{_prov_nguyen}](provider_number)",
                "@end",
                "@h3 Clinical history",
                "Low back pain with left-sided sciatica since a lifting injury at work on [17 March 2025](date). ? disc protrusion. "
                "Workers compensation claim.",
                "@h3 Technique",
                "Sagittal T1, T2 and STIR and axial T2 sequences through the lumbar spine. No contrast was given.",
                "@h3 Findings",
                "There are five lumbar-type vertebral bodies. Vertebral body heights are maintained. A <Modic type 1 endplate "
                "change>(eponym) is present at the inferior endplate of L4. A small <Schmorl's node>(eponym) is noted at the superior "
                "endplate of L2. The conus terminates normally at L1.",
                "",
                "L3/4: Mild disc desiccation. No canal or foraminal stenosis.",
                "",
                "L4/5: <Broad-based disc bulge>(clinical) with a superimposed left paracentral protrusion contacting the traversing "
                "left L5 nerve root in the lateral recess. Mild left foraminal narrowing.",
                "",
                "L5/S1: Normal disc. Facet joints are unremarkable.",
                "@h3 Conclusion",
                "@li 1. | Left paracentral disc protrusion at L4/5 contacting the traversing left L5 nerve root, consistent with the "
                "stated symptoms.",
                "@li 2. | Modic type 1 change at L4, suggesting active degenerative endplate change.",
                "",
                "Thank you for referring Ms [Henderson](person).",
                "@gap 6",
                "Reported by: Dr [Elena Petrova](person) FRANZCR",
                "Electronically signed [08/04/2025](date) 16:42",
                "@sig",
                "@small Copy to: Dr [Thomas Nguyen](person), [Manly Village Medical](organisation), fax [(02) 5550 4790](phone)",
            ]
        )
    ],
)

SOLICITOR_LETTER = DocSpec(
    slug="low-03-solicitor-instruction-letter",
    tier="low",
    title="Solicitor's letter instructing an independent medical examiner",
    challenges=[
        "Clean 300 dpi scan, two pages, running header on page 2.",
        "Law-firm references that embed staff initials (BL:KM:240917).",
        "Case citation naming a company (Malec v JC Hutton Pty Ltd) and the AMA Guides: both must stay.",
        "The claimant's surname alone appears many times ('Ms Morgan').",
    ],
    size=10.0,
    scan=LOW,
    seed=13,
    header="Ms [L. Morgan](person)  -  Claim [CTP-24-118473](reference)",
    footer="Page {page} of {pages}",
    pages=[
        Page(
            [
                "@text 56,40,16,Bold | [BENNETT LOWE LAWYERS](organisation)",
                f"@text 56,62,8.5,Regular | [Level 9, 88 Albert Street](street_address), [Brisbane](locality) QLD [4000](postcode)   "
                f"T [(07) 5550 2290](phone)   ABN [{_abn('629017342')}](abn)",
                "@gap 14",
                "@rule",
                "[17 April 2025](date)",
                "",
                "Our ref: [BL:KM:240917](reference)",
                "Your ref: [IME-7731](reference)",
                "",
                "@bold PRIVATE AND CONFIDENTIAL",
                "By email: [reports@qldmedicolegal.example](email)",
                "",
                "Dr [Samuel Okafor](person)",
                "[Queensland Medicolegal Assessments](organisation)",
                "[PO Box 5521](street_address), [West End](locality) QLD [4101](postcode)",
                "",
                "Dear Dr [Okafor](person),",
                "",
                "@bold Re: Our client Ms [Leanne Patricia Morgan](person), date of birth [27/05/1979](date_of_birth)",
                "@bold Claim number [CTP-24-118473](reference), insurer [Pacific Shield Insurance](organisation)",
                "",
                "We act for Ms [Morgan](person) in relation to injuries she sustained in a motor vehicle accident on "
                "[12 November 2024](date). Ms [Morgan](person) was the driver of a stationary vehicle that was struck from behind "
                "at traffic lights in [Kedron](locality). She reports neck pain, headaches and pain radiating into the right arm.",
                "",
                "We confirm that you have agreed to examine Ms [Morgan](person) at your rooms on [6 May 2025](date) at 10.30 am. "
                "She can be contacted directly on [0491 570 313](phone) if the appointment needs to change. She lives at "
                "[23 Lomandra Street](street_address), [Chermside](locality) QLD [4032](postcode).",
                "",
                "@h3 Enclosures",
                "@li 1. | Clinical notes of Dr [Hannah Lee](person), [Redcliffe Family Practice](organisation), "
                "[3 December 2024](date) to [28 March 2025](date).",
                "@li 2. | Physiotherapy report of Mr [Josh Turner](person) dated [3 March 2025](date).",
                "@li 3. | MRI cervical spine report from [Moreton Imaging](organisation) dated [14 January 2025](date).",
                "@li 4. | Statement of Ms [Morgan](person) dated [20 February 2025](date).",
            ]
        ),
        Page(
            [
                "@h3 Questions",
                "Please address the following matters in your report:",
                "@li 1. | What injuries did Ms [Morgan](person) sustain in the accident?",
                "@li 2. | Are her current symptoms causally related to the accident? Please comment on any pre-existing condition, "
                "including the <cervical spondylosis>(clinical) noted by Dr [Lee](person) in 2022.",
                "@li 3. | Has her condition reached maximum medical improvement?",
                "@li 4. | Please assess the degree of permanent impairment in accordance with the <AMA Guides to the Evaluation of "
                "Permanent Impairment, 5th Edition>(citation), giving reasons for the DRE category chosen.",
                "@li 5. | What is her capacity for work as a dental assistant, and what restrictions apply?",
                "@li 6. | What future treatment is reasonably required, and what is its likely cost?",
                "@li 7. | Having regard to <Malec v JC Hutton Pty Ltd (1990) 169 CLR 638>(citation), what is the percentage risk "
                "that her symptoms would have developed in any event?",
                "",
                "Your report should be addressed to this office and should comply with the <Expert Witness Code of "
                "Conduct>(citation). Your account should be sent to [Pacific Shield Insurance](organisation), quoting the claim "
                "number above.",
                "",
                "Please direct any queries to our paralegal, Ms [Kylie Marsh](person), on [(07) 5550 2291](phone) or "
                "[kmarsh@bennettlowe.example](email).",
                "",
                "Yours faithfully,",
                "@sig",
                "[Daniel Bennett](person)",
                "Principal, [Bennett Lowe Lawyers](organisation)",
                "",
                "@small Liability limited by a scheme approved under professional standards legislation.",
            ]
        ),
    ],
)


# ------------------------------------------------------------------------------------------------
# Medium: 200 dpi colour scans, tilted, stamps and stickers, running headers, forms, handwriting
# ------------------------------------------------------------------------------------------------

_prov_kowalski = provider_number("308421", "K")

IME_REPORT = DocSpec(
    slug="medium-01-ime-report",
    tier="medium",
    title="Independent medical examination report (orthopaedic)",
    challenges=[
        "200 dpi colour scan, tilted about a degree, light speckle.",
        "Claimant's name and date of birth repeated in a running header on every page.",
        "Wife and supervisor named by first name only ('Priya', 'Gary').",
        "'Banks' is the claimant's surname and also an ordinary word ('creek banks'), which must stay.",
        "A red RECEIVED stamp naming the law firm, and a handwritten margin note.",
        "Driver licence number given without the usual layout.",
    ],
    scan=MEDIUM,
    size=10.0,
    seed=21,
    header="[BANKS, Daniel](person)   |   Claim [WC-2291-0456](reference)   |   DOB [11/09/1978](date_of_birth)",
    footer="Dr [M. Kowalski](person)  -  Independent Medical Examination  -  Page {page} of {pages}",
    pages=[
        Page(
            [
                "@logo 56,40 | RM",
                "@text 104,44,13,Bold | [Riverside Medicolegal Services](organisation)",
                "@text 104,62,8.5,Regular | [Suite 3, 101 Flinders Lane](street_address), [Melbourne](locality) VIC [3000](postcode)",
                "@text 104,74,8.5,Regular | T [(03) 5550 7712](phone)   [bookings@riversidemls.example](email)",
                "@gap 28",
                "@rule",
                "@h1 INDEPENDENT MEDICAL EXAMINATION",
                "@table 130,353",
                "Claimant | Mr [Daniel Robert Banks](person)",
                "Date of birth | [11/09/1978](date_of_birth)",
                "Address | [6 Pelican Court](street_address), [Werribee](locality) VIC [3030](postcode)",
                "Employer | [Coastline Freight Pty Ltd](organisation)",
                "Claim number | [WC-2291-0456](reference)",
                "Date of injury | [14 August 2024](date)",
                "Date of examination | [9 May 2025](date)",
                "Requested by | Ms [Fiona Halloran](person), [Halloran & Pike Lawyers](organisation), ref [FH/25/0193](reference)",
                "@end",
                "@h2 1. Introduction",
                "I examined Mr [Banks](person) at my rooms on [9 May 2025](date) at the request of Ms [Halloran](person). He "
                "attended with his wife, [Priya](person). I confirmed his identity by sighting his driver licence, number "
                "[058412337](driver_licence). I explained that I was not his treating doctor and that my report would go to the "
                "party who requested it. <The claimant>(role) consented to the examination.",
                "@h2 2. History of injury",
                "Mr [Banks](person) is a [46-year-old](age) forklift operator. On [14 August 2024](date) he was unloading a pallet "
                "of tiles at the [Laverton North](locality) depot when the pallet shifted and he twisted to hold it. He felt "
                "immediate pain in his lower back. His supervisor, [Gary](person), drove him to the [Westgate Medical "
                "Centre](organisation), where Dr [Leila Haddad](person) certified him unfit for work.",
                "",
                "He was referred for physiotherapy with Ms [Chloe Tran](person) and later for hydrotherapy at [Altona Aquatic "
                "Centre](organisation). A CT scan on [2 September 2024](date) showed a <disc protrusion at L4/5>(clinical). He "
                "returned to modified duties of four hours a day in [November 2024](date) but stopped after two weeks because of "
                "increasing pain.",
                "@stamp 345,640,-9 | RECEIVED | [HALLORAN & PIKE](organisation) | [22 MAY 2025](date)",
            ]
        ),
        Page(
            [
                "@h2 3. Current symptoms",
                "Mr [Banks](person) describes constant aching pain across the lower back, rated 6 out of 10, with intermittent pain "
                "into the left buttock and back of the thigh. Pain is worse with sitting for more than 30 minutes, bending and "
                "lifting. He sleeps poorly. He used to fish from the creek <banks>(homograph) near his home most weekends but can "
                "no longer manage the uneven ground.",
                "@h2 4. Current treatment",
                "@li - | <Pregabalin (Lyrica) 75 mg>(medication) twice daily",
                "@li - | <Tapentadol SR 50 mg>(medication) twice daily",
                "@li - | <Paracetamol 1 g>(medication) up to four times daily",
                "He continues to see Dr [Haddad](person) monthly and has a review with neurosurgeon Mr [Owen Fitzgerald](person) "
                "on [3 June 2025](date).",
                "@h2 5. Examination",
                "He walked with a normal gait and sat for 25 minutes before needing to stand. There was tenderness over the L4/5 "
                "and L5/S1 segments. <Straight leg raise>(clinical) was 60 degrees on the left and 80 degrees on the right. "
                "Reflexes were symmetrical. <Hoover's sign>(eponym) was negative and <Waddell's signs>(eponym) were 1 of 5.",
                "@table 160,110,110,103",
                "Lumbar movement | Measured | Normal | Comment",
                "Flexion | 40 deg | 60 deg | pain at end range",
                "Extension | 10 deg | 25 deg | pain",
                "Left lateral flexion | 15 deg | 25 deg | -",
                "Right lateral flexion | 20 deg | 25 deg | -",
                "@end",
                "@h2 6. Investigations reviewed",
                "CT lumbar spine, [Westgate Imaging](organisation), [2 September 2024](date): left paracentral disc protrusion at "
                "L4/5. MRI lumbar spine, [Werribee Radiology](organisation), [20 January 2025](date): unchanged protrusion with mild "
                "left lateral recess narrowing.",
                "@hand 455,250,-6 | see p.3 - | [Fiona](person)",
            ]
        ),
        Page(
            [
                "@h2 7. Diagnosis",
                "<Lumbar disc protrusion at L4/5 with left-sided radicular pain>(clinical), caused by the incident at work on "
                "[14 August 2024](date).",
                "@h2 8. Whole person impairment",
                "@table 200,180,103",
                "Body part | Method | WPI",
                "Lumbar spine | DRE Category III | 10%",
                "Lower extremity | Not rated | 0%",
                "Total |  | 10%",
                "@end",
                "The assessment follows the <AMA Guides to the Evaluation of Permanent Impairment, 5th Edition>(citation) as "
                "modified by the state guidelines. No deduction is made for a pre-existing condition.",
                "@h2 9. Capacity for work",
                "Mr [Banks](person) has capacity for suitable duties of six hours a day, three days a week, with no lifting over "
                "10 kg and the ability to change position every 30 minutes. He is not fit to drive a forklift at present. His "
                "employer's return to work coordinator, Ms [Janelle Ward](person) ([0491 571 491](phone)), has offered dispatch "
                "duties.",
                "@h2 10. Declaration",
                "I have read and complied with the <Expert Witness Code of Conduct>(citation). I have made all the inquiries I "
                "believe are desirable and appropriate.",
                "",
                "@sig",
                "Dr [Marek Kowalski](person) MBBS FRACS (Orth)",
                "Consultant Orthopaedic Surgeon",
                f"AHPRA [MED0001823456](ahpra)     Provider No [{_prov_kowalski}](provider_number)",
            ]
        ),
    ],
)

_mc_harris = medicare_number("51094726", 2)
_ihi_harris = _ihi("123456789")
_STICKER = (
    "@label 345,36,MRN00561287 | [HARRIS, Margaret Anne](person)   F | DOB [23/04/1952](date_of_birth)   MRN [00561287](clinical_id)"
    f" | Medicare [{_mc_harris}](medicare) | Ward 4B   Bed 12"
)

DISCHARGE = DocSpec(
    slug="medium-02-discharge-summary",
    tier="medium",
    title="Hospital discharge summary",
    challenges=[
        "200 dpi colour scan, tilted, with a hospital logo.",
        "Patient sticker on each page: name, date of birth, MRN, Medicare and a barcode.",
        "Healthcare identifier (IHI), pager number and a printing user ID.",
        "A network file path in the footer that contains the patient's surname.",
        "Next of kin table with family names, phones and an email.",
        "Must keep: medicines table, diagnoses, dates.",
    ],
    scan=MEDIUM,
    size=10.0,
    seed=22,
    footer=r"Printed [09/06/2025](date) 15:12 by [SLAM2](employee_id)   [\\sbph-fs01\discharge\2025\06\HARRIS_M_00561287.pdf](file_path)"
    "   Page {page} of {pages}",
    pages=[
        Page(
            [
                "@logo 56,40 | SB",
                "@text 104,46,12.5,Bold | [St Brigid's Private Hospital](organisation)",
                "@text 104,63,8,Regular | [220 Racecourse Road](street_address), [Ascot](locality) QLD [4007](postcode)",
                "@text 104,74,8,Regular | Ph [(07) 5550 3300](phone)",
                _STICKER,
                "@gap 52",
                "@rule",
                "@h1 DISCHARGE SUMMARY",
                "@table 110,131,110,132",
                "Admitted | [02/06/2025](date) | Discharged | [09/06/2025](date)",
                "Consultant | Dr [Rajesh Iyer](person) | Unit | Orthopaedics",
                f"IHI | [{_ihi_harris}](ihi) | Discharged to | Home",
                "@end",
                "@h3 Principal diagnosis",
                "<Right total knee replacement for primary osteoarthritis>(clinical)",
                "@h3 Complications",
                "<Post-operative delirium>(clinical) (day 2, resolved); <hyponatraemia>(clinical) (Na 128, resolved with fluid "
                "restriction).",
                "@h3 Clinical summary",
                "Mrs [Harris](person) is a [73-year-old](age) retired primary school teacher who lives with her husband "
                "[Keith](person) in a single-storey home in [Ascot](locality). She was admitted for an elective right total knee "
                "replacement, performed on [02/06/2025](date) by Dr [Iyer](person) under spinal anaesthesia. On day 2 she became "
                "confused overnight; a delirium screen was negative apart from mild hyponatraemia, which resolved. She mobilised "
                "with a four-wheeled walker by day 4 and was cleared by physiotherapy (Ms [Grace Holloway](person)) and "
                "occupational therapy for discharge home.",
                "@h3 Next of kin",
                "@table 160,120,203",
                "Name | Relationship | Contact",
                "[Keith Harris](person) | Husband | [(07) 5550 3381](phone)",
                "[Joanne Fraser](person) | Daughter | [0491 570 159](phone), [jo.fraser@example.net](email)",
                "@end",
            ]
        ),
        Page(
            [
                _STICKER,
                "@gap 52",
                "@h2 Medications on discharge",
                "@table 150,80,110,143",
                "Medicine | Dose | Frequency | Comment",
                "<Apixaban>(medication) | 2.5 mg | twice daily | 14 days, then stop",
                "<Targin (oxycodone/naloxone)>(medication) | 10/5 mg | twice daily | wean over 2 weeks",
                "<Paracetamol>(medication) | 1 g | four times daily | regular",
                "<Pantoprazole>(medication) | 40 mg | daily | while on analgesia",
                "<Amlodipine>(medication) | 5 mg | daily | unchanged",
                "@end",
                "Ceased: <ibuprofen>(medication) (renal function). Allergy: <codeine>(medication) (nausea).",
                "@h3 Follow-up",
                "@li - | Review with Dr [Iyer](person) at [Brisbane Joint Clinic](organisation) in six weeks; phone "
                "[(07) 5550 3400](phone) to book.",
                "@li - | GP review with Dr [Owen McCarthy](person), [Clayfield Medical Practice](organisation), within one week for "
                "a wound check and sodium; fax [(07) 5550 3488](phone).",
                "@li - | Community physiotherapy with [MoveWell Physiotherapy](organisation), [Hendra](locality), twice weekly.",
                "@li - | Staples to be removed on [16/06/2025](date).",
                "@h3 Information for the patient",
                "If the wound becomes red, hot or discharges fluid, or if you develop calf pain or shortness of breath, contact the "
                "ward on [(07) 5550 3342](phone) or go to the emergency department.",
                "@gap 8",
                "Completed by Dr [Sophie Lam](person) (Resident Medical Officer), pager [4471](other_id)",
                "@sig",
                "Electronically approved by Dr [Rajesh Iyer](person) on [09/06/2025](date)",
            ]
        ),
    ],
)

_mc_karimi = medicare_number("38265104", 1)
_prov_okoye = provider_number("457213", "L")

CERTIFICATE = DocSpec(
    slug="medium-03-certificate-of-capacity",
    tier="medium",
    title="Workers compensation certificate of capacity (hand-filled form)",
    challenges=[
        "200 dpi colour scan of a form filled in by hand in blue pen.",
        "Values sit in boxes under printed labels; some boxes are crowded.",
        "A third party (a resident, 'Mr Doyle') is named inside handwriting.",
        "Practice stamp with the doctor's name and provider number overlaps the form.",
        "Two signatures (doctor and worker) and ticked checkboxes.",
    ],
    scan=MEDIUM,
    size=10.0,
    seed=23,
    footer="Certificate of capacity  -  Page {page} of {pages}",
    pages=[
        Page(
            [
                "@h1 WORKERS COMPENSATION",
                "@center Certificate of Capacity / Certificate of Fitness",
                "@small <This certificate must be completed by the treating medical practitioner. Please write clearly in block "
                "letters.>(boilerplate)",
                "@h3 Part A - Worker details",
                "@field Worker's full name | ~[Ahmed Karimi](person) ;; Date of birth | ~[17/10/1990](date_of_birth)",
                "@field Residential address | ~[51 Wentworth Avenue](street_address), [Blacktown](locality) [2148](postcode)",
                f"@field Telephone | ~[0491 572 665](phone) ;; Medicare number | ~[{_mc_karimi}](medicare)",
                "@field Claim number | ~[WCN-7781-2209](reference) ;; Occupation | ~Personal care worker",
                "@field Employer | ~[Brightside Aged Care Pty Ltd](organisation) ;; Employer phone | ~[(02) 5550 8820](phone)",
                "@h3 Part B - Injury",
                "@field Date of injury | ~[28/04/2025](date) ;; Date first seen for this injury | ~[29/04/2025](date)",
                "How did the injury happen?",
                "@write 2 | ~Transferring resident (Mr [Doyle](person), room 14) from bed to chair, felt sharp pain low back L side.",
                "Diagnosis",
                "@write 2 | ~<Acute lumbar strain>(clinical), L paraspinal spasm. No neuro deficit.",
                "@h3 Part C - Capacity",
                "@check - | The worker is fit for pre-injury work.",
                "@check x | The worker has capacity for some type of employment from [12/05/2025](date) to [09/06/2025](date).",
                "@check - | The worker has no current work capacity for any employment.",
                "@field Lifting | ~max 5 kg ;; Sitting / standing | ~change every 30 min ;; Hours | ~4 h/day, 3 days/wk",
                "Treatment plan",
                "@write 2 | ~<Physio>(clinical) 2x weekly, <Celebrex 200mg>(medication) daily, <Panadol Osteo>(medication). Review 2 wks.",
            ]
        ),
        Page(
            [
                "@h3 Part D - Return to work",
                "@field Return to work coordinator | ~[Deb Saunders](person) ;; Phone | ~[0491 571 266](phone)",
                "Have you discussed return to work with the employer?",
                "@check x | Yes, by telephone on [13/05/2025](date)",
                "@check - | No",
                "@h3 Part E - Medical practitioner",
                f"@field Name | ~Dr [Grace Okoye](person) ;; Provider number | [{_prov_okoye}](provider_number)",
                "@field Practice | [Parkview Medical Centre](organisation), [3/88 Main Street](street_address), [Blacktown](locality) "
                "NSW [2148](postcode)",
                "@field Phone | [(02) 5550 8711](phone) ;; Email | [admin@parkviewmc.example](email)",
                "Signature of medical practitioner",
                "@sig",
                "@field Date | ~[12/05/2025](date) ;; Next review | ~[26/05/2025](date)",
                f"@stamp 318,236,-6 | [PARKVIEW MEDICAL CENTRE](organisation) | Dr [G. OKOYE](person) | Prov. [{_prov_okoye}](provider_number) | [(02) 5550 8711](phone)",
                "@h3 Part F - Worker declaration",
                "<I consent to my treating practitioner and the insurer exchanging information relevant to my injury and return to "
                "work.>(boilerplate)",
                "Signature of worker",
                "@sig",
                "@field Name | ~[AHMED KARIMI](person) ;; Date | ~[12/05/2025](date)",
                "@hand 380,640,-5 | RTW plan sent to | [Deb](person) 13/5",
            ]
        ),
    ],
)


# ------------------------------------------------------------------------------------------------
# Hard: low resolution, faded, tilted, rotated pages, fax, show-through, stains, dense names
# ------------------------------------------------------------------------------------------------

PSYCH = DocSpec(
    slug="hard-01-psychiatric-report",
    tier="hard",
    title="Psychiatric medico-legal report (bullying claim)",
    challenges=[
        "150 dpi grey scan, faded toner, tilted about 2 degrees, shadow at the binding edge, text from the back of the "
        "sheet showing through, dust specks.",
        "Small serif type (9.5 pt) in dense narrative.",
        "Many people per page: married and former surnames (née Tran), a nickname ('Kel'), children and parents by first "
        "name, surname-only and initial-plus-surname forms, an ALL-CAPS header form.",
        "'Hope' is a psychologist's surname and also an ordinary word ('lost all hope'), which must stay.",
        "Page 3 is a landscape table scanned sideways.",
        "Received stamp over the text; handwritten note with a phone number.",
    ],
    font="Serif",
    size=9.5,
    seed=31,
    scan=Scan(
        dpi=150,
        skew=2.2,
        blur=1.0,
        noise=12.0,
        jpeg=40,
        paper=(246, 242, 230),
        fade=0.3,
        shadow=0.4,
        speckle=60,
        show_through=0.13,
    ),
    header="CONFIDENTIAL - PSYCHIATRIC REPORT - Ms [K. O'SULLIVAN](person) - PIC matter [2024/118832](court_file)",
    footer="Dr [A. Venkataraman](person)  -  Page {page} of {pages}",
    pages=[
        Page(
            [
                "@center Dr [Anand Venkataraman](person) MBBS FRANZCP",
                "@center Consultant Psychiatrist   AHPRA [MED0002345678](ahpra)",
                "@center [Level 2, 15 Hunter Street](street_address), [Newcastle](locality) NSW [2300](postcode)   T [(02) 5550 6610](phone)"
                "   [rooms@drvpsych.example](email)",
                "@rule",
                "[20 March 2025](date)",
                "",
                "Ms [Renee Walsh](person), Senior Associate, [Carter Webb Lawyers](organisation)",
                "[PO Box 902](street_address), [Newcastle](locality) NSW [2300](postcode)      Your reference: [RW:24/0917](reference)",
                "",
                "Dear Ms [Walsh](person),",
                "@bold Re: Ms [Kellie-Anne O'Sullivan](person) (nee [Tran](person)), born [08/12/1988](date_of_birth)",
                "@bold Employer: [Valley Community Care Ltd](organisation); deemed date of injury [1 August 2023](date)",
                "",
                "Thank you for asking me to assess Ms [O'Sullivan](person). I interviewed her by video on [11 March 2025](date) for "
                "two and a half hours. She was alone in a private room at her home in [Rutherford](locality). I explained the "
                "limits of confidentiality and she agreed to proceed.",
                "@h3 1. Documents reviewed",
                "@li (a) | Report of Ms [Joy Hope](person), clinical psychologist, dated [12 January 2025](date).",
                "@li (b) | Clinical notes of [Maitland Road Medical Centre](organisation), 2019 to 2025.",
                "@li (c) | Statement of Ms [O'Sullivan](person) dated [4 June 2024](date).",
                "@li (d) | Incident report by Mr [Craig Pemberton](person), Human Resources Manager, dated [9 August 2023](date).",
                "@li (e) | File note of the rehabilitation provider, [Return2Work Partners](organisation).",
                "@h3 2. History of the injury",
                "Ms [O'Sullivan](person), who prefers to be called '[Kel](person)', worked for [Valley Community Care](organisation) "
                "as a disability support coordinator from 2017. She enjoyed the role until early 2023, when Mr [Brett "
                "Lawson](person) was appointed as her team leader. She said [Brett](person) criticised her in front of clients, "
                "changed her rosters without notice and sent her text messages late at night from his personal mobile "
                "([0491 572 549](phone)). One message read: 'you're useless [Kellie](person), everyone knows it'. She forwarded "
                "copies to her own address, [kel.osullivan88@example.com](email), so that they could not be deleted.",
                "",
                "On [1 August 2023](date), after a meeting in which Mr [Lawson](person) told her she would be 'performance managed "
                "out', she drove to the car park of the [Greenhills](locality) shopping centre and sat crying for two hours. Her "
                "husband found her there. She has not returned to work since.",
                "@stamp 330,668,-12 | RECEIVED | [CARTER WEBB LAWYERS](organisation) | [24 MAR 2025](date)",
            ]
        ),
        Page(
            [
                "@h3 3. Personal and family history",
                "Ms [O'Sullivan](person) was born in [Cabramatta](locality) to parents who came to Australia from Vietnam in 1981. "
                "Her father, [Minh Tran](person), died of a heart attack in 2010. Her mother, [Lan](person), lives in "
                "[Fairfield](locality) and speaks limited English. Her brother, [Danny Tran](person), is a nurse at [Lakeside "
                "Private Hospital](organisation).",
                "",
                "She completed Year 12 at [Cabra Vale High School](organisation) and a Diploma of Community Services at [Hunter "
                "Institute of Technology](organisation). She married [Liam O'Sullivan](person), an electrician, in 2014. They have "
                "two children, [Ava](person) ([9](age)) and [Noah](person) ([6](age)). In her early twenties she was in a "
                "relationship with [Tuan Pham](person), which ended because of his gambling.",
                "@h3 4. Treatment",
                "Her general practitioner, Dr [Imogen Clarke](person), prepared a <Mental Health Treatment Plan>(clinical) in "
                "[August 2023](date) and started <sertraline>(medication). She was referred to Ms [Hope](person), whom she has seen "
                "fortnightly. She told me she had 'lost all <hope>(homograph)' by late 2023, and Ms [Hope](person) recorded "
                "<passive suicidal ideation>(clinical) at that time. She was admitted to [Lakeside Private Hospital](organisation) "
                "under Dr [Felix Moreau](person) from [19 March 2024](date) to [9 April 2024](date). A chronology is set out in "
                "Table 1 overleaf.",
                "@h3 5. Current medication",
                "@li - | <Desvenlafaxine (Pristiq) 100 mg>(medication) in the morning",
                "@li - | <Quetiapine 25 mg>(medication) at night",
                "@li - | <Diazepam 5 mg>(medication) as needed, about twice a week",
                "@h3 6. Mental state examination",
                "Ms [O'Sullivan](person) presented as a neatly groomed woman of Vietnamese-Australian background who appeared her "
                "stated age. She was tearful at times. Speech was normal in rate and volume. Mood was 'flat' and affect was "
                "restricted and congruent. There was no formal thought disorder. She described intrusive memories of the meetings "
                "with Mr [Lawson](person), avoidance of the [Maitland](locality) office and hypervigilance whenever her phone rings. "
                "She denied current suicidal intent. Insight and judgement were fair.",
                "@hand 372,640,8 | call [Kel](person) re appt | [0491 572 983](phone)",
            ]
        ),
        Page(
            [
                "@h2 Table 1 - Chronology of treatment",
                "@table 75,200,165,290",
                "Date | Provider | Contact | Notes",
                "[14/08/2023](date) | Dr [Imogen Clarke](person), [Maitland Road Medical Centre](organisation) | "
                "[(02) 5550 6702](phone) | <Mental Health Treatment Plan>(clinical); <sertraline 50 mg>(medication) started",
                "[02/10/2023](date) | Ms [Joy Hope](person), psychologist | [joy.hope@mindpath.example](email) | 10 sessions under "
                "Better Access; <PHQ-9>(clinical) score 21",
                "[19/03/2024](date) | Dr [Felix Moreau](person), [Lakeside Private Hospital](organisation) | MRN "
                "[LPH-0094412](clinical_id) | Admission for 3 weeks; changed to <desvenlafaxine>(medication)",
                "[05/11/2024](date) | Ms [Tara Singh](person), [Return2Work Partners](organisation) | [0491 572 665](phone) | "
                "Vocational assessment; not fit for previous role",
                "[12/01/2025](date) | Ms [Joy Hope](person) | [joy.hope@mindpath.example](email) | Report; <DASS-21>(clinical) in "
                "the severe range",
                "[11/03/2025](date) | Dr [A. Venkataraman](person) | - | This assessment",
                "@end",
                "@small Abbreviations: PHQ-9, Patient Health Questionnaire; DASS-21, Depression Anxiety Stress Scales; MRN, medical "
                "record number.",
            ],
            landscape=True,
        ),
        Page(
            [
                "@h3 7. Diagnosis",
                "Using <DSM-5-TR>(clinical) criteria, Ms [O'Sullivan](person) has a <major depressive disorder, single episode, "
                "moderate, with anxious distress>(clinical). The predominant cause is the treatment she received from Mr "
                "[Lawson](person) at work.",
                "@h3 8. Whole person impairment (<PIRS>(clinical))",
                "@table 210,60,213",
                "Category | Class | Reason",
                "Self care and personal hygiene | 1 | independent",
                "Social and recreational activities | 3 | stopped netball, rarely leaves home",
                "Travel | 2 | avoids driving past the [Maitland](locality) office",
                "Social functioning | 2 | strained relationship with husband",
                "Concentration, persistence and pace | 3 | cannot read more than a page",
                "Employability | 4 | cannot work in any role at present",
                "@end",
                "This gives a median class of 2.5 and a total of 15% whole person impairment under the <Guidelines for the "
                "Evaluation of Permanent Impairment>(citation).",
                "@h3 9. Opinion",
                "In my opinion Ms [O'Sullivan](person) has no current capacity for work. She requires psychiatric review at least "
                "monthly and continued psychological treatment. I would review her again in six months.",
                "",
                "I have read the <Expert Witness Code of Conduct>(citation) and agree to be bound by it.",
                "",
                "Yours sincerely,",
                "@sig",
                "Dr [Anand Venkataraman](person)",
                "Consultant Psychiatrist",
                "@stamp 150,520,7 | COPY | [CARTER WEBB LAWYERS](organisation) | File [RW:24/0917](reference)",
            ]
        ),
    ],
)

_mc_gill = medicare_number("27719450", 4)
_ihi_gill = _ihi("987654321")

FAX = DocSpec(
    slug="hard-02-faxed-clinical-notes",
    tier="hard",
    title="Faxed GP clinical notes with a certified licence copy",
    challenges=[
        "Fax quality: black-and-white only, half vertical resolution, jagged letters, heavy speckle, tilted.",
        "Fax header line on every page with the sending practice and fax numbers.",
        "Typewriter-style notes with abbreviations; doctors referred to by initials only ('NI', 'KM').",
        "Partial dates ('03/01') and unlabelled health identifiers.",
        "A copy of a driver licence card: photo, licence and card numbers, address, signature.",
        "Justice of the Peace stamp containing a name and registration number; handwritten notes.",
        "Printing user ID and a local file path that contain identifiers.",
    ],
    seed=32,
    size=10.0,
    scan=Scan(
        dpi=200,
        skew=1.5,
        blur=0.6,
        noise=8.0,
        speckle=120,
        fax="[18/02/2025](date) 09:14  FROM: [WOLLONGONG ST MEDICAL](organisation) [+61 2 5550 9012](phone)"
        "  TO: [0255509300](phone)  P.{page}/{pages}",
    ),
    pages=[
        Page(
            [
                "@gap 10",
                "@h1 FACSIMILE",
                "@gap 8",
                "@field To | Mr [Ben Kirby](person), [Illawarra Claims Management](organisation) ;; Fax | [(02) 5550 9300](phone)",
                "@field From | Dr [Natasha Ivanova](person), [Wollongong Street Medical](organisation) ;; Phone | [(02) 5550 9012](phone)",
                "@field Date | [18/02/2025](date) ;; Pages (including this one) | 3",
                "@field Re | [GILL, Harpreet Singh](person)  -  DOB [30/06/1975](date_of_birth)  -  Claim [ICM/0398817](reference)",
                "@gap 10",
                "Further to your request of [12 February 2025](date), please find attached the clinical notes for the above patient "
                "from [6 January 2025](date) to [10 February 2025](date), and a certified copy of his photo identification. A "
                "signed authority from Mr [Gill](person) is held on file.",
                "@hand 300,400,-4 | [Ben](person) - notes as requested. | Pls call if you need the XR report. | - [Nat](person)",
                "@gap 330",
                "@small <CONFIDENTIALITY NOTICE: This facsimile contains confidential health information intended only for the "
                "addressee. If you have received it in error, please notify the sender immediately by telephone and destroy all "
                "copies.>(boilerplate)",
                f"@small [Wollongong Street Medical](organisation)  -  [41 Keira Street](street_address), [Wollongong](locality) NSW "
                f"[2500](postcode)  -  ABN [{_abn('118204557')}](abn)",
            ]
        ),
        Page(
            [
                "@size 8.3",
                "@pre [WOLLONGONG STREET MEDICAL](organisation) - CLINICAL NOTES (printed [17/02/2025](date))",
                "@pre Patient: Mr [Harpreet Singh GILL](person)    DOB: [30/06/1975](date_of_birth)    Chart: [004417](clinical_id)",
                "@pre Address: [9/41 Corrimal St](street_address), [Wollongong](locality) [2500](postcode)    Mob: [0491 573 770](phone)",
                f"@pre Medicare: [{_mc_gill}](medicare)    [{_ihi_gill}](ihi)",
                "@pre " + "-" * 88,
                "@pre [06/01/2025](date) 09:40   Dr [N Ivanova](person)",
                "@pre   S: Fell approx 1.5 m from ladder at work ([Illawarra Steel Fabrication](organisation)) on [03/01](date).",
                "@pre      FOOSH. R wrist pain, swelling. Works as boilermaker, R hand dominant.",
                "@pre   O: Tender anatomical snuffbox. <Finkelstein's>(eponym) neg. Grip reduced R.",
                "@pre   A: ?<scaphoid #>(clinical)",
                "@pre   P: XR R wrist + scaphoid views -> [Southern IR](organisation). Thumb spica. WC cert issued,",
                "@pre      unfit 2/52. Review 2/7. Claim no. per pt [ICM/0398817](reference).",
                "@pre",
                "@pre [13/01/2025](date) 10:15   Dr [N Ivanova](person)",
                "@pre   S: XR NAD per report. Ongoing pain. Wife [Gurpreet](person) present, worried re income.",
                "@pre   O: Snuffbox still tender.",
                "@pre   P: CT scaphoid. Ref Dr [K Mehta](person) (hand surgeon), [Figtree Hand Clinic](organisation)",
                "@pre      ph [(02) 5550 9045](phone). Cert unfit to [27/01](date).",
                "@pre",
                "@pre [27/01/2025](date) 16:50   Dr [A Kwan](person) (locum)",
                "@pre   S: CT shows undisplaced waist # scaphoid. Seen by Dr [Mehta](person) [24/1](date) - cast 6/52.",
                "@pre      Pt asks for letter for course, coordinator [Sharon](person) at [Illawarra Trade College](organisation)",
                "@pre      ph [(02) 5550 9530](phone).",
                "@pre   P: Letter given. <Panadeine Forte>(medication) 1-2 qid prn, 20 tabs. Cert unfit to [10/02](date).",
                "@pre",
                "@pre [10/02/2025](date) 11:05   [NI](person)",
                "@pre   S: Cast on. Pain 3/10. s/b [KM](person) [7/2](date) - happy with progress. Wants to return to",
                "@pre      light duties; supervisor [Dave Rossi](person) [0491 574 118](phone) says can offer store work.",
                "@pre   P: Cert suitable duties, no use R hand, 4 h/day from [17/02](date). Review 2/52.",
                "@pre      Pt email for copies: [hsgill75@example.com](email)",
                "@pre " + "-" * 88,
                r"@pre Printed by: [rec03](employee_id)    [C:\MD3\Reports\Gill_004417_notes.rtf](file_path)",
            ]
        ),
        Page(
            [
                "@h2 Certified copy of photographic identification",
                "@box 120,120,340,215",
                "@text 136,132,10,Bold | DRIVER LICENCE",
                "@text 330,132,9,Bold | NEW SOUTH WALES",
                "@photo 136,155,80,100",
                "@text 230,158,9,Bold | [HARPREET SINGH GILL](person)",
                "@text 230,174,8,Regular | [9/41 CORRIMAL ST](street_address)",
                "@text 230,185,8,Regular | [WOLLONGONG](locality) NSW [2500](postcode)",
                "@text 230,204,7,Regular | Licence No.",
                "@text 230,213,10,Bold | [18734562](driver_licence)",
                "@text 340,204,7,Regular | Date of birth",
                "@text 340,213,10,Bold | [30 JUN 1975](date_of_birth)",
                "@text 230,232,7,Regular | Card No.",
                "@text 230,241,9,Regular | [2091847365](driver_licence)",
                "@text 340,232,7,Regular | Expiry",
                "@text 340,241,9,Regular | [30 JUN 2029](date)",
                "@text 136,265,7,Regular | Class C",
                "@sig 230,290",
                "@gap 300",
                "I certify that this is a true copy of the original document sighted by me.",
                "@gap 4",
                "@field Name of authorised witness | Ms [Margaret O'Neill](person) ;; Qualification | Justice of the Peace",
                "@field JP registration number | [182736](other_id) ;; Date | [14/02/2025](date)",
                "@sig",
                "@stamp 330,520,-10 | JUSTICE OF THE PEACE | [MARGARET O'NEILL](person) | Reg. No. [182736](other_id) | NSW",
                "@hand 70,660,-3 | pls redact before forwarding | - [Tom](person), ext [4412](other_id)",
            ]
        ),
    ],
)

_tfn_dunn = _tfn("45673812")
_abn_cafe = _abn("146572093")
_acn_cafe = _acn("14657209")

CTP = DocSpec(
    slug="hard-03-ctp-claim-bundle",
    tier="hard",
    title="Motor accident (CTP) claim bundle",
    challenges=[
        "150 dpi colour scan on yellowed paper, faded, tilted about 2.5 degrees, copier edges and punch holes.",
        "Page 2 was scanned upside down; page 3 has a coffee ring over the text.",
        "Vehicle registrations, a VIN, a police event number and a court file number.",
        "Bank BSB and account, tax file number, Centrelink CRN, employer ABN and ACN, employee number.",
        "Claimant appears as 'Jessica Mae Dunn', 'J M DUNN', 'Ms Dunn' and the nickname 'Jess'.",
        "QR code encoding the claimant's details; handwritten notes naming her mother.",
    ],
    seed=33,
    size=9.5,
    scan=Scan(
        dpi=150,
        colour=True,
        skew=2.5,
        blur=1.0,
        noise=12.0,
        jpeg=40,
        paper=(244, 236, 212),
        fade=0.25,
        shadow=0.3,
        speckle=40,
        copier=True,
        coffee=(2,),
    ),
    pages=[
        Page(
            [
                "@text 56,40,15,Bold | [MORELAND QUINN LAWYERS](organisation)",
                "@text 56,60,8.5,Regular | [Level 4, 33 Bridge Street](street_address), [Sydney](locality) NSW [2000](postcode)   "
                "T [(02) 5550 7400](phone)   [mail@morelandquinn.example](email)",
                "@gap 10",
                "@rule",
                "[2 July 2025](date)",
                "",
                "Claims Officer, [Austral Mutual CTP](organisation)",
                "[Locked Bag 4410](street_address), [Parramatta](locality) NSW [2124](postcode)",
                "Your claim no: [AMC-25-0661930](reference)      Our ref: [MQ/JD/25-2207](reference)",
                "",
                "Dear Claims Officer,",
                "@bold Re: [Jessica Mae Dunn](person) v [Raymond Kostas](person), District Court of NSW proceedings "
                "[2025/00187342](court_file)",
                "",
                "We act for the plaintiff, Ms [Dunn](person) (born [15/02/1988](date_of_birth)), who was injured on [21 March "
                "2025](date) when her car, registration [CXR45L](vehicle_reg), was struck in [Ryde](locality) by a white Toyota "
                "HiLux, registration [DK82QM](vehicle_reg), VIN [MR0HA3CD100912347](vin), driven by the defendant, Mr "
                "[Kostas](person). The accident was reported to police, event number [E412389765](reference). An independent "
                "witness, Mr [Tomasz Wieczorek](person), can be contacted on [0491 573 087](phone).",
                "",
                "Ms [Dunn](person) suffered a <fractured left clavicle>(clinical) and a <whiplash-associated disorder, grade "
                "II>(clinical). She has been treated by Dr [Lauren Pike](person) at [Gladesville Family Doctors](organisation) and "
                "by physiotherapist Mr [Arjun Mehra](person).",
                "",
                "We enclose her application for weekly payments, with her bank details, and her physiotherapist's progress report. "
                "Please direct correspondence to the writer, [Joanna Dimitriou](person), at "
                "[jdimitriou@morelandquinn.example](email) or on [(02) 5550 7415](phone). Ms [Dunn](person) may be contacted at "
                "[jess.dunn88@example.com](email).",
                "",
                "Yours faithfully,",
                "@sig",
                "[Joanna Dimitriou](person)",
                "Senior Associate",
                "@gap 20",
                f"@small [Moreland Quinn Lawyers](organisation)   ABN [{_abn('208813374')}](abn)   Liability limited by a scheme "
                "approved under professional standards legislation.",
                "@qr 480,730,60 | Claim [AMC-25-0661930](reference) [DUNN J M](person) DOB [15/02/1988](date_of_birth)",
            ]
        ),
        Page(
            [
                "@h1 Application for weekly payments",
                "@small <Part 2 - Payment and employment details. Complete in block letters.>(boilerplate)",
                "@field Claimant | ~[Jessica Mae Dunn](person) ;; Claim number | ~[AMC-25-0661930](reference)",
                "@field Postal address | ~[Unit 7, 19 Tennyson Road](street_address), [Gladesville](locality) [2111](postcode)",
                "@field Mobile | ~[0491 574 632](phone) ;; Email | ~[jess.dunn88@example.com](email)",
                f"@field Tax file number | ~[{_tfn_dunn}](tfn) ;; Centrelink CRN | ~[302 418 557K](crn)",
                "@h3 Bank account for payments",
                "@field Account name | ~[J M DUNN](person) ;; BSB | ~[062-948](bank_account) ;; Account number | ~[1048 2276](bank_account)",
                "@h3 Pre-injury employment",
                f"@field Employer | ~[Harbour City Cafes Pty Ltd](organisation) ;; ABN | ~[{_abn_cafe}](abn)",
                "@field Employer address | ~[140 Darling Street](street_address), [Balmain](locality) NSW [2041](postcode)",
                "@field Position | ~Barista / shift supervisor ;; Employee no. | ~[HCC-0457](employee_id)",
                "@field Average gross weekly earnings | ~$1,140.00 ;; Hours per week | ~38",
                "@field Manager | ~[Sophie Leclerc](person) ;; Manager phone | ~[(02) 5550 7522](phone)",
                "@h3 Declaration",
                "<I declare that the information in this application is true and correct. I understand that giving false or "
                "misleading information is a serious offence.>(boilerplate)",
                "Signature of claimant",
                "@sig",
                "@field Date | ~[30/06/2025](date)",
                f"@hand 340,92,7 | ACN per ABN lookup | [{_acn_cafe}](acn)",
            ],
            upside_down=True,
        ),
        Page(
            [
                "@text 56,40,13,Bold | [Gladesville Sports & Spinal Physiotherapy](organisation)",
                "@text 56,58,8.5,Regular | [2/301 Victoria Road](street_address), [Gladesville](locality) NSW [2111](postcode)   "
                "[(02) 5550 7690](phone)",
                "@gap 8",
                "@rule",
                "@h2 Physiotherapy progress report",
                "@table 120,363",
                "Patient | Ms [Jessica Dunn](person) ('[Jess](person)')",
                "Date of birth | [15/02/1988](date_of_birth)",
                "Claim | [AMC-25-0661930](reference)",
                "Referring doctor | Dr [Lauren Pike](person)",
                "Sessions to date | 14 (since [2 April 2025](date))",
                "@end",
                "@h3 Progress",
                "[Jess](person) reports neck pain of 4 out of 10, down from 8 at the initial assessment, and occasional headaches. "
                "Left shoulder elevation has improved to 150 degrees. She still has difficulty lifting milk crates and with sustained "
                "overhead work, which limits her return to barista duties at [Harbour City Cafes](organisation).",
                "@h3 Objective",
                "Cervical rotation 60 degrees left, 70 degrees right. <Spurling's test>(eponym) negative. Shoulder flexion 150 "
                "degrees, abduction 140 degrees. Grip strength left 22 kg, right 30 kg.",
                "@h3 Plan",
                "Continue weekly sessions for six weeks with a graded strengthening program. A graduated return to work is "
                "recommended, starting with three four-hour shifts a week from [21 July 2025](date).",
                "@gap 6",
                "[Arjun Mehra](person)",
                "Physiotherapist   AHPRA [PHY0001987654](ahpra)",
                "@sig",
                "@hand 290,600,-8 | Spoke to [Jess](person)'s mum [Carol](person) | [(02) 5550 7431](phone) - Wed pm",
            ]
        ),
    ],
)

DOCS: list[DocSpec] = [GP_REFERRAL, RADIOLOGY, SOLICITOR_LETTER, IME_REPORT, DISCHARGE, CERTIFICATE, PSYCH, FAX, CTP]
