# ADR 0006 — Whole addresses: one grammar, one join, one fake per address

- Status: accepted
- Date: 2026-10-24

## Context

Until now each part of an address followed its own type's action. The client profile replaced the street line and
kept the suburb and postcode (`locality: keep`, `postcode: keep`, the client's answer to Q4), and the other profiles
replaced the three parts separately. That left four problems:

* a kept suburb and postcode beside a replaced street line still narrow an address to a place, and a matter with
  three suburbs shared by nine addresses gives up to five of them away;
* the street line, the suburb and the postcode were faked independently, so a fake address was not an address (a
  suburb in any state, a postcode unrelated to it, the state slot blank);
* the rule detectors cut a place from a street with regexes, which anchored 'Act 2002' lines, 'WorkCover NSW 2006' and
  'Orthopaedic Surgeon Penrith NSW 2750', and replaced 'RSD 3 months', '6 Minute Walk Test' and '3 Local Court
  convictions';
* the fake suburb pool had 109 entries, collided across documents and ran out (a job with four suburbs in the ACT was
  quarantined with no output).

On 2026-10-22 the owner decided to replace every postal or street address as one unit and to keep places mentioned on
their own. This departs from the client's literal answer to Q4, and `addresses: parts` restores it.

## Decision

**One address is one unit, replaced whole, in every profile** (`addresses: whole`, the default; the validator refuses
it when `street_address` is `keep`). The unit runs from its first word (building or property name, floor or unit words,
house number, corner form, street, or box kind and number) to its last (suburb, state, postcode, country).
Structure words (unit, level, suite, shop, lot and floor words, box words, `Cnr`, `&`, `DX`, `Australia`, the state,
punctuation) stay; every other word of the unit is identifying and is replaced (fail closed). Court, hospital and public
body names are never part of a unit (the organisation rule replaces them), nor are citations, Traditional Place lines or
care-of lines. Overseas addresses are replaced whole too, with the country word kept, no foreign gazetteer and no
state inferred.

**One shared grammar** (`text/address/`, pure) reads addresses for detection, policy, the fakes, verification and the
benchmark's reporting. A place anchors only when its postcode lies in the state's Australia Post ranges and its suburb
is a locality of the state in the suburb list. This one rule removes the 'Act <year>' anchors. Months, weekdays,
measure words, clinical and legal-record nouns and box kinds in the wrong case or context never make a street.

**One suburb list** (`resources/gazetteer/`, 15,707 rows from ABS ASGS Edition 3 data, CC BY 4.0) serves the grammar and
the fakes: the rows flagged `fake_pool` (about 6,900 real localities with their own postcodes and a metro or regional
class) are the fake pool. It ships with its attribution and provenance, verified against `SHA256SUMS` on first use.

**One join** (`policy/addresses.py`) turns the rule detectors' mentions and the grammar's units into whole-address
units: it joins a street line to its place on the same line, in the block below or above, across a page break or in a
form's boxes; retypes a weak locality or postcode span that lies in a unit; makes place-only units of a suburb, state
and postcode with no street line; finds repeats of an address (footers); and records the places on their own with their
positions. Places mentioned on their own stay as their type's action says. A suburb, state and postcode right after a
place-of-birth, injury, accident, incident or declaration phrase stays kept (postcode included); after a court's or
hospital's name it is an address. A street without a house number in running text is kept unless its suburb, state and
postcode follow it directly.

**One fake per address, chosen from its canonical key** under the scope key (see ARCHITECTURE.md §7): the house, unit
and level numbers change digit for digit, the street name and type change, the suburb is a fake-pool locality of the
same state and class with its own postcode, and nothing depends on the rest of the document. `address_state: keep`
(default) writes the original state; `replace` moves all addresses of one state to one other state. An empty choice
widens (class, then state) instead of raising.

**Verification follows the unit.** V2 searches the output's text layer for the unit's targets (compound and whole-token
needles) and accounts for every occurrence by an exempt position (this run's fakes, places recorded as on their own,
tokens of kept mentions); V4 probes the same targets, joins the re-read output again and re-plans a repeat of a found
address with that address's fake; V5 checks the fakes against every identifying word and number of every address.

## Consequences

* A place that is not part of an address stays readable in the client profile, together with its postcode ('postcode
  2750' alone, 'born in Dubbo NSW 2830').
* The bench reports each unit as a street entry and a `<id>.place` locality entry, so both ground-truth parts are
  scored; the oracle modes build the same units from the injected mentions and their fakes equal the end-to-end fakes.
* Rules-only over the five real documents' ground truth (perfect text, no models): 30 of 32 street lines exact and
  31 of 32 wholly inside a unit, 39 of 39 localities exact, 40 units, no unit word on a kept place, a citation or any
  other annotated word, none of the 15 'Act <year>' lines anchored, the claim form's 7 street lines and 7 localities in
  7 units, all 9 PO-box footers whole. The reference parser's figures are the same.
* Whole-token targets can match an ordinary word that equals a street-name or building word of a unit (6 words on 6
  lines of two documents in that regression): V2 holds such a job unless the word is exempt, so look-alike positions
  (boilerplate, a building named in prose) may need exporting from the join.
* Open: a numbered street name split over a wrapped line of running text ('23 Lomandra' / 'Street, Chermside QLD
  4032.') is read as a place alone and the first line stays; two capitalised words before a house number with no
  building word stay (a rule cannot tell them from an untitled name).
