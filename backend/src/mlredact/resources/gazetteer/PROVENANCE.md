# au_localities.csv v2: provenance

Built 2026-10-22 by `build_v2.py` (Python standard library only) from the same official Australian Bureau of Statistics (ABS) open data as v1, pinned by sha256 in the v1 folder's `raw/`. All sources are public reference data; no document content or personal data was used or sent anywhere.

Purpose: one list of real Australian localities for tools 1 and 2. Rows with `fake_pool = 1` are the fake-address pool (a same-state, same-class suburb and that suburb's own postcode). Every row, plus `aliases.csv`, is the parser's gazetteer, and `cross_border.csv` holds the parser's exceptions to the state-range gate.

**What v2 changes.** The v1 files in the parent folder are unchanged.
- v1's 15,389 rows are rebuilt from `raw/` with v1's own code, and the build checks that they equal v1's `au_localities.csv` byte for byte (sha256 `4c08aef8…7e39`). Their 13 columns are unchanged and keep their order. Four columns are appended: `fake_pool`, `display_name`, `source` (`sal`) and `locality_pid_2021` (empty).
- 318 detection-only rows are added for the 283 official 2021 localities that the ABS Suburbs and Localities (SAL) list lacks (`source = official_locality`, `fake_pool = 0`).
- New files: `aliases.csv` (17 aliases), `cross_border.csv` (55 rows), `ATTRIBUTION.txt` and `build_v2.log`.
- Corrections to v1's provenance: canonical ASGS URLs (sections 2 and 3), the 2026 postcode count (section 8), the Mesh Block Counts page link, and the 'Modified' note (section 3).

| Output | Bytes | Rows | sha256 |
|---|---|---|---|
| `au_localities.csv` | 1,554,929 | 15,707 data rows + 1 header row | `76a1c7b51d556f0b706797573a311f980b1c4e73d8b3a31bba23eb3b14b3d455` |
| `aliases.csv` | 2,384 | 17 + 1 header row | `42c35d6c0cd5d5f1fe335574cf3a5e282e5de3237dd8c565c9fb529c904d1850` |
| `cross_border.csv` | 1,070 | 55 + 1 header row | `4ff914805f26324a7b8bfdf1e35fd343a77d909608044c0109cd77ec17c89237` |
| `ATTRIBUTION.txt` | 1,370 | - | `f6b22022f8943c395834c839fc8c504b5dfe4a27347971f8dec5fe8dbd128adb` |
| `build_v2.log` | - | - | `779023c96670305f855f20bdc3499087fbafd20e10ddcea1c7f8b532189c4082` |

- **Format.** UTF-8 without BOM, LF line endings, comma-separated with minimal quoting (no field contains a comma or quote today). The three CSV files are ASCII; `ATTRIBUTION.txt` holds one non-ASCII character (©).
- **Determinism.** Two builds, run from different working folders, gave identical sha256 for all five outputs.
- **Rebuild.** `python -I -B build_v2.py` from this folder takes about one minute. It reads `../raw/` read-only, prints and logs every check, and writes the data files only when every check passes (the log is always written). Exit code 2 means a failed check or input.

## 1. Columns of au_localities.csv

One row per locality and postcode. Columns 1-13 are v1's.

| Column | Meaning |
|---|---|
| `state` | NSW, VIC, QLD, SA, WA, TAS, NT or ACT. SAL rows: from the ABS state code (first digit of the SAL code, checked against the SAL allocation file's STATE_CODE_2021 and STATE_NAME_2021). Official rows: the prefix of the official locality id (`NSW5119` is NSW). |
| `locality` | The ABS name. SAL rows: the SAL name with its disambiguation suffix removed (section 4, step 8); otherwise ABS spelling and case are kept, for example `St Ives`, `Mount Victoria` and `Gulf Of Carpentaria`. Official rows: the official name as the ABS prints it, in upper case (`BARRINGUN`). Show `display_name` instead. |
| `postcode` | The 4-digit Postal Area code, with leading zeros (`0800`). |
| `class` | `metro` when at least half of the locality's 2021 population lives in a Greater Capital City Statistical Area (GCCSA), otherwise `regional`. The ACT's GCCSA is the whole territory, so every ACT locality is `metro`. Official rows: the same rule over the official locality's population-weighted SA2 shares (section 4, step 14). |
| `share` | The share of the locality's population inside this postcode: the ABS population-weighted `RATIO_FROM_TO`, 7 decimals. Shares below 0.01 are the records the ABS flags as "below minimum output size". |
| `primary` | `1` for the locality's largest-share postcode, otherwise `0`. Each locality has exactly one primary row. Ties go to the lower postcode; two occur, both with population 0: Lake Macquarie NSW (2259/2280) and Lake Eppalock VIC (3444/3523/3551). |
| `in_state_range` | `1` when the postcode lies in the state's Australia Post ranges (section 6.3), otherwise `0`. The 55 rows with `0` are real cross-border postcodes and are listed in `cross_border.csv`. |
| `population_2021` | SAL rows: persons usually resident in the whole locality, all postcodes together, on Census night 2021. It is the sum of the ABS Mesh Block counts, which the ABS randomly adjusts for confidentiality. Official rows: empty, because the ABS publishes no population for official localities. |
| `capital_share` | The share of the locality's 2021 population inside the capital-city GCCSA, 4 decimals. |
| `sal_code_2021`, `sal_name_2021` | Source ids: the ABS SAL code, and the ABS name with its suffix. Empty on official rows. |
| `poa_code_2021` | Source id: the ABS Postal Area code (equal to `postcode`). |
| `gccsa_code_2021` | Source id: the GCCSA holding most of the locality's population, for example `1GSYD` or `1RNSW`. |
| `fake_pool` | `1` when the row may be drawn as a fake address: `primary = 1`, `in_state_range = 1`, `population_2021` of 200 or more, and not a reviewed non-address area (section 5). Otherwise `0`, and always `0` on official rows. |
| `display_name` | The locality in display casing. SAL rows: the ABS name with linking words in lower case inside a name, `Mc` followed by a capital, and three acronyms (section 6.4). Official rows: the official name in title case in the ABS style. It differs from `locality` only in case. |
| `source` | `sal` for v1's rows (ABS Suburbs and Localities); `official_locality` for the detection-only rows. |
| `locality_pid_2021` | Source id of an official row: the official locality id in the ABS correspondence (for example `SA210006269`). Empty on SAL rows. |

Sort order: state (ACT, NSW, NT, QLD, SA, TAS, VIC, WA), then locality (case-insensitive), then locality as written, postcode, source (`sal` first), SAL code and official locality id. v1's rows keep their v1 order relative to each other.

## 2. Sources

Accessed on 2026-10-22. Every file is kept unchanged in the v1 folder's `raw/`, and `build_v2.py` refuses to build if a sha256 differs.

| # | Publisher, dataset, edition | File in `raw/` | Official URL | Size (bytes) | sha256 | Date |
|---|---|---|---|---|---|---|
| S1 | ABS (ABS Geospatial Solutions) on data.gov.au: *ASGS Geographic Correspondences (2021) Edition 3*, resource "ASGS Edition 3 (2021) Correspondence Files" | `asgs2021correspondences.zip` | https://data.gov.au/data/dataset/2c79581f-600e-4560-80a8-98adb1922dfc/resource/33d822ba-138e-47ae-a15f-460279c3acc3/download/asgs2021correspondences.zip (dataset page https://data.gov.au/data/dataset/asgs-edition-3-2021-correspondences) | 35,381,784 | `3c4ca9307b86dd9c51626920ddbb64fbe38936fd30f4e2c846eb71569dcb1c60` | Last-Modified 14 Oct 2026 03:05 GMT |
| S1a | Member of S1: SAL 2021 to POA 2021, population weighted | `CG_SAL_2021_POA_2021.xlsx` (inside S1) | as S1 | 667,926 | `ea4b70c538e2e2c6187900fb1b76a880d96649d9b9716ba24e1e9b7566491584` | zip entry 30 Jan 2026 |
| S1b | Member of S1: SAL 2021 to SA2 2021, population weighted | `CG_SAL_2021_SA2_2021.csv` (inside S1) | as S1 | 1,031,342 | `9874aed1b5fc64b59e33bdedc6f42a33a57a4dcc173f7f45b4801bf40f961e5d` | zip entry 3 Feb 2026 |
| S1c | Member of S1 (new in v2): official 2021 localities to POA 2021, population weighted | `CG_LOCALITY_2021_POA_2021.xlsx` (inside S1) | as S1 | 839,665 | `b8490b9bbf3e104cd48ced3bed91b1f9bbb52fc90b495b180b1e91a5fcfebc9b` | zip entry 5 Aug 2022 |
| S1d | Member of S1 (new in v2): official 2021 localities to SA2 2021, population weighted | `CG_LOCALITY_2021_SA2_2021.csv` (inside S1) | as S1 | 1,192,157 | `13196ba11532a09350556d970420c2d5d170610d2f586f7e8e3109161f8c0a8e` | zip entry 23 Sep 2022 |
| S1e | Member of S1, read only for the Edition 4 notes in the log: SAL 2021 to SAL 2026 | `CG_SAL_2021_SAL_2026.csv` (inside S1) | as S1 | 784,978 | `1dff075c8456ea72aa69606ce1930dd6a20270f5d1289dd63f562590edf952b1` | zip entry 8 Oct 2026 |
| S1f | Member of S1, read only for the Edition 4 notes in the log: POA 2021 to POA 2026 | `CG_POA_2021_POA_2026.csv` (inside S1) | as S1 | 96,147 | `52746d435c473cbf06918acd95c2cf3df5c3e2f2f50880674da250f26d2107aa` | zip entry 15 Sep 2026 |
| S2 | ABS, ASGS Edition 3 (July 2021 - June 2026), allocation file for Suburbs and Localities: Mesh Block to SAL, with state (SAL structure released 6 Oct 2021) | `SAL_2021_AUST.xlsx` | https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/allocation-files/SAL_2021_AUST.xlsx | 20,446,326 | `e6be7af2fc4670375d49dfd9d83c98d48eea5049d038327d8eea94b598c7c9a3` | Last-Modified 15 Sep 2021 |
| S3 | ABS, ASGS Edition 3, allocation file for Statistical Areas Level 2: SA2 to GCCSA | `SA2_2021_AUST.xlsx` | .../allocation-files/SA2_2021_AUST.xlsx (same base as S2) | 250,022 | `953cebfb02e55c7c36ed7f5c202d88a287528d9188d2f2f7d2b32b95834115b4` | Last-Modified 24 Jun 2021 |
| S4 | ABS, ASGS Edition 3, allocation file for Greater Capital City Statistical Areas: codes and names | `GCCSA_2021_AUST.xlsx` | .../allocation-files/GCCSA_2021_AUST.xlsx (same base as S2) | 13,523 | `febd301dae0d3792fd429d7c9fef68c1062c2b2406d2c92398e9a3b4d753f187` | Last-Modified 24 Jun 2021 |
| S5 | ABS, *Census of Population and Housing: Mesh Block Counts, 2021*, released 28 Jun 2022. Published as `Mesh Block Counts, 2021.xlsx` and saved under a name without spaces. | `Mesh_Block_Counts_2021.xlsx` | https://www.abs.gov.au/census/guide-census-data/mesh-block-counts/2021/Mesh%20Block%20Counts%2C%202021.xlsx (release page https://www.abs.gov.au/census/guide-census-data/mesh-block-counts/2021) | 15,147,466 | `1ab520c2ff0e07ca88729258c30ee9f7a7133df113a81bac94812582f3b540a3` | Last-Modified 20 Jun 2022 |
| S6 | Australia Post, "Postcodes turn 50!" (Philatelic Team, 1 July 2017), "Current postal ranges for the states and territories". Only these ranges are used, as facts, for the range check and `in_state_range`; no text is reused. | `pages/auspost_postcodes_turn_50.html` | https://collectables.auspost.com.au/community-and-events/articles/postcodes-turn-50 | 116,844 | `6612102fe0308509f1ddd48f5f8d7331a479551ccfd8e00cd14fe8bba942fa74` | page as fetched |

The ABS URLs above are the canonical forms, as the saved pages' own canonical links give them. v1's citations used another form (`.../australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/...`), which v2 corrects. The Mesh Block Counts page is cited by its release path `/2021`: the saved copy's canonical link is `/latest-release`, which will move to the next release.

Reference pages saved as evidence in `raw/pages/` (sha256 of each copy as fetched):

| Copy | What it shows | sha256 |
|---|---|---|
| `asgs3_sal_methodology.html` | ABS SAL design criteria: remainder areas, special codes, name suffixes, coding. https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/non-abs-structures/suburbs-and-localities | `8e7ce38bfafa95cd0632ad211edb8b7b47bb5c74d32e4e604456925d00f72b03` |
| `asgs3_allocation_files.html` | ABS Edition 3 allocation-file download page. https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/allocation-files | `ab68643a3e1f75c727dd2660adb1c6898cc94e43259a17aa4ba481fc54049582` |
| `asgs3_correspondences.html` | ABS Edition 3 correspondence page. It offers no SAL-POA file; the data.gov.au zip (S1) does. https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/correspondences | `3e2eca81e18327cefb8e0847f0ad7400639882c968dc622a9be66d529da2dfce` |
| `abs_copyright.html` | ABS licence and attribution statement (section 3). https://www.abs.gov.au/website-privacy-copyright-and-disclaimer | `9a4c92f0b70935aa631c0499490dbc6ce5f5c7c3fa7b4cac9703ccdf97f04c66` |
| `abs_how_cite.html` | ABS "How to cite ABS sources". The "Attributing ABS material" link redirects here. https://www.abs.gov.au/how-cite-abs-sources | `f165c542c6e8f66fdf6245d7611d306f79c27d77e194b8f0286bf0c73cb292d0` |
| `abs_mesh_block_counts_2021.html` | Mesh Block Counts release page; its links sit under `/census/guide-census-data/mesh-block-counts/2021/` | `d573543495275ad73dc93b5c470075854dbd5e6a9935e9074062a1aacb8d6b09` |
| `dga_asgs3_correspondences.json`, `dga_asgs3_correspondences_page.html` | data.gov.au record for S1: licence, publisher, modified date | `6faa048b50590140195e0f4bd54a19f68e278130100dacc2932015bcabc5c8f3`, `c4f1f3c3f899e7e7517ce87a29ffe715fe03838d8fb7b2c2641f1fb964b341af` |
| `asgs_latest_release.html` | ASGS Edition 4 (July 2026 - June 2031) is now the latest release (section 8). https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/latest-release | `966fc7b0ab8235c602562c7aad08cb0134e6d4fe505f74d802ce049aa7dc310d` |

The ratio method, quality indicators and the below-minimum and null flags are documented by the ABS in `Correspondence Documentation.pdf` and `Machine Readable CSV Format Correspondence Documentation.pdf`, inside S1. The flag codes v2 relies on: 0 a normal record; 1 below minimum output size; 2 the "to" unit is null; 3 the "from" unit is null; 4 below minimum output size and the "to" unit is null; 5 below minimum output size and the "from" unit is null.

## 3. Licence and attribution

- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/ (legal code https://creativecommons.org/licenses/by/4.0/legalcode).
  - The ABS copyright page (https://www.abs.gov.au/website-privacy-copyright-and-disclaimer) states: "All material presented on this website is provided under a Creative Commons Attribution 4.0 International licence, with the exception of:".
  - It lists these exceptions: the Coat of Arms; the ABS logo; trade-marked material; unit record data (microdata); content supplied by third parties; sub-brands; and certain artwork and branding.
  - None of the exceptions applies here. The Mesh Block counts are published aggregates, not microdata. SAL is the ABS's own Mesh Block approximation of official localities, published as ABS material and not labelled as third-party content. The official-locality correspondences (S1c, S1d) are ABS correspondence files inside the same CC BY 4.0 dataset.
  - The data.gov.au record for S1 states licence "Creative Commons Attribution 4.0 International" (http://creativecommons.org/licenses/by/4.0), publisher "ABS Geospatial Solutions".
- **What the ABS asks for (verbatim from its copyright page):** "Material obtained from this website is to be attributed to this department. If you use ABS material, there are certain obligations you must fulfil." Its citation guide ("How to cite ABS sources") gives the form: Australian Bureau of Statistics (Reference period) 'Title' [data set], Title of issue [URL], accessed DD Month YYYY.
- **Attribution text to ship wherever the list or a file derived from it is shipped.** It is `ATTRIBUTION.txt`, reproduced exactly below. The first paragraph is v1's text with the two new correspondences and the extended 'Modified' note. The links that follow are the ones CC BY 4.0 section 3(a)(1)(A)(v) asks for: a URI or hyperlink to the licensed material, to the extent reasonably practicable, plus the licence.

```text
Source: Australian Bureau of Statistics (ABS), Australian Statistical Geography Standard (ASGS) Edition 3, July 2021 - June 2026 (allocation files SAL_2021_AUST, SA2_2021_AUST, GCCSA_2021_AUST; correspondences CG_SAL_2021_POA_2021, CG_SAL_2021_SA2_2021, CG_LOCALITY_2021_POA_2021, CG_LOCALITY_2021_SA2_2021) and Census of Population and Housing: Mesh Block Counts, 2021, accessed 22 October 2026. © Commonwealth of Australia, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Modified: ABS name suffixes removed, non-places and Other Territories removed, tables combined, official localities missing from the Suburbs and Localities list added, and derived columns added (population, class, capital share, primary, range flag, fake_pool, display_name, source); the alias and cross-border lists are derived from the same data.

Licensed material:
- ASGS Edition 3 allocation files: https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/allocation-files
- ASGS Geographic Correspondences (2021) Edition 3: https://data.gov.au/data/dataset/asgs-edition-3-2021-correspondences
- Census of Population and Housing: Mesh Block Counts, 2021: https://www.abs.gov.au/census/guide-census-data/mesh-block-counts/2021
Licence: https://creativecommons.org/licenses/by/4.0/
```

- **Full citations, in the ABS form:**
  - Australian Bureau of Statistics (July 2021 - June 2026) 'SAL_2021_AUST', 'SA2_2021_AUST' and 'GCCSA_2021_AUST' [data sets], *Australian Statistical Geography Standard (ASGS) Edition 3* [https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/allocation-files], accessed 22 October 2026.
  - Australian Bureau of Statistics (2021) 'CG_SAL_2021_POA_2021', 'CG_SAL_2021_SA2_2021', 'CG_LOCALITY_2021_POA_2021' and 'CG_LOCALITY_2021_SA2_2021' [data sets], *ASGS Geographic Correspondences (2021) Edition 3* [https://data.gov.au/data/dataset/asgs-edition-3-2021-correspondences], accessed 22 October 2026.
  - Australian Bureau of Statistics (2021) 'Mesh Block Counts, 2021' [data set], *Census of Population and Housing: Mesh Block Counts, 2021* [https://www.abs.gov.au/census/guide-census-data/mesh-block-counts/2021], accessed 22 October 2026.
- **Australia Post ranges (S6):** these are facts, used only for a check, a flag and the cross-border list; nothing from that page is redistributed. Its own copyright terms are not open, so no part of the page should be copied into the tools.

## 4. Transformations

Steps 1-10 are v1's and run unchanged inside `build_v2.py`; steps 11-16 are new.

1. **Verify the inputs.** Each sha256 is checked against the values pinned in `build_v2.py` (the same as v1's), and the build stops on a mismatch.
2. **Read the workbooks as data.**
   - An .xlsx file is a zip of XML parts. The build reads those parts with expat; it runs no macros, formulas or external links.
   - XML with a DTD or entity declaration is refused, and zip members are size-capped.
   - Shared strings are loaded only for the cells used. The ABS SAL file has 2.9 million shared strings that are not de-duplicated, so this keeps memory low.
3. **Population per Mesh Block (S5).** The persons count is read from tables 1-9: 368,285 Mesh Blocks, with each code checked for 11 digits, no repeats and state = first digit.
4. **SAL list (S2).** Each of the 368,286 Mesh Blocks is allocated once. Its persons count is added to its SAL, giving 15,353 SALs.
   - One Mesh Block has no count: `ZZZZZZZZZZZ`, Outside Australia.
   - Every counted Mesh Block has a SAL.
   - Every SAL's state code equals the first digit of its code, and its state name matches.
5. **Postcodes (S1a).** Each SAL row gives a postcode and a population share.
   - Checks: header exact; SAL sets in S1a, S1b and S2 identical; SAL names identical to S2; no null records; shares sum to 1 per SAL (largest deviation 0.000001); no repeated postcode per SAL; postcodes are 4 digits.
   - The ABS below-minimum flag agrees with share < 0.01 on every row.
   - Ratios are kept as decimals and rounded half-even to 7 places.
6. **Class (S1b + S3 + S4).** For each SAL, the population shares of its SA2s are summed by the GCCSA of each SA2.
   - `capital_share` is the sum over the 8 capital-city GCCSAs: 1GSYD Greater Sydney, 2GMEL Greater Melbourne, 3GBRI Greater Brisbane, 4GADE Greater Adelaide, 5GPER Greater Perth, 6GHOB Greater Hobart, 7GDAR Greater Darwin and 8ACTE Australian Capital Territory. These codes and names are checked against S4.
   - `class` is `metro` when `capital_share` is 0.5 or more.
   - Checks: each SAL's SA2 shares sum to 1 (largest deviation 0.0000002), and every SA2 lies in the SAL's own state.
7. **Exclusions** (section 5).
8. **Name clean-up.**
   - Whitespace is normalised.
   - A trailing ABS suffix is removed only when it is a state abbreviation (`(NSW)`, `(Vic.)`, `(Qld)`, `(SA)`, `(WA)`, `(Tas.)`, `(NT)`, `(ACT)`) or an LGA name plus a state abbreviation (`(Mackay - Qld)`). In both cases it must name the SAL's own state.
   - Among the kept localities, 2,274 names carry a suffix, in 131 distinct forms: 8 state-only and 123 "LGA - state". Every one names the SAL's own state. Any other parenthesis would stop the build; none occurs.
   - The original name is kept in `sal_name_2021`.
9. **Flags.** `primary` marks the largest-share postcode per locality. `in_state_range` compares the postcode with the Australia Post ranges for the row's state.
10. **v1 table.** The rows are sorted and serialised exactly as v1 writes them, in memory. The build checks the bytes against v1's pinned sha256 (`4c08aef8…7e39`, 15,389 rows) and stops on a difference.
11. **Display casing.** `display_name` starts from the ABS name. Inside a name (never as its first word) the words of, on, by, and, at, in, the and upon are written in lower case. 'Mc' followed by a lower-case letter gets a capital. Three acronyms the ABS writes as words get capitals (section 6.4). `locality` keeps the ABS name.
12. **Fake pool.** `fake_pool = 1` when `primary = 1`, `in_state_range = 1`, `population_2021 >= 200` and the SAL is not in the reviewed non-address list (section 5). Pool candidates whose names hold a review word are decided by hand, one by one, in `build_v2.py`. The review words are the ones the independent check named (Military, National Park, State Forest, Reservoir, Harbour, Gulf, Desert, Island, Islands) plus defence, airport and institution words (Airport, Base, Barracks, HMAS, RAAF, Army, Navy, Naval, Defence, Training Area, University, Campus, Prison, Correctional, Gaol, Hospital). A new candidate with such a word stops the build until it is decided.
13. **Official localities (S1c).** From the official-locality correspondence, every official 2021 locality of the 8 states whose name (upper case, whitespace normalised) is not a SAL name of its state gets one row per postcode.
    - Only records with both a locality and a postcode count (flags 0 and 1). Flag 2 and 4 records, the parts of a locality outside every Postal Area (mostly sea), are left out. For every added locality the ratios, null records included, sum to 1.
    - `share` is the ABS ratio rounded half-even to 7 places, and `primary` marks the largest share (ties to the lower postcode). `in_state_range` is set as for SAL rows.
    - The S1c header is checked exactly. Each id has one name, the same in S1c and S1d. Every record without a locality id is a null "from" record for 9494, 9797 or ZZZZ.
    - Two placeholders named "UNNAMED LOCALITY" are left out (section 5).
14. **Class of official localities (S1d + S3).** The population-weighted SA2 shares of the locality's records that have an SA2 (flags 0 and 1) are summed by GCCSA and divided by their total, so a part with no SA2 (a null record) does not count. `capital_share`, `class` and `gccsa_code_2021` then follow the SAL rule. Every SA2 must lie in the locality's own state. All 283 added localities lie wholly in one GCCSA: 13 are metro and 270 regional.
15. **Aliases.** '<Town> City' and '<Town> Central' localities whose '<Town>' is not itself a locality of the state are derived exactly as the reference parser derives them (17), and each is decided in `build_v2.py`: 15 kept, 2 rejected. Two hand-made aliases for Canberra are added. Each alias maps to the primary row of the ABS locality covering that centre (section 6.5).
16. **Cross-border list, attribution and write.** `cross_border.csv` lists every row with `in_state_range = 0`. The rows are sorted and the files are written with fixed formats, each moved into place atomically, after every check has passed.

Localities that share a name across states stay separate rows: 892 SAL names occur in two or more states. Within a state, 138 SAL names belong to two or more SALs that the ABS told apart only by an LGA suffix; these also stay separate, each with its own `sal_code_2021`.

## 5. Exclusions and drops

**41 of the 15,353 SALs are excluded (v1, unchanged).**

*ABS special purpose codes (19).* The ABS says: "Non-spatial special purpose codes are included as balancing items. Mesh Blocks allocated to these codes are not part of legally designated Suburbs and Localities."
- `x9494` "No usual address (state)", one per state (×9, POA 9494).
- `x9797` "Migratory - Offshore - Shipping (state)", one per state (×9, POA 9797).
- `ZZZZZ` "Outside Australia" (POA ZZZZ).
- Postcodes 9494 and 9797 sit inside Queensland's 9000-9999 range, so they are removed by code and name, never by range.

*ABS-created remainder areas (17).* The ABS says: "In ASGS Edition 3, the area of South Australia that is not covered by official localities has been assigned to the Suburbs and Localities SA Remainder. Similarly, the area of the ACT that is not currently covered by official localities has been defined as ACT Remainder." These are not address localities.
- 41285 SA Remainder (population 0).
- 80001-80016 "ACT Remainder - Belconnen, Booth, Canberra Central, Coree, Cotter River, Gungahlin, Hall, Jerrabomberra, Kowen, Majura, Molonglo Valley, Paddys River, Rendezvous Creek, Stromlo, Tuggeranong, Weston Creek". Together they hold 613 people.

*Other Territories (5).*

| SAL | Postcode | Population |
|---|---|---|
| 90001 Christmas Island | 6798 | 1,699 |
| 90002 Home Island | 6799 | 446 |
| 90003 Jervis Bay | 2540 | 308 |
| 90004 Norfolk Island | 2899 | 2,185 |
| 90005 West Island | 6799 | 147 |

- In ASGS these form the statistical "state" Other Territories, which has no address state word.
- Their postcodes (2540, 2899, 6798, 6799) lie inside the NSW and WA ranges. Australia Post says the external territories "have postcodes in the states that coordinate their mail deliveries".
- Listing them under OT would break the same-state rule. Listing them under NSW or WA would need a postal-state mapping that these open sources do not give.
- For the same reason the 5 official localities with an `OT` id are not added.

**Official localities left out (2).** S1c holds 15,602 official localities: 15,597 in the 8 states and 5 in the Other Territories. 285 of the 15,597 are not SAL names of their state, the same 285 the independent check listed. Two are placeholders, not names anyone writes in an address, and are left out:
- QLD "UNNAMED LOCALITY" `QLD3384`, postcode 4892;
- QLD "UNNAMED LOCALITY" `QLD3395`, postcodes 4581 and 4655.

**Dropped from the fake pool (4), kept for detection.** These SALs pass the pool rule but are clear non-address areas:

| SAL | Locality | Postcode | Population | Class | Reason |
|---|---|---|---|---|---|
| 13566 | Singleton Military Area NSW | 2331 | 588 | regional | Army training area; not a civilian address area. Named in the independent check. |
| 50529 | Garden Island WA | 6168 | 772 | metro | Naval base island (Fleet Base West, HMAS Stirling); no civilian addresses. Found by the word "Island". |
| 21195 | Hmas Cerberus VIC (display HMAS Cerberus) | 3920 | 1,122 | metro | Naval training base; the locality is the base and bears its name. Found by the word "HMAS". |
| 10883 | Charles Sturt University NSW | 2678 | 433 | regional | University campus; the locality is the campus and bears its name. Found by the word "University". |

The other 41 pool candidates with a review word are kept: residential suburbs or towns (Coffs Harbour, Secret Harbour, Reservoir, Airport West, Desert Springs, Mackay Harbour, Hope Island, Boyne Island, Clear Island Waters, St Huberts Island, Bend Of Islands, Garden Island Creek, and Boat Harbour in NSW and TAS), inhabited islands with street addresses (Dangar, Lord Howe, Mitchells, Oxley, Palmers, Scotland, Woodford, Coochiemudlo, Karragarra, Lamb, Macleay, Russell, Hindmarsh and Raymond Islands), Torres Strait Islander or Aboriginal island communities (Badu, Erub, Iama, Keriri, Mabuiag, Masig, Mer, Moa, Saibai, Thursday and Palm Islands; Wellesley Islands, which holds Mornington Island's community), and Tiwi Islands NT (the remote parts outside the three community localities, 202 residents). Each is listed with its reason in `build_v2.py` and `build_v2.log`.

**Kept, but worth knowing:**
- 683 SAL localities have population 0, for example national parks, industrial areas and water bodies (Port Stephens, Lake Eppalock, Gulf Of Carpentaria). They stay for detection and are never in the pool.
- Some localities that hold a defence base have place names and civilian or married-quarters addresses, so they stay in the pool: for example Puckapunyal VIC (1,108), Kapooka NSW (395), Tindal NT (850), Bandiana VIC (853), Edinburgh SA (371), Amberley QLD (618) and Williamtown NSW (769). They hold no review word, so the build does not flag them.
- The official rows include water bodies and parks that are official localities (Botany Bay, Middle Harbour, North Harbour, Pittwater, Jervis Bay NSW, Darwin Harbour, Beagle Gulf, Francois Peron National Park, Simpson Desert, Strzelecki Desert). They are detection-only. NSW's "Jervis Bay" (the bay, `NSW2046`, 2540) is not the Jervis Bay Territory.

## 6. Checks (output of the final build)

Every line of `build_v2.log` marked `[ok]` is a hard check. The v1 part of the log is identical to v1's `scripts/build.log`, apart from its last line's label and line endings.

### 6.1 Rows per state

| State | SAL rows | SAL localities | Official rows | Official localities | All rows | All localities |
|---|---|---|---|---|---|---|
| ACT | 120 | 120 | 0 | 0 | 120 | 120 |
| NSW | 4,571 | 4,542 | 51 | 32 | 4,622 | 4,574 |
| NT | 303 | 303 | 3 | 3 | 306 | 306 |
| QLD | 3,248 | 3,233 | 11 | 10 | 3,259 | 3,243 |
| SA | 1,699 | 1,695 | 206 | 198 | 1,905 | 1,893 |
| TAS | 778 | 776 | 1 | 1 | 779 | 777 |
| VIC | 2,965 | 2,944 | 33 | 28 | 2,998 | 2,972 |
| WA | 1,705 | 1,699 | 13 | 11 | 1,718 | 1,710 |
| **Total** | **15,389** | **15,312** | **318** | **283** | **15,707** | **15,595** |

- SAL rows, from v1: 69 localities have several postcodes (32 with two or more shares of at least 0.01), 3,372 are metro and 11,940 regional, 683 have population 0, and 2,638 distinct postcodes occur. Population in the listed localities: 25,360,108 (metro 17,015,588, regional 8,344,520). 40 rows have a share below 0.01.
- Official rows: 21 localities have several postcodes (Botany Bay 7, Middle Harbour 9 and North Harbour 4 among them), and 12 rows have a share below 0.01. They bring no postcode that the SAL rows lack. 13 localities are metro: Bickley Vale, Botany Bay, Currawong Beach, McCarrs Creek, Middle Harbour, North Harbour and Pittwater NSW; Flinders Lakes and Lynford QLD; Fieldstone and Tantaraboo VIC; Oakley and Point Grey WA.
- SAL 'Murputja - Nyapari' SA is one SAL for two official localities. Both are added as official rows, Murputja (`SA210006269`) and Nyapari (`SA210006270`), each 0872 and regional. The SAL row stays.
- One (state, locality, postcode, source) key belongs to two rows: QLD Wooroonooran 4860 `sal`, which the ABS lists as two SALs (33161 Cairns, 33162 Cassowary Coast). No other key repeats, and no official name repeats a SAL name of its state.
- 39 SAL localities lie partly inside a capital-city GCCSA. Their class follows the majority, for example Megalong Valley NSW at 0.68 is metro and Moonee NSW at 0.33 is regional. All 39 are listed in `build_v2.log`.
- Spot checks (v1's, unchanged):

  | Locality | Postcode | Class |
  |---|---|---|
  | Penrith NSW | 2750 | metro |
  | Katoomba NSW | 2780 | metro |
  | Kempsey NSW | 2440 | regional |
  | Sydney NSW | 2000 | metro |
  | Newcastle NSW | 2300 | regional |
  | Richmond VIC | 3121 | metro |
  | Richmond NSW | 2753 | metro |
  | Richmond QLD (two localities) | 4740 and 4822 | regional |
  | Richmond TAS | 7025 | metro |
  | Richmond SA | 5033 | metro |
  | Darwin City NT | 0800 | metro |
  | Alice Springs NT | 0870 | regional |
  | Hobart TAS | 7000 | metro |
  | City ACT | 2601 | metro |
  | Surfers Paradise QLD | 4217 | regional |
  | Jerrabomberra NSW | 2619 | regional |

### 6.2 Fake pool

6,938 SAL localities meet the pool rule; after the 4 drops, 6,934 rows have `fake_pool = 1`.

| State | Metro | Regional | Total |
|---|---|---|---|
| ACT | 106 | none | 106 |
| NSW | 815 | 1,330 | 2,145 |
| NT | 65 | 97 | 162 |
| QLD | 470 | 1,056 | 1,526 |
| SA | 441 | 261 | 702 |
| TAS | 103 | 226 | 329 |
| VIC | 516 | 751 | 1,267 |
| WA | 347 | 350 | 697 |
| **All** | **2,863** | **4,071** | **6,934** |

- Every state and class that holds a locality has a non-empty pool; the ACT has no regional class.
- **Fake postcode rule.** For every 4-digit code in each state's official ranges, the pool of each class offers a same-state postcode that differs at every digit after the digits the state fixes: ACT 2, NSW 2, NT 08, QLD 4, SA 5, TAS 7, VIC 3, WA 6. No code lacks a candidate.

### 6.3 Postcode ranges and cross-border rows

Official ranges come from Australia Post, S6. "Found" is the span of postcodes in the state's in-range rows.

| State | Official ranges | Found (in range) | Rows outside the official ranges (`in_state_range = 0`) |
|---|---|---|---|
| ACT | 0200-0299, 2600-2619, 2900-2920 | 2600-2914 | 4 rows, all 2620 (NSW range): Beard, Hume, Oaks Estate, Tharwa |
| NSW | 1000-2599, 2620-2899, 2921-2999 | 2000-2898 | 20 rows (see below) |
| NT | 0800-0999 | 0800-0886 | 2 rows, 4825 (QLD range): Alpurrurulam, Ranken |
| QLD | 4000-4999, 9000-9999 | 4000-4895 | 1 row, 2406 (NSW range): Mungindi |
| SA | 5000-5999 | 5000-5950 | 16 rows, 0872 (NT range), Anangu Pitjantjatjara Yankunytjatjara lands: 11 SAL rows (Amata, Anangu Pitjantjatjara Yankunytjatjara, Iwantja, Kalka, Kaltjiti, Kanpi, Mimili, Murputja - Nyapari, Pipalyatjara, Pukatja, Watarru) and 5 official rows (Murputja, Nyapari, Umuwa, Watinuma, Yunyarinyi) |
| TAS | 7000-7999 | 7000-7470 | none |
| VIC | 3000-3999, 8000-8999 | 3000-3996 | none |
| WA | 6000-6999 | 6000-6770 | 12 rows, 0872 (NT range), Ngaanyatjarra lands: Gibson Desert North, Gibson Desert South, Irrunytju, Kanpa, Mantamaru, Ngaanyatjarra-Giles, Papulankutja, Patjarr, Tjirrkarli, Tjukurla, Wanarn, Warakurna |

The 20 NSW rows outside the official ranges:
- QLD range: Amosfield 4380, Camp Creek 4385, Cottonvale 4375, Jennings 4383, Maryland 4377, Mingoola 4380, Ruby Creek 4380, Texas 4385, Undercliffe 4380.
- VIC range: Barooga 3644, Lalalty 3644, Bringenbrong 3707, Lake Hume Village 3691.
- ACT range: Bimberi 2611, Brindabella 2611, Cooleman 2611, Uriarra 2611, Springrange 2618, Wallaroo 2618, Jerrabomberra 2619.

- `cross_border.csv` lists these 55 rows: v1's 50, all kept, and the 5 official SA rows. They form 16 (state, postcode) pairs; the official rows add no new pair. Every row is its locality's only postcode, so `primary = 1`.
- Every alias and every official row resolves to a postcode in its state's ranges or a listed cross-border pair.
- Note on 2619: the Australia Post article puts 2619 in the ACT range, but its only locality is Jerrabomberra NSW (population 9,599).
- The official-locality correspondence also links three VIC localities that are already SALs to NSW-range postcodes: Cobberas to 2627 (share 0.12), Tom Groggin to 2627 and 2642 (shares 0.0013 and 0.0012), and Ulupna to 2714 (share 0.000001). These are boundary overlaps in areas with almost no residents, not postal usage. v2 does not add them, so the VIC range gate stays as it was.
- PO-box and large-volume-receiver postcodes (for example NSW 1000-1999, VIC 8000-8999, QLD 9000-9999, ACT 0200-0299, NT 0900-0999) do not occur among these Postal Areas, which are built from where people live. They never appear in the list, so they are never drawn as fakes.

Leading postcode digits shared by every postcode of a state (v1's check, unchanged):

| State | All rows | In-range rows |
|---|---|---|
| ACT | 2 | 2 |
| NSW | (none) | 2 |
| NT | (none) | 08 |
| QLD | (none) | 4 |
| SA | (none) | 5 |
| TAS | 7 | 7 |
| VIC | 3 | 3 |
| WA | (none) | 6 |

### 6.4 Display names

`display_name` differs from the ABS name for 21 SAL localities:
- Linking words (15): Come by Chance NSW, Vale of Clwydd NSW, Gulf of Carpentaria QLD, Port of Brisbane QLD, Prince of Wales QLD, Valley of Lagoons QLD, Bay of Shoals SA, Chain of Ponds SA, Kingston on Murray SA, Chain of Lagoons TAS, Eggs and Bacon Bay TAS, Walls of Jerusalem TAS, Bend of Islands VIC, Bridgewater on Loddon VIC, Woodstock on Loddon VIC.
- 'Mc' followed by a capital (3): Mount McEuen QLD, Mount McIntyre SA, Mount McKenzie SA.
- Acronyms written as words (3, a different kind of casing fault found while checking): MP Creek QLD (ABS 'Mp Creek'), Brownlow KI SA (ABS 'Brownlow Ki'; KI is Kangaroo Island) and HMAS Cerberus VIC (ABS 'Hmas Cerberus').

The first 18 are the casing faults the independent check found; no other name of either kind occurs. Left as the ABS writes them, because the right casing is not certain: Mount Kuring-Gai NSW (often written Mount Kuring-gai), Run-o-Waters NSW, Murray-sunset VIC (often written Murray-Sunset) and Stun'Sail Boom SA.

The title-case rule used for official names gives the same result as `display_name` for all but 8 of the 15,160 distinct (state, SAL name) pairs: the irregular hyphens and apostrophes Ku-ring-gai Chase, Mount Coot-tha, Run-o-Waters, Murray-sunset and Stun'Sail Boom, and the three acronyms. No official name added here has a hyphen or an apostrophe. Examples of official display names: McCarrs Creek, McDouall Peak, De Rose Hill, Grawin Opal Fields.

### 6.5 Aliases

`aliases.csv` (alias, state, locality, postcode, reason) holds 17 names that people write and that the ABS list lacks or names differently. Each maps to the primary row of the ABS locality covering that centre, and the reason names that ABS row (SAL code, name, postcode, class). Aliases are detection-only.

| Alias | State | ABS locality (SAL) | Postcode | Class | Why |
|---|---|---|---|---|---|
| Canberra | ACT | City (80038) | 2601 | metro | capital city; the ABS has no Canberra locality |
| Canberra City | ACT | City (80038) | 2601 | metro | common written form of the city centre |
| Darwin | NT | Darwin City (70073) | 0800 | metro | capital city written without 'City' |
| Palmerston | NT | Palmerston City (70216) | 0830 | metro | city written without 'City' |
| Brisbane | QLD | Brisbane City (30379) | 4000 | metro | capital city written without 'City' |
| Bundaberg | QLD | Bundaberg Central (30420) | 4670 | regional | city written without 'Central' |
| Cairns | QLD | Cairns City (30478) | 4870 | regional | city written without 'City' |
| Charters Towers | QLD | Charters Towers City (30585) | 4820 | regional | town written without 'City' |
| Gladstone | QLD | Gladstone Central (31132) | 4680 | regional | city written without 'Central' |
| Logan | QLD | Logan Central (31677) | 4114 | metro | City of Logan, whose centre is Logan Central |
| Rockhampton | QLD | Rockhampton City (32449) | 4700 | regional | city written without 'City' |
| Sapphire | QLD | Sapphire Central (32516) | 4702 | regional | the town of Sapphire lies in Sapphire Central |
| Thuringowa | QLD | Thuringowa Central (32822) | 4817 | regional | former City of Thuringowa, whose centre is Thuringowa Central |
| Toowoomba | QLD | Toowoomba City (32855) | 4350 | regional | city written without 'City' |
| Townsville | QLD | Townsville City (32867) | 4810 | regional | city written without 'City' |
| Ballarat | VIC | Ballarat Central (20111) | 3350 | regional | city written without 'Central' |
| Cockburn | WA | Cockburn Central (50304) | 6164 | metro | City of Cockburn, whose centre is Cockburn Central |

- The capitals: Sydney, Melbourne, Adelaide, Perth and Hobart are SAL names of their states. Brisbane, Darwin and Canberra are aliases. No capital name is missing.
- Derived but rejected (2), because neither is a town: Isis (Isis Central QLD is a rural locality named after the former Isis Shire) and Everard (Everard Central SA is a rural locality). The reference parser's own derivation adds both. The tools should load `aliases.csv` rather than derive aliases.
- Checks: every alias names a SAL locality's primary row; no alias is already a locality name of its state; aliases are unique; every capital is a locality or an alias.

### 6.6 Other checks

- v1's rows keep their 13 fields and their order in v2. v1's own `scripts/check_output.py`, run on v2's SAL rows projected to v1's columns, prints exactly what it prints for v1's file.
- Every field is in its fixed format, every locality has exactly one primary row, the per-locality fields are identical across a locality's rows, and the rows are in the documented order.
- Class agrees with the GCCSA kind and with `capital_share` in every row. `in_state_range` agrees with the ranges in every row.
- `display_name` differs from `locality` only in case; no display name keeps a capitalised linking word or 'Mc' followed by a lower-case letter.
- Official rows: empty population and SAL ids, `fake_pool = 0`, an id with the state's prefix, and an upper-case name of letters, digits, spaces, apostrophes and hyphens.
- Encodings: no BOM, no CR, ASCII CSV files, and `au_localities.csv` reads back to the same rows.
- An independent script (`scripts/verify_v2.py`), which does not import the build, confirms from the shipped files: the v1 projection equals v1's file byte for byte; the official rows equal the independent check's `lost_localities.csv` minus the placeholder, postcodes included; the pool rule, the alias targets, the cross-border list and the duplicate keys.
- **Regression with the reference parser** (`addr/proto`, parser sha256 `c05992cf…`). Run with v2's list:
  - on the case table, it gives results identical to its v1-list runs, case by case: 206 of 206 core cases pass, plus 4 of 9 stretch cases under the plan's rules and 9 of 9 with the stretch rules;
  - on the real GT (counts and ids only), its report is identical to `gt_baseline_v3.json`.

## 7. Notes for the tools

- **Fake pool.** Draw from rows with `fake_pool = 1`, of the original's state and class. Write `display_name` in the original's case, and the row's own `postcode`. There is no ACT regional class; an ACT original is always metro.
- **Gazetteer.** Use every row of both sources, plus `aliases.csv`. Localities with no people, official rows and cross-border postcodes are real addresses too. Match names case-insensitively, folding hyphens, dots, apostrophes and 'Mt'/'Saint' as the reference parser does: `locality` holds ABS case (`Gulf Of Carpentaria`) and, on official rows, upper case.
- **Range gate.** A state and postcode anchor when the postcode lies in the state's ranges or the (state, postcode) pair is in `cross_border.csv`.
- **The original's class.** Take it from the matched row (an official row has its own class) or, for an alias, from the row the alias names.
- **Do not parse `population_2021` as a number on official rows**: it is empty there.
- **Class is GCCSA-based.**
  - Large non-capital cities are `regional`: Newcastle, Wollongong, Gold Coast (Surfers Paradise), Geelong and Townsville.
  - Queanbeyan and Jerrabomberra are `regional`, although they adjoin Canberra.
  - All of the ACT is `metro`, including Tharwa and Uriarra Village.
- **Rural ACT district names** used as address localities appear only if the ABS also has a SAL of that name, because the ACT Remainder areas were removed. For example, Majura and Kowen exist only as "ACT Remainder - ..." areas, and they are not official localities either.
- **Attribution.** Ship `ATTRIBUTION.txt` unchanged with the list and with any file derived from it, for example in the header of the tools' localities resource.

## 8. Edition

- This list is ASGS Edition 3 (2021). The ABS has since released ASGS Edition 4 (July 2026 - June 2031): its main structures came first, and SAL 2026 and POA 2026 followed on 14 Oct 2026.
- S1 already contains `CG_SAL_2021_SAL_2026.csv`. Of the 15,353 SALs:
  - 14,583 keep their name with one target;
  - 656 are renamed with one target, mostly a suffix style change ("(Vic.)" to "(Vic)", "Regional - NSW" to "- NSW"), plus some true renames such as Amosfield to Liston and Environa to Tralee;
  - 59 are split, with two or more shares of at least 0.01;
  - the remaining 55 move only slivers below 0.01.
- **Postal Areas (corrected).** The 2021 set has 2,644 codes (2,641 Postal Areas plus the special codes 9494, 9797 and ZZZZ). The 2026 set has 2,645: it adds 4513, carved from 4510 (ratio 0.0031489, about 0.3% of 4510's 2021 population, which the ABS flags as below minimum output size), and removes none. v1 reported "the same 2,644 codes" because its script compared the 2021 code column with the 2021 name column. `build_v2.log` recomputes both figures.
- A move to Edition 4 would need these suffix rules and a new SAL-POA correspondence, and would add postcode 4513. Population weights would also lag: the 2021 Mesh Block counts came out in June 2022, so 2026 counts are unlikely before 2027.

## 9. Files

- `build_v2.py`: the reproducible build. Sources and their sha256, v1's pinned table hash, the state ranges, the exclusion rules, the review words and decisions, the display fixes, the alias decisions and the attribution text are all constants near its top. It is self-contained: v1's reader and build code are copied into it, so it does not import v1's `build.py`.
- `au_localities.csv`, `aliases.csv`, `cross_border.csv`, `ATTRIBUTION.txt`: the outputs to ship.
- `build_v2.log`: the full check output of the final build.
- `PROVENANCE.md`: this file, written by hand from the build log.
- `scripts/`: one-off inspection scripts (all run with `python -I -B`) and their logs, the independent check `verify_v2.py`, the v1 projection used for v1's check script (`v1proj/`), the case-table and real-GT regression outputs (counts and ids only), the two ABS documentation PDFs extracted from S1 for reading (`pdf_extract/`), and `v1_files_before.sha256`, the hashes of every v1 file taken before v2 was built.
- `../raw/` (the v1 folder): the downloaded sources, unchanged, and `raw/pages/` with copies of the cited web pages. Treat it as untrusted data: read it, never execute it.
