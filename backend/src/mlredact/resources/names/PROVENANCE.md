# given_gender.csv: provenance

Built 2026-10-31 by `backend/tools/build_given_gender.py` (Python standard library only) from public reference data.
No document content or personal data was used.

Purpose: a lookup list that tells the person resolver whether a given name is usually a man's or a woman's name, so
a fake first name keeps the person's gender when the document gives no title (the list is read before a kinship
word next to the name, which only decides names the list does not gender). It is never a source of fakes: fakes
come only from `resources/surrogates`.

| Item | Value |
|---|---|
| Input | NSW BDM "Popular Baby Names 1952 to 2025", `popular_baby_names_1952_to_2025.csv`, 378,688 bytes, sha256 `1335e7311868520ea172a1de107a1a1d5d655909c77afa531c1d92fbee1e37c2` |
| Input content | the 100 most registered names per year in NSW, for boys and for girls, 1952 to 2025 (14,801 rows) |
| Output | `given_gender.csv`, 11,373 bytes, 731 names + 1 header row, sha256 `4dd9decf842b782268fa17a615f5a2bb620ef0be9add269568b4f949a2dd2836` |
| Licence | CC BY 4.0 (see `ATTRIBUTION.txt`) |

Columns: `name` (lower-case letters only, tool 2's comparison key: `JO-ANNE` is `joanne`), `male` and `female` (the
registrations summed over all years), `gender` (`m` or `f` when at least 90% of the registrations are of that gender,
otherwise `u`: 402 f, 326 m, 3 u).

Limits: only names that reached a yearly top 100 in NSW are listed, so nicknames (Tom, Lizzie), rarer names and most
non-English names are missing, and a name can look single-gender only because its other-gender use never reached a
top 100 (the resolver therefore also treats the names of `surrogates/given_neutral.txt` as giving no gender).

Format: UTF-8 (ASCII) without BOM, LF line endings, sorted by name. Two builds gave identical bytes.

Rebuild: `python -I -B tools/build_given_gender.py <input csv> src/mlredact/resources/names` from `backend/`.
