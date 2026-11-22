# Confusion set changelog

## v1 (2026-11-17): native-speaker review

115 pairs reviewed; 102 valid, 13 excluded.

- `beh_joined`: pair ب / ن marked **invalid** (L)
- `beh_joined`: pair ب / ڻ marked **invalid** (L)
- `beh_joined`: pair ب / ي marked **invalid** (L)
- `beh_joined`: pair ت / ن marked **invalid** (L)
- `beh_joined`: pair ت / ڻ marked **invalid** (L)
- `beh_joined`: pair ت / ي marked **invalid** (L)
- `beh_joined`: pair ن / ي marked **invalid** (L)
- `beh_joined`: pair ڻ / ي marked **invalid** (L)
- `jeem_digraph`: pair جه / ڄ marked **invalid** (M)
- `feh`: pair ف / ق marked **invalid** (M)
- `feh`: pair ڦ / ق marked **invalid** (M)
- `yeh_enc`: description of ی changed from "U+06CC farsi yeh (no dots when final)" to "Farsi/Urdu yeh: no dots in final/isolated form; Urdu/Persian code point, rare in Sindhi text (reviewer)"
- `kaf_enc`: pair ڪ / ك marked **invalid** (L): ك this is urdu or arabic and is not used that much in sindhi
- `kaf_enc`: description of ك changed from "U+0643 arabic kaf" to "Arabic kaf; Urdu/Arabic code point, rarely used in Sindhi text (reviewer)"
- `kaf_enc`: pair ک / ك marked **invalid** (L): ك this is urdu or arabic and is not used that much in sindhi

Draft kept as `data/confusion_set_draft1.json`.

### Effect on the fixed synthetic sets (rebuilt with v1; draft-1 sets kept in `data/synth_draft1/`)

Same seeds, same sentences, same positives (val 2,011 / test_iid 2,082 / test_font 2,024); 0 invisible relabels.
Confusion negatives per group, draft-1 -> v1 (val / test_iid / test_font):
beh_joined 127/154/152 -> 78/70/71; feh 35/53/47 -> 22/28/24; noon, jeem, seen rose (they absorb the weight).
Still too rare for per-group numbers: tah 3/4/4, sad 6/4/5, gaf_digraph 6/7/1 -> a per-group balanced set is needed.
