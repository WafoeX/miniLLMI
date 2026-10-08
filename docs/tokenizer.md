# Deterministic byte tokenizer — Stage 15 C3

Tokenizer vocabulary version 1 is deliberately minimal and exactly matches the
frozen tiny decoder vocabulary of 258 IDs:

- IDs `0..255`: literal unsigned byte values;
- `256`: BOS;
- `257`: EOS.

`ByteTokenizer` accepts only a valid decoder configuration with `vocab == 258`
and vocabulary version `1`. There is no unknown-token ID: decoding any other
ID is an error rather than a silent substitution.

## Policy

`encode(bytes)` is deterministic and defaults to `prepend_bos=true`,
`append_eos=false`; callers may explicitly enable EOS. Every input byte maps to
one literal token, including arbitrary UTF-8 or invalid byte sequences.

`decode(tokens)` defaults to skipping BOS/EOS and restores literal bytes exactly.
With `skip_special=false`, encountering either special ID is an error because
special IDs are not bytes. `decode_for_display(tokens)` first performs this
byte-preserving decode, then renders invalid UTF-8 sequences as U+FFFD for
terminal safety. Rendering does not modify token IDs or the byte-oriented
round-trip.

This is a fixed byte tokenizer, not BPE or a language-quality claim. Stage
15-C4 BPE remains `skipped_optional`; the byte path stays the required fallback.
