# SAFETY

This project fine-tunes **bodhan-ai/indic-speak** (Indic-Speak) with LoRA. It adds no new
voices, no new style vocabulary, and no new tokens; it uses only the closed 44-voice library.

## Attribution (required)

> **Built with Indic-Speak from Bodhan AI / AI4Bharat.**

This string must appear in any product, demo, or research output built on the model or these
adapters. Citation is in the project README.

## License

- **Indic-Speak / Bodhan AI Open Model License** governs the base model, its SNAC codec, and the
  fine-tuned Vocos decoder (`reference/Bodhan_AI_Open_Model_License.md`, `indic-open-license.md`).
  Our LoRA adapters are derivative of the base model and inherit its license terms; distribute
  them under the same license with the attribution above.
- **Datasets** (docs/DATA.md §1): SPRINGLab IndicTTS (Marathi/Hindi/English) and ai4bharat/Rasa.
  Each source's license must be confirmed against its dataset card before redistributing any
  derived data or audio (Decision 10). The pipeline's license gate hard-rejects any clip whose
  license is unknown. We redistribute **no source audio** — only code, configs, adapters, and the
  eval manifests (which reference held-out clips by hash/locator, not their audio).
- The Marathi ASR used for eval (`bodhan-ai/indic-transcribe-core`) is under the same Bodhan Open
  Model License and is gated.

## Speaker-use policy (our own)

- **Closed library only.** Only the 44 named library voices are used; the data pipeline hard-rejects
  any clip whose speaker is not in the roster (`src/indic_speak_ft/voices.py`).
- **No voice cloning, no new voices, no impersonation.** The base model does not support cloning and
  we add none. These voices are synthetic studio personas, not real individuals, and must not be
  presented as a specific real person.
- **No misleading or deceptive content.** Do not use synthesized speech to impersonate a real
  person, fabricate statements attributed to someone, or produce content designed to deceive
  (fake news, scam audio, fraudulent authorization).

## Prohibited uses (inherited + project)

- Impersonation of real individuals or organizations; fabricated records or authorizations.
- Disinformation, harassment, or content that deceives about its synthetic origin where disclosure
  is expected.
- Any use prohibited by the Indic / Bodhan AI Open Model License.
- Circumventing the closed-voice constraint (adding voices, cloning, or editing speaker embeddings).

## Scope limits (honesty — Decision 9)

This is a **Marathi quality-lift** fine-tune (SPRINGLab general Marathi as the signal), not a
domain-specialist STEM model — STEM appears mainly in the *eval* target. Voice drift toward the
SPRINGLab timbre is a known, accepted trade (Decision 7/D15), monitored on both target voices. We
do not claim more than the data supports.
