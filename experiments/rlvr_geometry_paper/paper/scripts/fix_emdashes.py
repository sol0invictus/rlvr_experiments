#!/usr/bin/env python3
"""Replace em-dashes (---) in main.tex with grammatically appropriate punctuation.
En-dashes (--) used for numeric ranges are left untouched."""
import os
import re

path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.tex")
text = open(path).read()

# (old, new) contextual replacements applied BEFORE the generic comma pass.
specials = [
    # --- periods (independent clauses) ---
    (r"at 4B --- training", r"at 4B. Training"),
    (r"between model sizes ---", r"between model sizes."),
    (r"the optimal LR sits between", r"The optimal LR sits between"),
    (r"the rest --- its top singular", r"the rest. Its top singular"),
    (r"the sign pattern is robust ---", r"the sign pattern is robust."),
    (r"pairing this with the Phase-2 regime rule yields",
     r"Pairing this with the Phase-2 regime rule yields"),
    (r"each merge --- no cheap probe substitutes.", r"each merge. No cheap probe substitutes."),
    (r"Approximate GPU-hours --- Phase 1: 100", r"Approximate GPU-hours. Phase 1: 100"),
    # --- colons (term: definition / explanation) ---
    (r"(grounding)} --- parse", r"(grounding)}: parse"),
    (r"(local reasoning)} --- a single", r"(local reasoning)}: a single"),
    (r"(global planning)} --- the multi", r"(global planning)}: the multi"),
    (r"compression --- GRPO shortens", r"compression: GRPO shortens"),
    (r"short ones} --- multi-step", r"short ones}: multi-step"),
    (r"exactly this --- of all", r"exactly this: of all"),
    (r"MLP sink --- the attention", r"MLP sink: the attention"),
    (r"Two things --- the \emph{direction}", r"Two things: the \emph{direction}"),
    (r"We regard this as a contribution", r"We regard this as a contribution:"),
    (r"--- the merging literature has implicitly assumed",
     r"the merging literature has implicitly assumed"),
    (r"merge-damage predictor --- two predictors", r"merge-damage predictor: two predictors"),
    (r"\texttt{kk} column --- GRPO on anything", r"\texttt{kk} column: GRPO on anything"),
    (r"(Fig.~\ref{fig:p3b}c) ---", r"(Fig.~\ref{fig:p3b}c):"),
    (r"to mid-late at 8B ---", r"to mid-late at 8B,"),
    # --- parentheses (paired appositives, often with internal commas) ---
    (r"base) --- GRPO", r"base; GRPO"),
    (r"destroys the model ---", r"destroys the model),"),
    (r"(iv) \emph{RFT-Base} ---", r"(iv) \emph{RFT-Base} ("),
    (r"\textbf{(iv) RFT-Base} ---", r"\textbf{(iv) RFT-Base} ("),
    (r"then GRPO --- is best", r"then GRPO) is best"),
    (r"it edits --- which depths", r"it edits (which depths"),
    ("in which direction\n--- is task-specific", "in which direction) is task-specific"),
    (r"a task introduces --- its \emph{novel subspace},",
     r"a task introduces (its \emph{novel subspace},"),
    (r"lying outside $\Wsft$'s --- and measure", r"lying outside $\Wsft$'s) and measure"),
    (r"back onto the base --- \emph{task arithmetic} ---", r"back onto the base (\emph{task arithmetic})"),
    (r"\emph{Feature projectors} --- ", r"\emph{Feature projectors} ("),
    (r"\texttt{attn.q\_proj}, \texttt{attn.k\_proj} --- choose",
     r"\texttt{attn.q\_proj}, \texttt{attn.k\_proj}) choose"),
    (r"\emph{Sinks} --- ", r"\emph{Sinks} ("),
    ("\\texttt{attn.o\\_proj}\n--- aggregate features",
     "\\texttt{attn.o\\_proj}) aggregate features"),
    # --- reword to avoid splice ---
    (r"insensitive to task identity --- only the direction and depth vary.",
     r"insensitive to task identity, with only the direction and depth varying."),
    # --- title ---
    (r"\section{Phase 4: Task-Vector Merging --- A Characterization}",
     r"\section{Phase 4: Characterizing Task-Vector Merging}"),
    # --- cosmetic: broken hyphenated word ---
    ("favorable-small-\nsample", "favorable-small-sample"),
]

missing = []
for old, new in specials:
    if old in text:
        text = text.replace(old, new)
    else:
        missing.append(old)

# table "not measured" cells: "& ---" -> "& n/a" (only occurs in the predictor table)
text = text.replace("& ---", "& n/a")

# neutralize the long hyphen rule in the header comment (4+ hyphens) so the
# generic pass below does not touch it
text = re.sub(r"-{4,}", "====", text)

# generic pass: every remaining em-dash becomes a comma (simple asides)
text = re.sub(r"\s*---\s*", ", ", text)

remaining = text.count("---")
open(path, "w").write(text)

print("specials applied:", len(specials) - len(missing), "/", len(specials))
if missing:
    print("MISSING (not found, fell through to comma pass):")
    for m in missing:
        print("   ", repr(m))
print("remaining '---' in file:", remaining)
