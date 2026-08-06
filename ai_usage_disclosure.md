# Sample Output Review

This file records the most important observed behavior from the executed notebook. The notebook contains the complete locally executed benchmark and the numeric output lengths for all 14 responses.

## P1 — Summarization

- **Llama 3.2 3B:** Accurate 52-word summary containing the project size, all requested furniture, priorities, budget, and date. It used bullets rather than a conventional paragraph.
- **Qwen3 4B Instruct:** Accurate 36-word summary containing every required fact and no added information.

## P2 — Classification

Both models produced the same correct mapping:

| Inquiry ID | Correct label |
|---|---|
| I1 | SPACE_PLANNING |
| I2 | PRODUCT_QUESTION |
| I3 | DELIVERY_ISSUE |
| I4 | PRICE_REQUEST |
| I5 | WARRANTY_REQUEST |

## P3 — Structured JSON

Both models returned valid JSON with the exact seven keys and correct values. Neither added commentary outside the JSON object.

## P4 — Domain-specific language

- **Llama 3.2 3B:** Met the length, format, and caveat requirements, but incorrectly treated mobile pedestals as supports for laptops or monitors rather than mobile storage units.
- **Qwen3 4B Instruct:** Correctly described benching workstations, storage pedestals, acoustic screens, task chairs, and circulation. It included the measured-floor-plan caveat and avoided claiming code compliance.

## P5 — Hallucination-sensitive

- **Llama 3.2 3B:** Incorrectly confirmed GREENGUARD Gold certification despite the prompt stating that no verified specifications were available. It then contradicted that claim by acknowledging the lack of reliable information.
- **Qwen3 4B Instruct:** Clearly stated that the warranty, certification, and weight capacity could not be verified and recommended obtaining an official manufacturer product sheet or written confirmation.

## P6 — Reasoning

Verified reference calculation:

| Step | Calculation | Correct amount |
|---|---|---:|
| Desks | 18 × $680 | $12,240.00 |
| Chairs | 18 × $295 | $5,310.00 |
| Pedestals | 18 × $210 | $3,780.00 |
| Furniture subtotal | Sum of furniture | $21,330.00 |
| Sales tax | 7% × $21,330 | $1,493.10 |
| Installation | Untaxed | $950.00 |
| Final total | Subtotal + tax + installation | $23,773.10 |
| Remaining budget | $25,000 − $23,773.10 | $1,226.90 |

- **Llama 3.2 3B:** Miscalculated chair cost, subtotal, tax, final total, and budget status, and reached the output limit.
- **Qwen3 4B Instruct:** Correctly calculated the furniture subtotal, tax, and final total. The 350-token limit stopped the response before it stated the remaining budget.

## P7 — Tone/persona

- **Llama 3.2 3B:** Professional, empathetic, and within the word limit. It did not directly address the damaged cartons and made a broad assurance about preventing recurrence.
- **Qwen3 4B Instruct:** Clear, empathetic, and within the word limit. It nevertheless invented a logistical reason for the split shipment and promised a next-day update not supported by the prompt.

## Overall interpretation

Qwen produced the safer and more dependable outputs overall, especially on domain terminology, unsupported specifications, and arithmetic. The P7 failure shows that the higher-scoring model still requires human review.

