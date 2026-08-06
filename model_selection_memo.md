# Model Selection Memo

**To:** Interior Design and Commercial Furniture Project Team  
**From:** Giovanna Zeglio  
**Subject:** Recommendation for a Local Open-Weight Language Model  
**Date:** August 5, 2026

## Decision

Select **Qwen3 4B Instruct** as the local model for routine, low-risk interior design and furniture-sales assistance on the tested Windows laptop. Keep human review mandatory, and continue to use verified product documents and professional judgment for specifications, prices, warranties, certifications, delivery commitments, and client-facing decisions.

## Evidence

The experiment tested Llama 3.2 3B and Qwen3 4B Instruct through Ollama using the same seven fictional prompts and generation settings. The prompts covered summarization, classification, JSON formatting, domain terminology, unsupported-claim resistance, arithmetic reasoning, and professional tone.

| Decision factor | Llama 3.2 3B | Qwen3 4B Instruct | Better result |
|---|---:|---:|---|
| Average response time | 5.57 s | 5.77 s | Llama |
| Average generation speed | 152.86 tokens/s | 118.34 tokens/s | Llama |
| Maximum peak GPU memory | 3,375 MiB | 3,971 MiB | Llama |
| Downloaded model size | 2.0 GB | 2.5 GB | Llama |
| Overall quality score | 4.04 / 5 | **4.76 / 5** | Qwen |
| Hallucination-sensitive task | Invented a certification | Refused unsupported claims | Qwen |
| Furniture calculation | Multiple errors | Correct subtotal, tax, and final total | Qwen |

The average response-time gap was only 0.20 seconds. Although Llama generated about 29% more tokens per second and used 596 MiB less peak GPU memory, both models remained comfortably within the available 16,303 MiB of VRAM. On this hardware, Qwen’s additional memory use was practical and the user-facing delay was small.

## Accepted trade-off

The recommendation accepts lower generation speed, a 0.5 GB larger download, and about 0.6 GB more observed peak GPU memory in exchange for better accuracy, grounding, and instruction following. This is appropriate because incorrect product claims or pricing calculations are more damaging to the selected use case than a small delay.

The test did not isolate quantization effects. Qwen used a Q4_K_M package; the exact quantization level of the default Ollama Llama package was not separately verified. Model family, parameter count, training, and quantization may all have contributed to the observed differences.

## Risks and controls

Qwen also made a meaningful error: in the customer-service prompt, it invented a reason for the split shipment and promised an update deadline that the prompt did not establish. A local model should therefore be used for drafts and organization, not autonomous final decisions.

Recommended controls:

- require human review before any client-facing use;
- check product claims against a current manufacturer document;
- verify all prices and calculations independently;
- remove unsupported promises about refunds, schedules, or delivery status;
- retain the original prompt and output for important decisions;
- escalate complex, high-risk, or ambiguous work to a qualified person.

## Local model versus hosted API

Local deployment is practical for routine summaries, classification, structured notes, and first drafts. It offers privacy, offline operation, and no per-request API charge. A hosted API may remain the better option when the task requires stronger reasoning, broader knowledge, enterprise support, higher reliability, or scaling across many users. If a hosted API is used, the organization must also consider data-sharing rules, internet dependence, variable usage cost, and vendor availability.

## Conclusion

Qwen3 4B Instruct is the best tested fit for this laptop and use case. It is responsive enough for interactive work and earned a 0.72-point quality advantage over Llama. Llama remains a reasonable alternative for simple, low-risk work when speed and memory efficiency are the highest priorities. Neither model should be treated as an authoritative source or used without review.

