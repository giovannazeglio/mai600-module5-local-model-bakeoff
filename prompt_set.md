# Fixed Prompt Set

All prompts are fictional, non-sensitive, and related to one professional use case: an interior design and commercial furniture sales assistant. The same seven prompts were tested on both models with temperature 0.2, seed 42, and a 350-token output limit.

## P1 — Summarization

**Expected behavior:** Summarize the supplied facts accurately in no more than 80 words.

> Summarize the following fictional client notes in no more than 80 words. Include the project size, requested furniture, priorities, budget, and deadline. Do not add information.
>
> Client: Northstar Creative Studio. Office size: 2,400 square feet. Requested furniture: 18 height-adjustable desks, 18 ergonomic task chairs, two six-person meeting tables, and four mobile storage pedestals. Priorities: acoustic comfort, flexible collaboration, and a clear central circulation path. Furniture budget: $28,000 before tax. Requested installation date: October 16, 2026.

## P2 — Classification

**Expected behavior:** Assign exactly one allowed label to each inquiry.

> Classify each fictional customer inquiry using exactly one of these labels: SPACE_PLANNING, PRODUCT_QUESTION, DELIVERY_ISSUE, PRICE_REQUEST, or WARRANTY_REQUEST. Return only a two-column Markdown table with Inquiry ID and Label.
>
> I1: Can twelve workstations fit while keeping the main aisle open?  
> I2: Does this chair include adjustable lumbar support?  
> I3: Our conference table arrived without the hardware box.  
> I4: Please send pricing for twenty desks and chairs.  
> I5: The height-adjustment mechanism stopped working after six months.

## P3 — Structured output

**Expected behavior:** Return valid JSON with the exact requested keys and no extra commentary.

> Convert the following fictional project information into valid JSON. Use exactly these keys: project_name, room_width_ft, room_length_ft, workstation_count, finish, budget_usd, and priority. Return JSON only, without Markdown.
>
> Project name: Harbor Point Office Refresh; room width: 24 feet; room length: 36 feet; workstations: 12; finish: white laminate with silver frames; budget: $21,500; priority: complete installation before staff return.

## P4 — Domain-specific language

**Expected behavior:** Use relevant interior design and furniture terminology while identifying assumptions.

> You are supporting a fictional commercial office furniture project. Explain how benching workstations, mobile pedestals, acoustic desk screens, task chairs, and a clear circulation path could work together in an open office. Write one professional paragraph of 120 to 160 words. Use the terms correctly, focus on function, and state that final dimensions must be verified in a measured floor plan. Do not claim code compliance.

## P5 — Hallucination-sensitive

**Expected behavior:** Refuse to invent missing specifications and request a reliable source.

> A client asks whether the fictional Aurelia X200 task chair has a 12-year warranty, GREENGUARD Gold certification, and a 300-pound weight capacity. No product sheet, manufacturer name, or verified specifications are available. Answer in 70 to 100 words. Clearly separate what is known from what cannot be verified, do not invent product facts, and recommend the next verification step.

## P6 — Reasoning

**Expected behavior:** Calculate a final total of $23,773.10 and a remaining budget of $1,226.90.

> Calculate the total for this fictional furniture order: 18 desks at $680 each, 18 task chairs at $295 each, and 18 mobile pedestals at $210 each. Sales tax is 7% and applies only to the furniture subtotal. Installation is $950 and is not taxable. The total budget is $25,000. Show the furniture subtotal, tax, installation, final total, and amount remaining or over budget. Round currency to two decimals and briefly explain each step.

## P7 — Tone/persona

**Expected behavior:** Create an empathetic professional reply without unsupported promises.

> Rewrite the following fictional customer complaint as a professional reply from an interior design sales consultant. Use an empathetic but calm tone, keep the response between 100 and 130 words, acknowledge the concern, explain that the delivery record will be reviewed, and provide a clear next step. Do not promise a refund or a delivery date that has not been confirmed.
>
> Complaint: Half of the chairs arrived today, two cartons look damaged, and nobody told us the order would be split. We need an answer now because our staff returns next week.

