import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoDir = path.resolve(process.argv[2] ?? ".");

const benchmarkHeaders = [
  "Model", "Prompt ID", "Response Time (s)", "Tokens/sec",
  "Peak GPU Memory (MiB)", "GPU Memory Increase (MiB)",
  "Output Length (tokens)", "Output Length (words)", "Notes",
];

const benchmarkRows = [
  ["llama3.2:3b", "P1", 5.197, 165.05, 3371, 2661, 92, 52, "Accurate summary; used bullets."],
  ["llama3.2:3b", "P2", 4.921, 164.81, 3371, 2661, 51, 36, "All five labels correct."],
  ["llama3.2:3b", "P3", 5.053, 165.00, 3375, 2665, 65, 25, "Valid JSON and exact schema."],
  ["llama3.2:3b", "P4", 5.729, 140.14, 3373, 2663, 160, 137, "Mischaracterized mobile pedestals."],
  ["llama3.2:3b", "P5", 5.348, 154.17, 3373, 2663, 110, 89, "Invented a certification claim."],
  ["llama3.2:3b", "P6", 7.093, 140.78, 3375, 2665, 350, 189, "Multiple arithmetic errors; output truncated."],
  ["llama3.2:3b", "P7", 5.637, 140.06, 3371, 2661, 141, 124, "Professional; did not directly address damaged cartons."],
  ["qwen3:4b-instruct", "P1", 5.171, 127.83, 3967, 3257, 78, 36, "Complete 36-word summary."],
  ["qwen3:4b-instruct", "P2", 5.008, 124.05, 3971, 3261, 57, 32, "All five labels correct."],
  ["qwen3:4b-instruct", "P3", 5.225, 125.26, 3969, 3259, 80, 27, "Valid JSON and exact schema."],
  ["qwen3:4b-instruct", "P4", 5.920, 112.39, 3967, 3257, 150, 122, "Correct terminology and caveat."],
  ["qwen3:4b-instruct", "P5", 5.419, 116.29, 3967, 3257, 100, 74, "Properly refused unsupported claims."],
  ["qwen3:4b-instruct", "P6", 7.773, 109.70, 3971, 3261, 350, 162, "Correct subtotal, tax, and final total; ending truncated."],
  ["qwen3:4b-instruct", "P7", 5.849, 112.85, 3971, 3261, 145, 121, "Added an unsupported reason and update deadline."],
];

const qualityHeaders = [
  "Model", "Prompt ID", "Accuracy", "Helpfulness", "Format",
  "Completeness", "Grounding", "Clarity", "Speed", "Overall Average", "Overall Notes",
];

const qualityRows = [
  ["Llama 3.2 3B", "P1", 5, 5, 4, 5, 5, 5, 5, 4.86, "Accurate and concise; used a bullet list rather than a conventional paragraph."],
  ["Llama 3.2 3B", "P2", 5, 5, 5, 5, 5, 5, 5, 5.00, "All five classifications were correct and the requested table format was followed."],
  ["Llama 3.2 3B", "P3", 5, 5, 5, 5, 5, 5, 5, 5.00, "Valid JSON with the exact requested keys and values."],
  ["Llama 3.2 3B", "P4", 3, 4, 5, 4, 5, 4, 5, 4.29, "Incorrectly described mobile pedestals as equipment supports instead of storage units."],
  ["Llama 3.2 3B", "P5", 1, 2, 4, 3, 1, 2, 5, 2.57, "Unsupportedly confirmed a certification, then contradicted the claim."],
  ["Llama 3.2 3B", "P6", 1, 1, 3, 2, 2, 2, 4, 2.14, "Multiple arithmetic errors and an incorrect over-budget conclusion; output truncated."],
  ["Llama 3.2 3B", "P7", 4, 4, 5, 4, 4, 5, 5, 4.43, "Professional and within the limit, but weak on damaged cartons and made a broad assurance."],
  ["Qwen3 4B Instruct", "P1", 5, 5, 5, 5, 5, 5, 5, 5.00, "Complete and accurate 36-word summary."],
  ["Qwen3 4B Instruct", "P2", 5, 5, 5, 5, 5, 5, 5, 5.00, "All five classifications were correct in the requested two-column table."],
  ["Qwen3 4B Instruct", "P3", 5, 5, 5, 5, 5, 5, 5, 5.00, "Valid JSON with the exact schema and correct values."],
  ["Qwen3 4B Instruct", "P4", 5, 5, 5, 5, 5, 5, 5, 5.00, "Correct terminology, length, measurement caveat, and no code-compliance claim."],
  ["Qwen3 4B Instruct", "P5", 5, 5, 5, 5, 5, 5, 5, 5.00, "Separated known and unknown information and refused to invent specifications."],
  ["Qwen3 4B Instruct", "P6", 5, 4, 4, 3, 5, 5, 4, 4.29, "Correct subtotal, tax, and final total; token limit cut off the remaining-budget line."],
  ["Qwen3 4B Instruct", "P7", 3, 4, 4, 5, 2, 5, 5, 4.00, "Invented a logistical reason and promised an unconfirmed update deadline."],
];

const promptHeaders = ["Prompt ID", "Prompt Type", "Prompt Text", "Expected Behavior"];
const promptRows = [
  ["P1", "Summarization", "Summarize the following fictional client notes in no more than 80 words. Include the project size, requested furniture, priorities, budget, and deadline. Do not add information.\n\nClient: Northstar Creative Studio. Office size: 2,400 square feet. Requested furniture: 18 height-adjustable desks, 18 ergonomic task chairs, two six-person meeting tables, and four mobile storage pedestals. Priorities: acoustic comfort, flexible collaboration, and a clear central circulation path. Furniture budget: $28,000 before tax. Requested installation date: October 16, 2026.", "Summarize the supplied facts accurately in no more than 80 words."],
  ["P2", "Classification", "Classify each fictional customer inquiry using exactly one of these labels: SPACE_PLANNING, PRODUCT_QUESTION, DELIVERY_ISSUE, PRICE_REQUEST, or WARRANTY_REQUEST. Return only a two-column Markdown table with Inquiry ID and Label.\n\nI1: Can twelve workstations fit while keeping the main aisle open?\nI2: Does this chair include adjustable lumbar support?\nI3: Our conference table arrived without the hardware box.\nI4: Please send pricing for twenty desks and chairs.\nI5: The height-adjustment mechanism stopped working after six months.", "Assign exactly one allowed label to each inquiry."],
  ["P3", "Structured output", "Convert the following fictional project information into valid JSON. Use exactly these keys: project_name, room_width_ft, room_length_ft, workstation_count, finish, budget_usd, and priority. Return JSON only, without Markdown.\n\nProject name: Harbor Point Office Refresh; room width: 24 feet; room length: 36 feet; workstations: 12; finish: white laminate with silver frames; budget: $21,500; priority: complete installation before staff return.", "Return valid JSON with the exact requested keys and no extra commentary."],
  ["P4", "Domain-specific", "You are supporting a fictional commercial office furniture project. Explain how benching workstations, mobile pedestals, acoustic desk screens, task chairs, and a clear circulation path could work together in an open office. Write one professional paragraph of 120 to 160 words. Use the terms correctly, focus on function, and state that final dimensions must be verified in a measured floor plan. Do not claim code compliance.", "Use relevant interior design and furniture terminology while identifying assumptions."],
  ["P5", "Hallucination-sensitive", "A client asks whether the fictional Aurelia X200 task chair has a 12-year warranty, GREENGUARD Gold certification, and a 300-pound weight capacity. No product sheet, manufacturer name, or verified specifications are available. Answer in 70 to 100 words. Clearly separate what is known from what cannot be verified, do not invent product facts, and recommend the next verification step.", "Refuse to invent missing specifications and request a reliable source."],
  ["P6", "Reasoning", "Calculate the total for this fictional furniture order: 18 desks at $680 each, 18 task chairs at $295 each, and 18 mobile pedestals at $210 each. Sales tax is 7% and applies only to the furniture subtotal. Installation is $950 and is not taxable. The total budget is $25,000. Show the furniture subtotal, tax, installation, final total, and amount remaining or over budget. Round currency to two decimals and briefly explain each step.", "Calculate a final total of $23,773.10 and remaining budget of $1,226.90."],
  ["P7", "Tone/persona", "Rewrite the following fictional customer complaint as a professional reply from an interior design sales consultant. Use an empathetic but calm tone, keep the response between 100 and 130 words, acknowledge the concern, explain that the delivery record will be reviewed, and provide a clear next step. Do not promise a refund or a delivery date that has not been confirmed.\n\nComplaint: Half of the chairs arrived today, two cartons look damaged, and nobody told us the order would be split. We need an answer now because our staff returns next week.", "Create an empathetic and professional reply without unsupported promises."],
];

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function csvText(headers, rows) {
  return [headers, ...rows].map(row => row.map(csvEscape).join(",")).join("\r\n") + "\r\n";
}

function applyTableStyle(sheet, rangeAddress, numericRanges = []) {
  sheet.showGridLines = false;
  const used = sheet.getRange(rangeAddress);
  used.format = {
    font: { name: "Aptos", size: 10, color: "#1F2937" },
    verticalAlignment: "top",
  };
  const header = used.getRow(0);
  header.format = {
    fill: "#1F4E78",
    font: { name: "Aptos Display", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "medium", color: "#17365D" },
  };
  used.format.borders = {
    insideHorizontal: { style: "thin", color: "#D9E2F3" },
    bottom: { style: "thin", color: "#A6A6A6" },
  };
  sheet.freezePanes.freezeRows(1);
  for (const item of numericRanges) {
    sheet.getRange(item.range).format.numberFormat = item.format;
    sheet.getRange(item.range).format.horizontalAlignment = "right";
  }
}

await fs.mkdir(path.join(repoDir, "data"), { recursive: true });
await fs.mkdir(path.join(repoDir, "results"), { recursive: true });

await fs.writeFile(path.join(repoDir, "benchmark_results.csv"), csvText(benchmarkHeaders, benchmarkRows), "utf8");
await fs.writeFile(path.join(repoDir, "quality_scores.csv"), csvText(qualityHeaders, qualityRows), "utf8");
await fs.writeFile(path.join(repoDir, "data", "prompt_inputs.csv"), csvText(promptHeaders, promptRows), "utf8");

const rawHeaders = ["Model", "Prompt ID", "Prompt Type", ...benchmarkHeaders.slice(2)];
const typeMap = Object.fromEntries(promptRows.map(row => [row[0], row[1]]));
const rawRows = benchmarkRows.map(row => [row[0], row[1], typeMap[row[1]], ...row.slice(2)]);
await fs.writeFile(path.join(repoDir, "results", "raw_benchmark_log.csv"), csvText(rawHeaders, rawRows), "utf8");

const workbook = Workbook.create();

const summary = workbook.worksheets.add("Summary");
summary.getRange("A1:E4").values = [
  ["Model", "Avg Response Time (s)", "Avg Tokens/sec", "Max Peak GPU (MiB)", "Overall Quality"],
  ["Llama 3.2 3B", 5.57, 152.86, 3375, 4.04],
  ["Qwen3 4B Instruct", 5.77, 118.34, 3971, 4.76],
  ["Difference (Qwen - Llama)", 0.20, -34.52, 596, 0.72],
];
applyTableStyle(summary, "A1:E4", [
  { range: "B2:B4", format: "0.00" },
  { range: "C2:C4", format: "0.00" },
  { range: "D2:D4", format: "#,##0" },
  { range: "E2:E4", format: "0.00" },
]);
summary.getRange("A1:E4").format.autofitColumns();
summary.getRange("A1:E4").format.autofitRows();
summary.getRange("A1:A4").format.columnWidth = 28;

const benchmark = workbook.worksheets.add("Benchmark Results");
benchmark.getRangeByIndexes(0, 0, benchmarkRows.length + 1, benchmarkHeaders.length).values = [benchmarkHeaders, ...benchmarkRows];
applyTableStyle(benchmark, `A1:I${benchmarkRows.length + 1}`, [
  { range: `C2:D${benchmarkRows.length + 1}`, format: "0.00" },
  { range: `E2:H${benchmarkRows.length + 1}`, format: "#,##0" },
]);
benchmark.getRange(`A1:I${benchmarkRows.length + 1}`).format.autofitColumns();
benchmark.getRange(`I2:I${benchmarkRows.length + 1}`).format.wrapText = true;
benchmark.getRange("A1:A15").format.columnWidth = 22;
benchmark.getRange("I1:I15").format.columnWidth = 52;

const quality = workbook.worksheets.add("Quality Scores");
quality.getRangeByIndexes(0, 0, qualityRows.length + 1, qualityHeaders.length).values = [qualityHeaders, ...qualityRows];
applyTableStyle(quality, `A1:K${qualityRows.length + 1}`, [
  { range: `C2:I${qualityRows.length + 1}`, format: "0" },
  { range: `J2:J${qualityRows.length + 1}`, format: "0.00" },
]);
quality.getRange(`A1:K${qualityRows.length + 1}`).format.autofitColumns();
quality.getRange(`K2:K${qualityRows.length + 1}`).format.wrapText = true;
quality.getRange("A1:A15").format.columnWidth = 24;
quality.getRange("K1:K15").format.columnWidth = 62;
quality.getRange(`J2:J${qualityRows.length + 1}`).conditionalFormats.add("colorScale", {
  colors: ["#F8696B", "#FFEB84", "#63BE7B"],
  thresholds: ["min", "50%", "max"],
});

const prompts = workbook.worksheets.add("Prompt Inputs");
prompts.getRangeByIndexes(0, 0, promptRows.length + 1, promptHeaders.length).values = [promptHeaders, ...promptRows];
applyTableStyle(prompts, `A1:D${promptRows.length + 1}`);
prompts.getRange(`A1:D${promptRows.length + 1}`).format.autofitColumns();
prompts.getRange(`C2:D${promptRows.length + 1}`).format.wrapText = true;
prompts.getRange("A1:A8").format.columnWidth = 12;
prompts.getRange("B1:B8").format.columnWidth = 24;
prompts.getRange("C1:C8").format.columnWidth = 90;
prompts.getRange("D1:D8").format.columnWidth = 52;
prompts.getRange("2:8").format.rowHeight = 72;

const inspection = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:E4",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 8,
  maxChars: 4000,
});
console.log(inspection.ndjson ?? inspection);

const preview = await workbook.render({ sheetName: "Summary", autoCrop: "all", scale: 1.5, format: "png" });
await fs.writeFile(path.join(repoDir, "results", "benchmark_tables_preview.png"), new Uint8Array(await preview.arrayBuffer()));

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(repoDir, "results", "benchmark_tables.xlsx"));

