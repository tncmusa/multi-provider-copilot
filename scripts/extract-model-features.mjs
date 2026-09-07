#!/usr/bin/env node

/**
 * Extract model features from the models.dev catalog.
 *
 * Fetches the official models.dev catalog (catalog.json, with a mirror
 * fallback), looks up each requested model in the `opencode-go` provider
 * section plus the global catalog, and prints a feature report.
 *
 * The report contains the raw catalog entries (provider-specific + global)
 * and the resolved features the extension derives from them via the same
 * inference rules as src/modelsDev.ts / src/catalogModels.ts:
 * thinking mode, reasoning efforts, default effort (incl. overrides from
 * src/modelOverrides.ts), vision, API mode, context/output limits, cost, etc.
 *
 * Usage:
 *   node scripts/extract-model-features.mjs [modelId...]
 *   node scripts/extract-model-features.mjs glm-5.1 glm-5.2
 *   node scripts/extract-model-features.mjs --json glm-5.1 glm-5.2
 *
 * Default model list: glm-5.1 glm-5.2 (OpenCode Go GLM 5.1 / 5.2).
 */

const CATALOG_URL = "https://models.dev/catalog.json";
const MIRROR_URL = "https://modelsdev-mirror.onesoft.top/catalog.json";
const PROVIDER_ID = "opencode-go";
const FETCH_TIMEOUT_MS = 30000;

const DEFAULT_MODEL_IDS = ["glm-5.1", "glm-5.2"];

/** Conservative defaults, matching src/catalogModels.ts. */
const DEFAULT_CONTEXT_LENGTH = 128000;
const DEFAULT_MAX_TOKENS = 4096;

/**
 * Per-model overrides mirroring src/modelOverrides.ts.
 * Keep in sync when the override table changes.
 */
const MODEL_OVERRIDES = {
    // GLM-5.2 defaults to "high" instead of the catalog's highest effort "max".
    "glm-5.2": { defaultReasoningEffort: "high" },
};

// ── Fetching ──

async function fetchJson(url, timeoutMs) {
    const response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
    if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
    }
    return await response.json();
}

/**
 * Load the catalog with the same two-tier fallback chain the extension uses
 * (official → mirror). Throws if both fail.
 */
async function fetchCatalog() {
    const sources = [
        { url: CATALOG_URL, label: "official" },
        { url: MIRROR_URL, label: "mirror" },
    ];
    for (const { url, label } of sources) {
        try {
            const data = await fetchJson(url, FETCH_TIMEOUT_MS);
            return { data, source: label };
        } catch (err) {
            console.error(`[warn] ${label} catalog fetch failed: ${err.message}`);
        }
    }
    throw new Error("All catalog sources failed (official + mirror)");
}

// ── Catalog lookups ──

/**
 * Find the global catalog entry for a model ID.
 * Direct full-ID match first (e.g. "zhipuai/glm-5.1"), then suffix match
 * on the short ID (e.g. "glm-5.1").
 */
function findGlobalEntry(models, modelId) {
    if (models[modelId]) return models[modelId];
    for (const [fullId, entry] of Object.entries(models)) {
        if (fullId.endsWith(`/${modelId}`)) return entry;
    }
    return undefined;
}

// ── Inference rules (mirror src/modelsDev.ts) ──

/**
 * Thinking mode:
 * - reasoning missing/false          → "always" (no thinking at all)
 * - reasoning_options empty/missing  → "always" (thinking always on)
 * - reasoning_options present        → "switchable"
 */
function inferThinkingMode(entry) {
    if (!entry?.reasoning) return "always";
    const opts = entry.reasoning_options;
    if (!opts || opts.length === 0) return "always";
    return "switchable";
}

/** Explicit effort values from a `{"type":"effort","values":[...]}` option. */
function inferReasoningEfforts(entry) {
    const opts = entry?.reasoning_options;
    if (!opts) return undefined;
    for (const opt of opts) {
        if (opt.type === "effort" && Array.isArray(opt.values) && opt.values.length > 0) {
            return opt.values;
        }
    }
    return undefined;
}

/** Default effort: the highest listed value, or "enabled" for simple on/off. */
function inferDefaultReasoningEffort(entry) {
    const efforts = inferReasoningEfforts(entry);
    if (efforts && efforts.length > 0) return efforts[efforts.length - 1];
    return "enabled";
}

/** Vision: `attachment: true` or image/video in `modalities.input`. */
function inferVision(entry) {
    if (entry?.attachment === true) return true;
    const input = entry?.modalities?.input;
    return Array.isArray(input) && (input.includes("image") || input.includes("video"));
}

/** Thinking budget from a `{"type":"budget_tokens", min?, max?}` option. */
function inferThinkingBudget(entry) {
    const opts = entry?.reasoning_options;
    if (!opts) return undefined;
    for (const opt of opts) {
        if (opt.type === "budget_tokens") {
            const result = {};
            if (typeof opt.min === "number") result.min = opt.min;
            if (typeof opt.max === "number") result.max = opt.max;
            return result;
        }
    }
    return undefined;
}

/**
 * API mode heuristics (mirror deduceApiModeFromFamily):
 * anthropic npm hint, Claude/Anthropic family, Qwen 3.6/3.7, Gemma.
 */
function deduceApiMode(modelId, entry) {
    if (entry?.provider?.npm?.includes("anthropic")) return "anthropic";
    const family = (entry?.family ?? "").toLowerCase();
    if (family.includes("claude") || family.includes("anthropic")) return "anthropic";
    if (family.includes("qwen")) {
        return /qwen[\s-]*3\.[67]/i.test(modelId) ? "anthropic" : "openai";
    }
    if (family.includes("gemma")) return "anthropic";
    return "openai";
}

// ── Feature resolution ──

/**
 * Resolve the full feature set for one model.
 * Merge chain (mirror resolveModelMeta): provider entry → global entry →
 * conservative defaults, then apply overrides.
 */
function resolveFeatures(modelId, providerEntry, globalEntry, provider) {
    const entry = providerEntry ?? globalEntry;
    const override = MODEL_OVERRIDES[modelId] ?? {};

    const rawEfforts = inferReasoningEfforts(entry);
    // "none"/"disabled" effort values map to the "disabled" picker option
    const supportedReasoningEfforts = (rawEfforts ?? []).filter(
        (e) => e !== "none" && e !== "disabled"
    );

    return {
        modelId,
        displayName: entry?.name ?? modelId,
        description: entry?.description,
        family: entry?.family,
        provider: PROVIDER_ID,
        providerName: provider?.name,
        providerApiBaseUrl: provider?.api ? provider.api.replace(/\/+$/, "") + "/" : undefined,
        apiMode: deduceApiMode(modelId, entry),
        contextLength: entry?.limit?.context ?? DEFAULT_CONTEXT_LENGTH,
        maxOutputTokens: entry?.limit?.output ?? DEFAULT_MAX_TOKENS,
        vision: inferVision(entry),
        thinkingMode: inferThinkingMode(entry),
        supportedReasoningEfforts,
        defaultReasoningEffort:
            override.defaultReasoningEffort ?? inferDefaultReasoningEffort(entry),
        thinkingBudget: inferThinkingBudget(entry),
        supportsTemperature: entry?.temperature ?? true,
        toolCalling: entry?.tool_call ?? true,
        structuredOutput: entry?.structured_output ?? false,
        interleaved: entry?.interleaved ?? null,
        knowledge: entry?.knowledge,
        releaseDate: entry?.release_date,
        lastUpdated: entry?.last_updated,
        openWeights: entry?.open_weights ?? false,
        status: entry?.status,
        modalities: entry?.modalities ?? null,
        cost: entry?.cost ?? { input: 0, output: 0, cache_read: 0 },
        weights: entry?.weights ?? [],
        benchmarks: entry?.benchmarks ?? [],
    };
}

// ── Reporting ──

function formatNumber(n) {
    return Number(n).toLocaleString("en-US");
}

function formatCost(cost) {
    const parts = [];
    if (typeof cost.input === "number") parts.push(`$${cost.input.toFixed(2)}/M input`);
    if (typeof cost.output === "number") parts.push(`$${cost.output.toFixed(2)}/M output`);
    if (typeof cost.cache_read === "number") parts.push(`$${cost.cache_read.toFixed(2)}/M cache read`);
    return parts.join(", ") || "n/a";
}

function printReport(features) {
    const line = "=".repeat(60);
    console.log(`\n${line}`);
    console.log(`Model: ${features.modelId} (${features.providerName ?? features.provider})`);
    console.log(line);

    const rows = [
        ["Display name", features.displayName],
        ["Description", features.description],
        ["Family", features.family ?? "n/a"],
        ["Provider", `${features.provider} (${features.providerName ?? "?"})`],
        ["API base URL", features.providerApiBaseUrl ?? "n/a"],
        ["API mode", features.apiMode],
        ["Context length", `${formatNumber(features.contextLength)} tokens`],
        ["Max output tokens", `${formatNumber(features.maxOutputTokens)} tokens`],
        ["Vision", features.vision ? "yes" : "no"],
        ["Thinking mode", features.thinkingMode],
        [
            "Reasoning efforts",
            features.supportedReasoningEfforts.length > 0
                ? features.supportedReasoningEfforts.join(", ")
                : "(none — simple on/off)",
        ],
        ["Default effort", features.defaultReasoningEffort],
        [
            "Thinking budget",
            features.thinkingBudget
                ? Object.entries(features.thinkingBudget)
                      .map(([k, v]) => `${k}=${formatNumber(v)}`)
                      .join(", ")
                : "n/a",
        ],
        ["Temperature", features.supportsTemperature ? "supported" : "not supported"],
        ["Tool calling", features.toolCalling ? "supported" : "not supported"],
        ["Structured output", features.structuredOutput ? "supported" : "not supported"],
        ["Interleaved field", features.interleaved?.field ?? "n/a"],
        ["Knowledge cutoff", features.knowledge ?? "n/a"],
        ["Release date", features.releaseDate ?? "n/a"],
        ["Last updated", features.lastUpdated ?? "n/a"],
        ["Open weights", features.openWeights ? "yes" : "no"],
        ["Status", features.status ?? "n/a"],
        [
            "Modalities",
            features.modalities
                ? `input=[${features.modalities.input?.join(", ") ?? "?"}] output=[${features.modalities.output?.join(", ") ?? "?"}]`
                : "n/a",
        ],
        ["Cost", formatCost(features.cost)],
    ];

    const width = Math.max(...rows.map(([k]) => k.length)) + 2;
    for (const [key, value] of rows) {
        console.log(`  ${key.padEnd(width)}: ${value}`);
    }

    if (features.benchmarks.length > 0) {
        console.log(`\n  Benchmarks:`);
        for (const b of features.benchmarks) {
            const meta = [b.harness, b.version, b.dataset, b.date].filter(Boolean).join(", ");
            console.log(`    - ${b.name}: ${b.score} (${b.metric})${meta ? ` [${meta}]` : ""}`);
        }
    }
}

// ── Main ──

async function main() {
    const args = process.argv.slice(2);
    const jsonMode = args.includes("--json");
    const modelIds = args.filter((a) => !a.startsWith("--"));
    const targets = modelIds.length > 0 ? modelIds : DEFAULT_MODEL_IDS;

    const { data, source } = await fetchCatalog();
    console.error(`[info] catalog loaded from: ${source}`);

    const provider = data.providers?.[PROVIDER_ID];
    if (!provider) {
        console.error(
            `[error] provider "${PROVIDER_ID}" not found in catalog. ` +
                `Available: ${Object.keys(data.providers ?? {}).join(", ")}`
        );
        process.exit(1);
    }

    const report = targets.map((modelId) => {
        const providerEntry = provider.models?.[modelId];
        const globalEntry = findGlobalEntry(data.models ?? {}, modelId);
        return {
            features: resolveFeatures(modelId, providerEntry, globalEntry, provider),
            rawProviderEntry: providerEntry ?? null,
            rawGlobalEntry: globalEntry ?? null,
        };
    });

    if (jsonMode) {
        console.log(JSON.stringify(report, null, 2));
        return;
    }

    for (const { features, rawProviderEntry, rawGlobalEntry } of report) {
        printReport(features);

        console.log(`\n  Raw catalog entry (provider "${PROVIDER_ID}"):`);
        if (rawProviderEntry) {
            console.log(JSON.stringify(rawProviderEntry, null, 2));
        } else {
            console.log("    (no provider-specific entry found)");
        }

        console.log(`\n  Raw catalog entry (global):`);
        if (rawGlobalEntry) {
            console.log(JSON.stringify(rawGlobalEntry, null, 2));
        } else {
            console.log("    (no global entry found)");
        }
    }
    console.log();
}

main().catch((err) => {
    console.error(`[error] ${err.message}`);
    process.exit(1);
});
