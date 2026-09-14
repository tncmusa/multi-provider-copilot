/**
 * `x-opencode-session` header support for the OpenCode Go API.
 *
 * OpenCode Go requires a stable per-conversation ID on every inference request
 * (used server-side for routing and prompt-cache optimization; requests without
 * it error since 2026-09-05). The header is sent only for `opencode-go`
 * provider models — never for OpenCode Zen (`-free`), Cline Pass, Ollama
 * Cloud, or NanoGPT endpoints.
 */

import { createHash, randomUUID } from "crypto";
import * as vscode from "vscode";

/**
 * Format a 32-char hex digest as a canonical UUID (8-4-4-4-12).
 */
function formatSessionUuid(hex: string): string {
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20, 32)}`;
}

/**
 * Derive a stable per-conversation session ID for the `x-opencode-session` header.
 *
 * VS Code does not expose a conversation identifier to language model providers,
 * so the ID is derived deterministically from the target model ID plus the
 * conversation's first user message text: chat clients re-send the same history
 * on every turn of a conversation, so the derived ID stays stable across turns
 * while differing between conversations.
 *
 * @param modelId The model ID the request targets (keeps sessions distinct per model).
 * @param messages The request messages from VS Code.
 * @returns A UUID-formatted session ID, or a random UUID when the conversation has no user text anchor (e.g. image-only requests).
 */
export function deriveOpencodeSessionId(
    modelId: string,
    messages: readonly vscode.LanguageModelChatRequestMessage[]
): string {
    for (const message of messages) {
        if (message.role !== vscode.LanguageModelChatMessageRole.User) {
            continue;
        }
        // Collect text parts only — binary data parts (images) are skipped so the
        // hash stays cheap and the ID does not depend on image bytes.
        const anchorText = message.content
            .map((part) => {
                if (typeof part === "string") return part;
                if (part instanceof vscode.LanguageModelTextPart) return part.value;
                return "";
            })
            .join("");
        if (!anchorText.trim()) {
            continue;
        }
        const hash = createHash("sha256");
        hash.update(modelId);
        hash.update(anchorText);
        return formatSessionUuid(hash.digest("hex").slice(0, 32));
    }
    return randomUUID();
}

/**
 * Text-only variant of `deriveOpencodeSessionId` for single-shot plain-text
 * request paths (e.g. Git commit message generation) that do not carry VS Code
 * chat message objects.
 *
 * @param modelId The model ID the request targets.
 * @param text The request prompt text used as the session anchor.
 * @returns A UUID-formatted session ID, or a random UUID when the text is empty.
 */
export function deriveOpencodeSessionIdFromText(modelId: string, text: string): string {
    if (!text.trim()) {
        return randomUUID();
    }
    const hash = createHash("sha256");
    hash.update(modelId);
    hash.update(text);
    return formatSessionUuid(hash.digest("hex").slice(0, 32));
}
