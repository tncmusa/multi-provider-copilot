import * as vscode from "vscode";
import { convertToolsToOpenAI, createDataUrl, isImageMimeType, isResourceLinkMimeType, isToolResultPart, mapRole, parseResourceLinkData, replaceDataUriImages, resolveResourceLinkToImage, storeDataUriImages, collectToolResultText, tryParseJSONObject } from "../utils";
import { CommonApi } from "../commonApi";
import type { OpenCodeGoModelItem } from "../types";
import type { ResponsesInputContent, ResponsesInputItem, ResponsesRequestBody, ResponsesResponse } from "./responsesTypes";
import { ASK_IMAGE_TOOL_DEF, ASK_IMAGE_TOOL_NAME, ASK_WITH_MULTI_IMAGE_TOOL_DEF, ASK_WITH_MULTI_IMAGE_TOOL_NAME } from "../vision/types";
import { parseVisionToolHistoryPart } from "../vision/historyPart";
import { logger } from "../logger";

function visionItems(entry: { id: string; name: string; args: unknown; result: string }): ResponsesInputItem[] {
    return [
        { type: "function_call", call_id: entry.id, name: entry.name, arguments: JSON.stringify(entry.args) },
        { type: "function_call_output", call_id: entry.id, output: entry.result },
    ];
}

/** OpenAI Responses API adapter for `/responses`. */
export class ResponsesApi extends CommonApi<ResponsesInputItem, ResponsesRequestBody> {
    private _hasImages = false;

    async convertMessages(messages: readonly vscode.LanguageModelChatRequestMessage[], modelConfig: { includeReasoningInRequest: boolean; vision?: boolean }): Promise<ResponsesInputItem[]> {
        const supportsVision = modelConfig.vision !== false;
        const output: ResponsesInputItem[] = [];
        let imageIndex = 0;
        if (!supportsVision) {
            const images: Array<{ data: Uint8Array; mimeType: string }> = [];
            for (const message of messages) {
                for (const part of message.content ?? []) {
                    if (part instanceof vscode.LanguageModelDataPart && isImageMimeType(part.mimeType)) images.push({ data: part.data, mimeType: part.mimeType });
                    if (part instanceof vscode.LanguageModelTextPart) storeDataUriImages(part.value, images);
                }
            }
            if (images.length) { this._localImages = images; this._hasImages = true; }
        }
        for (const message of messages) {
            const role = mapRole(message);
            if (role === "system") {
                const text = collectToolResultText(message);
                if (text) this._systemContent = this._systemContent ? `${this._systemContent}\n${text}` : text;
                continue;
            }
            const text: string[] = [];
            const content: ResponsesInputContent[] = [];
            const toolOutputs: ResponsesInputItem[] = [];
            for (const part of message.content ?? []) {
                const history = parseVisionToolHistoryPart(part);
                if (history) { output.push(...visionItems(history)); continue; }
                if (part instanceof vscode.LanguageModelTextPart) {
                    const converted = supportsVision ? { text: part.value, count: 0 } : replaceDataUriImages(part.value, imageIndex);
                    imageIndex += converted.count;
                    text.push(converted.text);
                } else if (part instanceof vscode.LanguageModelDataPart && isImageMimeType(part.mimeType)) {
                    if (supportsVision) content.push({ type: "input_image", image_url: createDataUrl(part), detail: "auto" });
                    else { text.push(`[The user sent an image (imageIndex=${imageIndex}). Call ${ASK_IMAGE_TOOL_NAME}.]`); imageIndex++; }
                } else if (part instanceof vscode.LanguageModelToolCallPart) {
                    output.push({ type: "function_call", call_id: part.callId, name: part.name, arguments: JSON.stringify(part.input ?? {}) });
                } else if (isToolResultPart(part)) {
                    const values: string[] = [];
                    for (const inner of part.content ?? []) {
                        if (inner instanceof vscode.LanguageModelTextPart) values.push(inner.value);
                        else if (inner instanceof vscode.LanguageModelDataPart && isImageMimeType(inner.mimeType)) values.push(supportsVision ? `[Image: ${createDataUrl(inner)}]` : `[Image data (imageIndex=${imageIndex}); call ${ASK_IMAGE_TOOL_NAME}.]`);
                        else if (inner instanceof vscode.LanguageModelDataPart && isResourceLinkMimeType(inner.mimeType)) {
                            const resolved = await resolveResourceLinkToImage(inner.data);
                            values.push(resolved ? `[Image: ${createDataUrl(new vscode.LanguageModelDataPart(resolved.data, resolved.mimeType))}]` : `[Resource link: ${parseResourceLinkData(inner.data)?.uri ?? "unavailable"}]`);
                        }
                    }
                    toolOutputs.push({ type: "function_call_output", call_id: part.callId, output: values.join("\n") });
                }
            }
            const joined = text.join("").trim();
            if (role === "user" || role === "assistant") {
                if (joined || content.length) output.push({ role, content: [...content, ...(joined ? [{ type: "input_text" as const, text: joined }] : [])] });
            }
            output.push(...toolOutputs);
        }
        this._originalApiMessages = output;
        return output;
    }

    prepareRequestBody(body: ResponsesRequestBody, model: OpenCodeGoModelItem | undefined, options?: vscode.ProvideLanguageModelChatResponseOptions): ResponsesRequestBody {
        if (this._systemContent) body.instructions = this._systemContent;
        if (model?.supportsTemperature !== false && model?.temperature != null) body.temperature = model.temperature;
        if (model?.top_p != null) body.top_p = model.top_p;
        if (model?.max_completion_tokens !== undefined) body.max_output_tokens = model.max_completion_tokens;
        else if (model?.max_tokens !== undefined) body.max_output_tokens = model.max_tokens;
        if (model?.enable_thinking !== false && model?.reasoning_effort && !["enabled", "adaptive"].includes(model.reasoning_effort)) body.reasoning = { effort: model.reasoning_effort };
        const converted = convertToolsToOpenAI(options);
        const tools: Array<Record<string, unknown>> = (converted.tools ?? []).map((tool) => ({ type: "function", name: tool.function.name, description: tool.function.description, parameters: tool.function.parameters }));
        if (this._hasImages) {
            tools.push({ type: "function", name: ASK_IMAGE_TOOL_DEF.function.name, description: ASK_IMAGE_TOOL_DEF.function.description, parameters: ASK_IMAGE_TOOL_DEF.function.parameters });
            if (this._localImages.length >= 2) tools.push({ type: "function", name: ASK_WITH_MULTI_IMAGE_TOOL_DEF.function.name, description: ASK_WITH_MULTI_IMAGE_TOOL_DEF.function.description, parameters: ASK_WITH_MULTI_IMAGE_TOOL_DEF.function.parameters });
        }
        if (tools.length) body.tools = tools;
        if (this._hasImages) body.tool_choice = "auto";
        return body;
    }

    async processResponse(response: Response, progress: vscode.Progress<vscode.LanguageModelResponsePart>): Promise<void> {
        const data = await response.json() as ResponsesResponse;
        const text = data.output_text ?? data.output?.flatMap((item) => item.content ?? []).filter((part) => part.type === "output_text" && part.text).map((part) => part.text).join("") ?? "";
        if (text) this.processTextContent(text, progress);
        for (const item of data.output ?? []) {
            if (item.type === "function_call" && item.name && item.arguments) {
                const parsed = tryParseJSONObject(item.arguments);
                if (parsed.ok && item.call_id && item.name !== ASK_IMAGE_TOOL_NAME && item.name !== ASK_WITH_MULTI_IMAGE_TOOL_NAME) progress.report(new vscode.LanguageModelToolCallPart(item.call_id, item.name, parsed.value));
                else if (item.call_id && item.name && parsed.ok) this.interceptedToolCall = { id: item.call_id, name: item.name as typeof ASK_IMAGE_TOOL_NAME, args: parsed.value as never };
            }
        }
        const usage = data.usage;
        if (usage) this._onUsage?.({ promptTokens: usage.input_tokens ?? 0, completionTokens: usage.output_tokens ?? 0, cacheHitTokens: usage.input_tokens_details?.cached_tokens });
        this.reportEndThinking(progress);
    }

    async processStreamingResponse(body: ReadableStream<Uint8Array>, progress: vscode.Progress<vscode.LanguageModelResponsePart>, token: vscode.CancellationToken): Promise<void> {
        this._resetStreamState();
        const reader = body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        const disposable = token.onCancellationRequested(() => { void reader.cancel(); });
        try {
            while (!token.isCancellationRequested) {
                const { done, value } = await reader.read(); if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n"); buffer = lines.pop() ?? "";
                for (const line of lines) {
                    if (!line.startsWith("data:")) continue;
                    const raw = line.slice(5).trim(); if (raw === "[DONE]") continue;
                    try {
                        const event = JSON.parse(raw) as { type?: string; delta?: string; response?: ResponsesResponse; item?: { type?: string; call_id?: string; name?: string; arguments?: string } };
                        if (event.type === "response.output_text.delta" && event.delta) this.processTextContent(event.delta, progress);
                        else if ((event.type === "response.reasoning_summary_text.delta" || event.type === "response.reasoning_text.delta") && event.delta) this.bufferThinkingContent(event.delta, progress);
                        else if (event.type === "response.completed" && event.response?.usage) this._onUsage?.({ promptTokens: event.response.usage.input_tokens ?? 0, completionTokens: event.response.usage.output_tokens ?? 0, cacheHitTokens: event.response.usage.input_tokens_details?.cached_tokens });
                    } catch (error) { logger.warn("responses.stream.chunk.error", { error: String(error) }); }
                }
            }
        } finally { disposable.dispose(); reader.releaseLock(); this.reportEndThinking(progress); }
    }
}
