<div align="center">

![logo](/assets/logo.png)

# Multi Provider for Copilot / Opencode Go - Cline Pass - Ollama Cloud - NanoGPT


**Use OpenCode Go, Cline Pass, Ollama Cloud and NanoGPT models directly inside GitHub Copilot Chat.**

![VS Code](https://img.shields.io/badge/VS%20Code-%5E1.116.0-007ACC.svg)
![Version](https://img.shields.io/badge/version-1.0.0-4B8BBE.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

</div>

> [!IMPORTANT]
> **This extension is not affiliated with, officially maintained by, or endorsed by OpenCode, Cline, Ollama, or NanoGPT.**

> [!NOTE]
> **Fork notice:** this project is a fork of [OnesoftQwQ/opencode-go-copilot](https://github.com/OnesoftQwQ/opencode-go-copilot).

---

## Overview

A VS Code extension that integrates multiple AI model providers into **GitHub Copilot Chat**, so you can chat with state-of-the-art LLMs — with streaming responses, thinking/reasoning display, tool calling and image support — without leaving your editor.

---

## Features

| Feature | Description |
|---------|-------------|
| **Automatic model discovery** | Model lists are pulled from the `models.dev` catalog (with a mirror and a built-in fallback snapshot), so new models appear without updating the extension. Capabilities (context length, vision, thinking mode, reasoning effort, temperature support, endpoint) are resolved automatically — no hardcoded model list. |
| **Streaming chat with thinking** | Responses stream live with reasoning/thinking display. Models with switchable thinking expose reasoning effort levels (Disabled / Low / Medium / High / Extra High / Maximum), and `grok-4.5` runs through the OpenAI Responses API with thinking always on. |
| **Tool calling** | Full support for VS Code tool calls (e.g. `read_file`) and tools exposed by MCP servers. |
| **Vision for text-only models** | Non-vision models can still "see" images by calling a vision-capable helper model through the `ask_image` tool, asking specific questions about the image and answering from its description (multi-image comparison via `ask_with_multi_image`). |
| **Image support from MCP tools** | Screenshots and other images returned by MCP tools are read and sent directly to vision models. |
| **Token usage indicators** | Reports usage to the native Copilot token indicator, plus an optional advanced status-bar counter with cumulative input/output tokens and cache hit rate. |
| **Git commit messages** | One click on the magic-wand button in the Source Control panel generates a Conventional Commit message. Language is auto-detected from your commit history (or fixed, e.g. Turkish). |
| **Temperature presets** | Quick switch between Precise / Balanced / Creative / Extra Creative presets, or set your own `temperature` and `top_p`. |
| **Reliability** | Configurable request timeout, retries with exponential backoff, optional inter-request delay to avoid rate limits, and HTTP security checks (HTTPS enforced for remote endpoints). |
| **Localization** | Bilingual interface — English and Simplified Chinese. |

---

## Supported Providers

| Provider | Endpoint | API Mode | Model List |
|----------|----------|----------|------------|
| **OpenCode Go** | `opencode.ai/zen/go/v1` | OpenAI Chat Completions, Anthropic Messages, OpenAI Responses | Auto-discovered from `models.dev` (OpenCode Go catalog) |
| **OpenCode Zen** *(optional)* | `opencode.ai` | OpenAI-compatible | Free models from the catalib (`-free` suffix), disabled by default |
| **Cline Pass** | `api.cline.bot/api/v1` | OpenAI Chat Completions | 11 curated open-source models |
| **Ollama Cloud** | `ollama.com/api/chat` | Ollama-native API (NDJSON streaming) | 15 curated open-weight models |
| **NanoGPT** | `nano-gpt.com/api/subscription/v1` | OpenAI Chat Completions | Fetched live from the NanoGPT API |

### OpenCode Go
The main provider. The model list is 100% catalog-driven — new models appear automatically. Families include **GLM** (`glm-5`, `5.1`, `5.2`), **Kimi** (`k3`, `k2.7-code`, `k2.6`, `k2.5`), **DeepSeek** (`v4-pro`, `v4-flash`), **MiMo**, **MiniMax** (`m3`, `m2.7`, `m2.5`), **Qwen** (`qwen3.8-max`, `3.7-max/plus`, `3.6-plus`, `3.5-plus`), plus **gpt-5.6-luna**, **grok-4.5** and **hy3**. Reasoning strengths auto-derive from the catalog manifest (`reasoning_options`).

### OpenCode Zen (optional free models)
Toggle `opencodego.enableZenFreeModels` in settings to append free `-free` models (e.g. `big-pickle`, `deepseek-v4-flash-free`, `minimax-m3-free`, `nemotron-3-super-free`) to the model picker, labeled **OpenCode Zen**.

### Cline Pass
A monthly subscription with **11 open-source models**: GLM 5.2, Kimi K3 / K2.7 / K2.6, DeepSeek V4 Pro / Flash, MiMo V2.5 / V2.5-Pro, MiniMax M3, Qwen3.7 Max / Plus — served through one OpenAI-compatible endpoint.

### Ollama Cloud
**15 open-weight models** including **GPT-OSS 120B**, **Gemma 4 31B**, **Nemotron 3** (super / ultra), **Mistral Large 3**, GLM 5.1/5.2, Kimi K3 / K2.7 / K2.6, DeepSeek V4 Pro / Flash, MiniMax M3 / M2.7, Qwen 3.5. Uses the native Ollama streaming API (`/api/chat`, newline-delimited JSON) with `think`-based reasoning levels.

### NanoGPT
Flat-rate subscription. The **entire model catalog is fetched live from the NanoGPT API** (`/models?detailed=true`), including capability metadata, context length and reasoning effort levels. IDs follow `provider/model` format, with `:thinking`-suffixed reasoning variants.

---

## Quick Start

> **Requires VS Code 1.116+.**

1. **Install** the extension.
2. **Set your API key(s)** — `Ctrl+Shift+P` and run the matching command for each provider you want to use:

   | Provider | Set Key Command |
   |----------|-----------------|
   | OpenCode Go | `OpenCodeGo: Set OpenCode Go API Key` |
   | Cline Pass | `ClinePass: Set Cline Pass API Key` |
   | Ollama Cloud | `OllamaCloud: Set Ollama Cloud API Key` |
   | NanoGPT | `NanoGPT: Set NanoGPT API Key` |

3. **Show the models** — in the Copilot Chat model picker, click the settings icon, open the **Language Models** panel and set the models you want to **Visible**.
4. **Select a model** in the model picker — the providers appear as "OpenCode Go", "Cline Pass", "Ollama Cloud" and "NanoGPT".
5. **Chat!**

**Tip:** use the **Generate Git Commit Message** button (magic wand) in the Source Control panel to write commit messages for you.

---

## Screenshots

Advanced token indicator in the VS Code status bar (cumulative input/output tokens, cache hit rate):

![Token indicator](/assets/screenshots/token_counter.png)

---

## Commands

| Command | Description |
|---------|-------------|
| `OpenCodeGo: Set OpenCode Go API Key` | Store your OpenCode Go API key (SecretStorage) |
| `OpenCodeGo: Get OpenCode Go API Key` | Open the OpenCode AI website to get a key |
| `OpenCodeGo: Open Settings` | Open the extension settings page |
| `OpenCodeGo: Set Model Preset` | Switch temperature preset (Precise / Balanced / Creative / Extra Creative / custom) |
| `OpenCodeGo: Update Model List` | Force-refresh the model list (models.dev catalog + API availability) |
| `OpenCodeGo: Generate Git Commit Message` | Generate a conventional commit message for the current repo |
| `OpenCodeGo: Abort Git Commit Message` | Stop the running commit generation |
| `ClinePass: Set Cline Pass API Key` | Store your Cline Pass API key |
| `ClinePass: Get Cline Pass API Key` | Open the Cline website to get a key |
| `OllamaCloud: Set Ollama Cloud API Key` | Store your Ollama Cloud API key |
| `OllamaCloud: Get Ollama Cloud API Key` | Open the Ollama website to get a key |
| `NanoGPT: Set NanoGPT API Key` | Store your NanoGPT API key |
| `NanoGPT: Get NanoGPT API Key` | Open the NanoGPT website to get a key |

---

## Settings

Key configuration options (all under the `opencodego.*` namespace unless noted):

| Setting | Default | Description |
|---------|---------|-------------|
| `opencodego.enableZenFreeModels` | `false` | Append OpenCode Zen free models (`-free`) to the picker |
| `opencodego.enableAutoModelDiscovery` | `true` | Filter the model picker by the models actually available from the API |
| `opencodego.modelPreset` | `precise` | Temperature preset (`precise` / `balanced` / `creative` / `extra-creative` / `custom`) |
| `opencodego.temperature` / `opencodego.top_p` | — | Custom sampling parameters (used with `custom` preset) |
| `opencodego.requestTimeout` | `600000` | Request timeout in ms (default 10 minutes) |
| `opencodego.delay` | `0` | Optional delay before each request, to avoid rate limits |
| `opencodego.retry.enabled` / `max_attempts` / `interval_ms` / `status_codes` | `true` / `3` / `1000` / `[]` | Retry policy with exponential backoff |
| `opencodego.commitLanguage` | `auto` | Commit message language (auto-detects from history, or fixed, e.g. `Turkish`) |
| `opencodego.commitModel` | `deepseek-v4-flash` | Model used for commit message generation |
| `opencodego.visionProxyModel` | `qwen-plus-latest` | Vision model used by the `ask_image` proxy |
| `opencodego.visionProxyThinking` | `false` | Enable thinking for the vision proxy model |
| `opencodego.visionMaxRounds` | `5` | Max follow-up `ask_image` rounds per request |
| `opencodego.modelsDevMirrorUrl` / `modelsDevMirrorToken` | mirror URL | Fallback mirror for the models.dev catalog |
| `opencodego.showDeprecatedModels` | `false` | Show deprecated models in the picker |
| `opencodego.enableThirdPartyTokenIndicator` | `true` | Advanced status-bar token counter (native indicator always on) |
| `opencodego.readFileLines` | `0` | Auto-expand `read_file` tool requests by this many lines |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| **No models appear in the picker** | Set your API key, then run `OpenCodeGo: Update Model List`. Model discovery needs the `models.dev` catalog (or mirror/fallback) and a valid key. |
| **`Plain HTTP is only allowed for localhost or private network addresses`** | Remote endpoints must use HTTPS for security; `http://` is only accepted for local/private addresses. |
| **OpenCode Zen models show `401` errors** | Zen free models can expire; run `OpenCodeGo: Update Model List` after re-enabling them, or check your API key. |
| **`IMAGE_SENSITIVE` error** | The image you sent was flagged by content moderation — try a different image. |
| **Git commit message in the wrong language** | Set `opencodego.commitLanguage` explicitly (e.g. `Turkish`) instead of `auto`. |

---

## Build from Source

```bash
npm install
npm run compile   # type-check + build to ./out
npm run lint      # eslint
npm run build     # packages extension.vsix
```

Use the Extension Development Host (`F5`) to try the extension locally.

---

## License

[MIT](LICENSE) — feel free to fork, modify and share.