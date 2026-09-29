---
name: create-ai-labeling-block
description: "Create, adapt, test, or document an Edge Impulse custom AI labeling block for zero-shot bounding box labeling, including GPT-6 Astra vision requests, parameters.json, --data-ids-file, --propose-actions preview mode, Studio raw-data API updates, and publishing readiness. Use when adding a labeling provider or model to this repository."
argument-hint: "Describe the provider/model and its documented vision API contract"
---

# Create an AI Labeling Block

Use this skill to add or change a provider-backed AI labeling block in this repository. The active block is the workspace root.

## Required Facts

The Astra implementation has a confirmed provider contract: use `OPENAI_API_KEY`, call OpenAI's Responses API with `model: "gpt-6-astra"`, send the image as an `input_image` data URL with `"detail": "original"`, request a strict `json_schema` through `text.format`, and parse the `output_text` of the `message` output item. Boxes come back in pixel coordinates of the image as sent. Obtain provider documentation before adding any other model, detail level, or provider, and record its image input limits and coordinate convention.

Read [AGENTS.md](../../../AGENTS.md) before editing. For current platform details, first consult `https://docs.edgeimpulse.com/llms.txt`, then retrieve the relevant pages for custom AI labeling blocks, `parameters.json`, and `ids.json`.

## Procedure

1. Confirm `edge-impulse-blocks` is installed, Docker is responding, and root `parameters.json` has `"type": "ai-action"` with `images_object_detection` in `info.operatesOn`.
2. If the block has not been initialized, run `edge-impulse-blocks init` in the repository root and choose an AI labeling block. Do not reinitialize an existing block.
3. Define the Studio form in `parameters.json` before implementing code. Keep normal user settings as parameter items, use `secret` for a per-job credential, and use `info.requiredEnvVariables` for a block-level credential configured on push.
4. Keep the Docker entry point on `labeling.py`; use plain HTTP with `requests` unless a documented dependency simplifies a required provider feature.
5. Parse every declared non-secret parameter plus `--data-ids-file` and `--propose-actions`. Validate inputs and `OPENAI_API_KEY` before calling the provider.
6. For each sample ID, fetch the sample and image from the Studio API, resize only when the provider limit requires it, request detections, clamp and rescale boxes to original pixels, then apply size filters and per-label NMS.
7. In preview mode, post to `propose-changes` with the job ID. Otherwise post to `bounding-boxes` and `metadata`. Include model, settings, and prompt in metadata.
8. Build and test with Docker directly. AI labeling blocks do not support `edge-impulse-blocks runner`. Do not use `--network=none` when the test calls an external provider. Test preview mode in Studio with **Label preview data** before labeling a full dataset.
9. Report the boxes produced, the Studio result, and any remaining provider limitation. Push only after the user explicitly requests it.

## Security

Never ask the user to paste an API key into chat, source files, `parameters.json`, commit messages, or logs. Have the user enter secrets directly into the terminal or configure them in Studio. Keep `.env`, `.ei-block-config`, and `ids.json` out of version control.
