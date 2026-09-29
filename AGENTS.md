# Astra AI Labeling Block Instructions

This repository implements an Edge Impulse custom AI labeling block. It is a standalone transformation block (`"type": "ai-action"`) that reads samples from a project, asks GPT-6 Astra for bounding boxes, and writes them back through the Studio API. Read the [custom AI labeling blocks documentation](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-ai-labeling-blocks.md) before changing the provider adapter.

## Setup Gate

Before editing block code or testing a provider, complete these checks and report the result:

1. Confirm `edge-impulse-blocks` is installed and on `PATH`.
2. Confirm Docker Desktop is installed and the Docker daemon responds.
3. Confirm `parameters.json` declares `"type": "ai-action"` and `info.operatesOn` includes `images_object_detection`.
4. If `.ei-block-config` is absent, run `edge-impulse-blocks init` from the repository root and choose an AI labeling block. Do not reinitialize an existing block.

Do not request credentials in chat. The user must enter keys directly in the terminal or configure them in Studio.

## Provider Adapter Rules

The Astra provider contract is documented. Send `POST` requests to OpenAI's Responses API (`https://api.openai.com/v1/responses`) with `model: "gpt-6-astra"` and a user message containing `input_text` plus an `input_image` data URL with `"detail": "original"`. Request a strict `json_schema` in `text.format` and read the JSON from the `output_text` content of the `message` output item. Authenticate with `OPENAI_API_KEY`. `reasoning.effort` accepts `low`, `medium`, `high`, `xhigh`, and `max`.

Coordinates are pixel positions in the image as sent. With `original` detail, GPT-6 Astra keeps the image dimensions but rejects images needing more than 30,000 32x32 patches, so `labeling.py` downsizes those first and maps boxes back to the original size. OpenAI documents that vision models struggle with precise spatial localization, so treat boxes as proposals and recommend preview mode.

For a different provider, model, or undocumented setting, find the facts in the provider documentation or ask the user before changing the adapter.

For each model/provider implementation:

1. Add a user-facing parameter to `parameters.json` for every runtime option the user must choose.
2. Parse every parameter in `labeling.py` with the same `param` name as its command-line flag.
3. Use the existing `OPENAI_API_KEY` `secret` parameter for the OpenAI credential. Use a `secret` parameter for a new per-job credential, or `info.requiredEnvVariables` for a block-level value supplied during `edge-impulse-blocks push`.
4. Read `EI_PROJECT_API_KEY`, `EI_API_ENDPOINT`, and `EI_PROJECT_ID` from the environment. Never hard-code an Edge Impulse API key or provider secret.
5. Parse `--data-ids-file` (the `ids.json` format is `{"ids": [...]}`; also accept a bare array) and `--propose-actions`. When `--propose-actions <job-id>` is passed, stage changes through `POST /api/{projectId}/raw-data/{sampleId}/propose-changes` and never write bounding boxes or metadata directly.
6. Emit Edge Impulse bounding boxes as `{label, x, y, width, height}` in original-image pixels. Keep the `delete_existing_bounding_boxes`, object size filters, and NMS behavior.
7. Record the provider model, generation settings, and prompt in sample metadata. Metadata values must be strings.
8. Fail clearly on invalid parameters, missing environment variables, unreadable images, provider errors, refusals, incomplete responses, and unsuccessful Studio API responses. Exit non-zero if any sample fails.

## Test and Publish

AI labeling blocks cannot be tested with `edge-impulse-blocks runner`. Build and run the Docker image directly. The user sets `OPENAI_API_KEY` and `EI_PROJECT_API_KEY` in their shell, and `-e NAME` passes them through:

```bash
docker build -t astra-ai-labeling .
docker run --rm -e EI_PROJECT_API_KEY -e OPENAI_API_KEY \
  -v "$PWD/ids.json:/app/ids.json:ro" \
  astra-ai-labeling --data-ids-file ids.json --prompt 'Fish (fish)' \
  --delete_existing_bounding_boxes yes --reasoning-effort medium --nms
```

Use Docker's default network. Do not add `--network=none` to a test that must reach OpenAI and Edge Impulse.

After a successful local test, `edge-impulse-blocks push` publishes the current block. Only push when the user has explicitly asked to publish or approved that side effect.
