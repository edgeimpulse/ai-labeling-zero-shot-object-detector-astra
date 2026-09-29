# AI Labeling Block: Zero-shot Object Detection with GPT-6 Astra

<img width="128" height="128" alt="image" src="https://github.com/user-attachments/assets/56aec790-4110-4472-aa6d-0d923ed12024" />


An Edge Impulse AI labeling block that adds bounding boxes to image data using GPT-6 Astra. Describe the objects you want in plain text and the block labels every matching instance. In Studio it appears as **Bounding box labeling with GPT-6 Astra**.

Custom AI labeling blocks require the Edge Impulse Enterprise plan.

<img width="1804" height="910" alt="image" src="https://github.com/user-attachments/assets/202a1a2a-9b7d-475b-bff4-aff5e7d26c3a" />


[labeling.py](labeling.py) sends each image to the OpenAI Responses API with `gpt-6-astra`, `"detail": "original"`, and a strict JSON schema, so every response is a list of labeled pixel boxes. Images that exceed OpenAI's 30,000-patch limit are downsized before sending and the boxes are mapped back to the original size.

OpenAI notes that vision models can struggle with precise spatial localization. Use **Label preview data** in Studio to check boxes before applying them to a full dataset.

## Requirements

- [Edge Impulse CLI](https://docs.edgeimpulse.com/tools/clis/edge-impulse-cli/installation)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- An OpenAI API key with access to GPT-6 Astra
- An Edge Impulse project API key, from **Dashboard > Keys**

## Parameters

| Studio field | Passed as | Default |
| --- | --- | --- |
| OpenAI API Key | `OPENAI_API_KEY` environment variable | none |
| Prompt | `--prompt` | `Person (person)` and `Car (vehicle)` |
| Reasoning effort | `--reasoning-effort` (`low`, `medium`, `high`, `xhigh`, `max`) | `medium` |
| Delete existing bounding boxes | `--delete_existing_bounding_boxes` (`no`, `matching-prompt`, `yes`) | `yes` |
| Ignore objects smaller than (%) | `--ignore-objects-smaller-than` | `0` |
| Ignore objects larger than (%) | `--ignore-objects-larger-than` | `100` |
| Non-max suppression | `--nms` | on |
| NMS IoU threshold | `--nms-iou-threshold` | `0.2` |

Write one object per line in the prompt, with the label in parentheses: `A red fish (fish)`. Without parentheses the whole line is used as the label.

Studio also passes `--data-ids-file` and, in preview mode, `--propose-actions <job-id>`. In preview mode the block stages changes with the propose-changes API instead of writing them.

Each labeled sample gets `labeled_by`, `reasoning_effort`, and `prompt` metadata.

## Test Locally

AI labeling blocks don't work with `edge-impulse-blocks runner`, so build and run the container directly.

1. Create `ids.json` with the sample IDs to label. Find an ID with the expand button on **Data acquisition**:

   ```json
   [1299267659, 1299267609, 1299267606]
   ```

2. Set `OPENAI_API_KEY` and `EI_PROJECT_API_KEY` in your shell. `-e NAME` with no value passes each key through without writing it into the command.

3. Build and run:

   ```bash
   docker build -t astra-ai-labeling .
   docker run --rm \
     -e OPENAI_API_KEY \
     -e EI_PROJECT_API_KEY \
     -v "$PWD/ids.json:/app/ids.json:ro" \
     astra-ai-labeling \
     --data-ids-file ids.json \
     --prompt 'Fish (fish)' \
     --delete_existing_bounding_boxes yes \
     --reasoning-effort medium \
     --nms
   ```

To run without Docker:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -u labeling.py --data-ids-file ids.json --prompt 'Fish (fish)' --nms
```

## Publish

`.ei-block-config` is gitignored. On a fresh clone, run `edge-impulse-blocks init` first and choose the AI labeling block type. Then push:

```bash
edge-impulse-blocks push
```

In an Enterprise project, open **Data acquisition > AI labeling**, pick **Bounding box labeling with GPT-6 Astra**, and run **Label preview data** on a few samples first.

## Agent Skill

[.github/skills/create-ai-labeling-block/SKILL.md](.github/skills/create-ai-labeling-block/SKILL.md) is a GitHub Copilot skill for extending this block. Copilot loads skills from `.github/skills` when this folder is open. Run `/create-ai-labeling-block` in Copilot Chat, or name the skill in a prompt:

```text
Use the create-ai-labeling-block skill to add an image detail option to the Astra block.
Keep the strict JSON schema and preview mode handling.
```

[AGENTS.md](AGENTS.md) holds the rules that apply to any change in this repository.

## Further Reading

- [Custom AI labeling blocks](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-ai-labeling-blocks)
- [AI labeling](https://docs.edgeimpulse.com/studio/projects/data-acquisition/ai-labeling)
- [parameters.json reference](https://docs.edgeimpulse.com/tools/specifications/files/parameters-json)
- [ids.json reference](https://docs.edgeimpulse.com/tools/specifications/files/ids-json)
- [GPT-6 Astra model](https://developers.openai.com/api/docs/models/gpt-6-astra)
- [OpenAI images and vision](https://developers.openai.com/api/docs/guides/images-vision)
- [Gemini version of this block](https://github.com/edgeimpulse/ai-labeling-zero-shot-object-detector-gemini)
