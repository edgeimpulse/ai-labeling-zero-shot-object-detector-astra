import argparse
import base64
import io
import json
import math
import os
import re
import sys

import requests
from PIL import Image


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = "gpt-6-astra"
VALID_REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
VALID_DELETE_MODES = {"no", "matching-prompt", "yes"}
# OpenAI rejects images that need more than 30,000 32x32 patches.
MAX_PATCHES = 30000
PATCH_SIZE = 32


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot object detector with GPT-6 Astra")
    parser.add_argument("--prompt", required=True, help='Objects to label, one per line. Format: "A person (person)"')
    parser.add_argument("--data-ids-file", required=True, help="JSON file with the sample IDs to label")
    parser.add_argument("--propose-actions", type=int, help="Job ID; stage changes instead of applying them")
    parser.add_argument(
        "--delete_existing_bounding_boxes",
        default="yes",
        choices=sorted(VALID_DELETE_MODES),
        help="Delete existing bounding boxes",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=sorted(VALID_REASONING_EFFORTS),
        help="GPT-6 Astra reasoning effort",
    )
    parser.add_argument("--ignore-objects-smaller-than", type=float, help="Ignore objects smaller than this percentage of the image area")
    parser.add_argument("--ignore-objects-larger-than", type=float, help="Ignore objects larger than this percentage of the image area")
    parser.add_argument("--nms", action="store_true", help="Apply non-maximum suppression per label")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.2, help="IoU threshold for non-maximum suppression")
    return parser.parse_args()


def response_error(response):
    try:
        payload = response.json()
    except ValueError:
        return response.text

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return error.get("message") or error.get("code") or json.dumps(error)
    return json.dumps(payload)


def parse_prompt(prompt):
    objects = []
    for line in prompt.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.search(r"\((.*?)\)", line)
        if match:
            search_for = line[: match.start()].strip()
            label = match.group(1).strip()
        else:
            search_for = line
            label = line
        if not search_for or not label:
            raise RuntimeError(f"Invalid prompt line: {line!r}")
        objects.append({"search_for": search_for, "label": label})
    if not objects:
        raise RuntimeError("--prompt must contain at least one object")
    return objects


def get_project_id(api_endpoint, api_key):
    project_id = os.getenv("EI_PROJECT_ID")
    if project_id:
        return project_id
    response = requests.get(f"{api_endpoint}/api/projects", headers={"x-api-key": api_key}, timeout=30)
    if not response.ok:
        raise RuntimeError(f"Failed to fetch Edge Impulse project ({response.status_code}): {response_error(response)}")
    projects = response.json().get("projects") or []
    if not projects:
        raise RuntimeError("EI_PROJECT_API_KEY does not give access to any project")
    return projects[0]["id"]


def check_ei_response(response, action):
    if not response.ok:
        raise RuntimeError(f"{action} failed ({response.status_code}): {response_error(response)}")
    payload = response.json()
    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(f"{action} failed: {payload.get('error') or json.dumps(payload)}")
    return payload


def fit_patch_limit(image):
    patches = math.ceil(image.width / PATCH_SIZE) * math.ceil(image.height / PATCH_SIZE)
    if patches <= MAX_PATCHES:
        return image
    scale = math.sqrt(MAX_PATCHES / patches)
    while True:
        width = max(1, int(image.width * scale))
        height = max(1, int(image.height * scale))
        if math.ceil(width / PATCH_SIZE) * math.ceil(height / PATCH_SIZE) <= MAX_PATCHES:
            return image.resize((width, height), Image.LANCZOS)
        scale *= 0.99


def build_schema(labels):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["objects"],
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["label", "confidence", "x_min", "y_min", "x_max", "y_max"],
                    "properties": {
                        "label": {"type": "string", "enum": labels},
                        "confidence": {"type": "number"},
                        "x_min": {"type": "integer"},
                        "y_min": {"type": "integer"},
                        "x_max": {"type": "integer"},
                        "y_max": {"type": "integer"},
                    },
                },
            }
        },
    }


def build_instruction(objects, width, height):
    targets = "\n".join(f"- {obj['search_for']}: use label \"{obj['label']}\"" for obj in objects)
    return (
        "Detect every instance of the following objects in the image:\n"
        f"{targets}\n"
        f"The image is {width} x {height} pixels. Return one tight bounding box per instance in pixel "
        "coordinates, with (0, 0) at the top-left corner, x_min < x_max <= width and y_min < y_max <= height. "
        "Set confidence between 0 and 1. Return an empty list if none of the objects are present."
    )


def detect_objects(openai_api_key, image, objects, reasoning_effort):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    instruction = build_instruction(objects, image.width, image.height)
    labels = sorted({obj["label"] for obj in objects})

    response = requests.post(
        OPENAI_RESPONSES_URL,
        headers={"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"},
        json={
            "model": OPENAI_MODEL,
            "reasoning": {"effort": reasoning_effort},
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {"type": "input_image", "image_url": image_url, "detail": "original"},
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "object_detections",
                    "strict": True,
                    "schema": build_schema(labels),
                }
            },
        },
        timeout=300,
    )
    if not response.ok:
        raise RuntimeError(f"OpenAI request failed ({response.status_code}): {response_error(response)}")

    payload = response.json()
    if payload.get("status") != "completed":
        reason = (payload.get("incomplete_details") or {}).get("reason")
        raise RuntimeError(f"OpenAI response status {payload.get('status')!r} ({reason or 'no reason given'})")

    for output in payload.get("output", []):
        if output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if content.get("type") == "refusal":
                raise RuntimeError(f"GPT-6 Astra refused the request: {content.get('refusal')}")
            if content.get("type") == "output_text":
                try:
                    return json.loads(content["text"])["objects"], instruction
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError("GPT-6 Astra returned invalid detection JSON") from error
    raise RuntimeError("OpenAI response did not contain output_text")


def iou(a, b):
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["width"], b["x"] + b["width"])
    y2 = min(a["y"] + a["height"], b["y"] + b["height"])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union > 0 else 0


def non_max_suppression(boxes, iou_threshold):
    kept = []
    for box in sorted(boxes, key=lambda b: b["score"], reverse=True):
        if all(iou(box, other) <= iou_threshold for other in kept):
            kept.append(box)
    return kept


def to_bounding_boxes(detections, detected_size, original_size, args):
    scale_x = original_size[0] / detected_size[0]
    scale_y = original_size[1] / detected_size[1]
    image_area = original_size[0] * original_size[1]
    per_label = {}

    for detection in detections:
        x_min = max(0, min(original_size[0], round(detection["x_min"] * scale_x)))
        y_min = max(0, min(original_size[1], round(detection["y_min"] * scale_y)))
        x_max = max(0, min(original_size[0], round(detection["x_max"] * scale_x)))
        y_max = max(0, min(original_size[1], round(detection["y_max"] * scale_y)))
        if x_max <= x_min or y_max <= y_min:
            print(f"Discarding invalid box {detection}")
            continue

        area = (x_max - x_min) * (y_max - y_min) / image_area * 100
        if args.ignore_objects_smaller_than and area < args.ignore_objects_smaller_than:
            print(f"Discarding {detection['label']} with area {area:.2f}% (smaller than {args.ignore_objects_smaller_than}%)")
            continue
        if args.ignore_objects_larger_than and area > args.ignore_objects_larger_than:
            print(f"Discarding {detection['label']} with area {area:.2f}% (larger than {args.ignore_objects_larger_than}%)")
            continue

        per_label.setdefault(detection["label"], []).append({
            "label": detection["label"],
            "x": x_min,
            "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min,
            "score": detection["confidence"],
        })

    boxes = []
    for label_boxes in per_label.values():
        if args.nms:
            label_boxes = non_max_suppression(label_boxes, args.nms_iou_threshold)
        boxes.extend({k: v for k, v in box.items() if k != "score"} for box in label_boxes)
    return boxes


def label_sample(data_id, context, args, objects):
    api_endpoint, project_id, headers, openai_api_key = context
    sample_url = f"{api_endpoint}/api/{project_id}/raw-data/{data_id}"

    sample_response = requests.get(sample_url, headers=headers, timeout=60)
    sample = check_ei_response(sample_response, f"Fetching sample {data_id}")["sample"]
    filename = sample.get("filename", "unknown")
    print(f"Labeling {filename} (ID {data_id})...", flush=True)

    image_response = requests.get(f"{sample_url}/image", headers=headers, timeout=60)
    if not image_response.ok:
        raise RuntimeError(f"Fetching image for sample {data_id} failed ({image_response.status_code})")
    try:
        image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
    except OSError as error:
        raise RuntimeError(f"Sample {data_id} is not a readable image") from error

    detected_image = fit_patch_limit(image)
    detections, instruction = detect_objects(openai_api_key, detected_image, objects, args.reasoning_effort)
    print(f"Detected {len(detections)} objects: {json.dumps(detections)}")

    existing = sample.get("boundingBoxes") or []
    if args.delete_existing_bounding_boxes == "no":
        bounding_boxes = list(existing)
    elif args.delete_existing_bounding_boxes == "matching-prompt":
        prompt_labels = {obj["label"] for obj in objects}
        bounding_boxes = [bb for bb in existing if bb.get("label") not in prompt_labels]
    else:
        bounding_boxes = []
    bounding_boxes.extend(to_bounding_boxes(detections, detected_image.size, image.size, args))

    metadata = dict(sample.get("metadata") or {})
    metadata.update({
        "labeled_by": OPENAI_MODEL,
        "reasoning_effort": args.reasoning_effort,
        "prompt": instruction,
    })

    if args.propose_actions:
        response = requests.post(
            f"{sample_url}/propose-changes",
            headers=headers,
            json={
                "jobId": args.propose_actions,
                "proposedChanges": {"boundingBoxes": bounding_boxes, "metadata": metadata},
            },
            timeout=60,
        )
        check_ei_response(response, f"Proposing changes for sample {data_id}")
        print(f"Proposed {len(bounding_boxes)} bounding boxes for {filename}")
        return

    response = requests.post(
        f"{sample_url}/bounding-boxes", headers=headers, json={"boundingBoxes": bounding_boxes}, timeout=60
    )
    check_ei_response(response, f"Updating bounding boxes for sample {data_id}")
    response = requests.post(f"{sample_url}/metadata", headers=headers, json={"metadata": metadata}, timeout=60)
    check_ei_response(response, f"Updating metadata for sample {data_id}")
    print(f"Updated {filename} with {len(bounding_boxes)} bounding boxes")


def main():
    args = parse_arguments()
    objects = parse_prompt(args.prompt.replace("\\n", "\n"))
    for name in ("ignore_objects_smaller_than", "ignore_objects_larger_than"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 100:
            raise RuntimeError(f"--{name.replace('_', '-')} must be between 0 and 100")
    if not 0 <= args.nms_iou_threshold <= 1:
        raise RuntimeError("--nms-iou-threshold must be between 0 and 1")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    ei_api_key = os.getenv("EI_PROJECT_API_KEY")
    if not ei_api_key:
        raise RuntimeError("Missing EI_PROJECT_API_KEY")
    api_endpoint = os.getenv("EI_API_ENDPOINT", "https://studio.edgeimpulse.com/v1").rstrip("/")

    try:
        with open(args.data_ids_file, "r") as f:
            data_ids = json.load(f)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Failed to load data IDs from {args.data_ids_file}: {error}") from error
    # The ids.json spec is {"ids": [...]}; a bare array is also accepted.
    if isinstance(data_ids, dict):
        data_ids = data_ids.get("ids")
    if not isinstance(data_ids, list) or not all(isinstance(i, int) for i in data_ids):
        raise RuntimeError('--data-ids-file must contain {"ids": [...]} or a JSON array of integer sample IDs')

    project_id = get_project_id(api_endpoint, ei_api_key)
    headers = {"x-api-key": ei_api_key, "Accept": "*/*"}
    context = (api_endpoint, project_id, headers, openai_api_key)

    failed = []
    for data_id in data_ids:
        try:
            label_sample(data_id, context, args, objects)
        except (requests.RequestException, RuntimeError) as error:
            print(f"Failed to label sample {data_id}: {error}", file=sys.stderr, flush=True)
            failed.append(data_id)

    if failed:
        raise RuntimeError(f"{len(failed)} of {len(data_ids)} samples failed: {failed}")
    print("All done!")


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, RuntimeError, ValueError) as error:
        print(f"Astra labeling failed: {error}", file=sys.stderr)
        sys.exit(1)
