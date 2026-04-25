import json
import base64
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import io
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from utils.logger import make_log


OVERLAY_ANALYSIS_SYSTEM = """You are an expert document analysis AI specializing in precise field localization.

Return STRICT JSON only.

Rules:
- bbox must be tightly fitted to visible text
- Use normalized coordinates (0–1)
- Never exceed boundaries
- Avoid large boxes
- Prefer smaller, tighter boxes
- confidence < 0.6 if unsure

Format:
{
  "field_name": {
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.0-1.0,
    "notes": ""
  }
}
"""


# -------------------------------
# 🔧 BBOX SANITIZATION
# -------------------------------
def sanitize_bbox(bbox):
    try:
        x1, y1, x2, y2 = bbox

        x1, x2 = sorted([max(0, min(1, x1)), max(0, min(1, x2))])
        y1, y2 = sorted([max(0, min(1, y1)), max(0, min(1, y2))])

        if (x2 - x1) < 0.005 or (y2 - y1) < 0.005:
            return None

        return [x1, y1, x2, y2]
    except:
        return None


# -------------------------------
# 🔧 SNAP TO TEXT (IMPORTANT)
# -------------------------------
def refine_bbox_to_text(img, bbox):
    """Refine bbox using image contrast to snap to actual text"""
    w, h = img.size

    x1, y1, x2, y2 = [int(v) for v in [
        bbox[0] * w,
        bbox[1] * h,
        bbox[2] * w,
        bbox[3] * h,
    ]]

    crop = img.crop((x1, y1, x2, y2)).convert("L")
    crop = ImageOps.autocontrast(crop)

    arr = np.array(crop)

    # Detect dark text pixels
    mask = arr < 200

    coords = np.argwhere(mask)

    if coords.size == 0:
        return bbox  # fallback

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Expand slightly (smart padding)
    pad = 3
    x1_new = max(0, x1 + x_min - pad)
    y1_new = max(0, y1 + y_min - pad)
    x2_new = min(w, x1 + x_max + pad)
    y2_new = min(h, y1 + y_max + pad)

    return [
        x1_new / w,
        y1_new / h,
        x2_new / w,
        y2_new / h,
    ]


# -------------------------------
# 🔧 MAIN REFINEMENT
# -------------------------------
def refine_overlay_bboxes(document_base64: str, extracted_fields: dict, current_bboxes: dict) -> dict:
    logs = []

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
        )

        fields_list = "\n".join([f"- {k}: {v}" for k, v in extracted_fields.items() if v])

        prompt = f"""{OVERLAY_ANALYSIS_SYSTEM}

Fields:
{fields_list}
"""

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{document_base64}"
                    },
                },
                {"type": "text", "text": prompt},
            ]
        )

        response = llm.invoke([message])

        import re
        raw = response.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        model_output = json.loads(raw)

        # Load image for refinement
        img_data = base64.b64decode(document_base64)
        img = Image.open(io.BytesIO(img_data))

        refined = {}

        for field, data in model_output.items():
            if not isinstance(data, dict):
                continue

            bbox = sanitize_bbox(data.get("bbox"))
            conf = data.get("confidence", 0)

            # Reject low confidence
            if not bbox or conf < 0.5:
                continue

            # 🔥 SNAP TO ACTUAL TEXT
            bbox = refine_bbox_to_text(img, bbox)

            refined[field] = {
                "bbox": bbox,
                "confidence": conf,
                "notes": data.get("notes", "")
            }

        logs.append(make_log("OverlayAgent", "BBOX_REFINED", f"{len(refined)} boxes refined", "SUCCESS"))

        return refined, logs

    except Exception as e:
        logs.append(make_log("OverlayAgent", "REFINEMENT_FAILED", str(e), "WARNING"))
        return current_bboxes, logs


# -------------------------------
# 🎯 DRAWING (IMPROVED PRECISION)
# -------------------------------
def generate_precision_overlay_image(
    original_b64: str,
    final_results: dict,
    refined_bboxes: dict = None,
    human_decisions: dict = None,
) -> str:

    human_decisions = human_decisions or {}

    img_data = base64.b64decode(original_b64)
    img = Image.open(io.BytesIO(img_data)).convert("RGBA")

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = img.size

    colors = {
        "verified": (0, 200, 150, 255),
        "invalid": (255, 71, 87, 255),
        "unverifiable": (255, 179, 0, 255),
        "human_approved": (0, 200, 150, 255),
        "human_rejected": (255, 71, 87, 255),
    }

    fill_colors = {
        "verified": (0, 200, 150, 40),
        "invalid": (255, 71, 87, 40),
        "unverifiable": (255, 179, 0, 40),
        "human_approved": (0, 200, 150, 40),
        "human_rejected": (255, 71, 87, 40),
    }

    for field_name, field_data in final_results.items():

        if refined_bboxes and field_name in refined_bboxes:
            bbox = refined_bboxes[field_name]["bbox"]
        else:
            bbox = field_data.get("bbox")

        if not bbox:
            continue

        status = field_data.get("status", "unverifiable")

        if field_name in human_decisions:
            decision = human_decisions[field_name].get("decision")
            status = "human_approved" if decision == "approve" else "human_rejected"

        bbox = sanitize_bbox(bbox)
        if not bbox:
            continue

        # Convert to pixel (NO rounding drift)
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)

        if x2 <= x1 or y2 <= y1:
            continue

        # Draw fill
        draw.rectangle([x1, y1, x2, y2], fill=fill_colors[status])

        # Draw sharp border
        for i in range(3):  # multi-layer crisp border
            draw.rectangle(
                [x1 - i, y1 - i, x2 + i, y2 + i],
                outline=colors[status]
            )

    # Merge overlay
    final = Image.alpha_composite(img, overlay)

    output = io.BytesIO()
    final.save(output, format="PNG")
    output.seek(0)

    return base64.b64encode(output.getvalue()).decode("utf-8")