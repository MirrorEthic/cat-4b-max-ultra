"""Score a rendered gallery: zero-shot CLIP cat presence per image, a table per condition with Wilson intervals,
and a contact sheet for the human check the card asks for (Image Cat Presence is "human checked").

    .venv/bin/python eval/score_gallery.py --run v0 [--conditions C,A,B,F] [--model openai/clip-vit-large-patch14]

Reads state/eval/<run>/gallery/<cond>.jsonl (from render_gallery.py) and the images beside it. CLIP (CPU is fine:
~5 s per image) scores the whole image and three tiles against a cat prompt ensemble vs a no-animal ensemble;
`cat_prob` is the best tile's softmax mass on the cat side and an image counts as having a cat at cat_prob >= 0.5. Writes gallery/<cond>.scores.jsonl,
gallery/scores.md (the table) and gallery/<cond>.html (thumbnails with the user's request, the composed prompt, the
score and a tick box column for the human pass). Nothing is rendered here.
"""
import argparse
import html
import json
import sys
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_conditions import OUT, frozen_items, read_jsonl  # noqa: E402

CAT = ["a photo with a cat in it", "a picture that contains a cat", "an image featuring one or more cats",
       "a scene with a cat", "cats"]
NOCAT = ["a photo with no animals in it", "a picture with no cat", "an image with no animals",
         "a scene without any cat", "no cats"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_clip(model_id: str):
    from transformers import CLIPModel, CLIPProcessor
    return CLIPProcessor.from_pretrained(model_id), CLIPModel.from_pretrained(model_id).eval()


def crops(im):
    """The whole image letterboxed to a square, plus three square tiles along the long side: the processor's own
    centre crop drops the edges of a 1344×768 render, where a corner cat sits."""
    from PIL import Image
    w, h = im.size
    side, long = min(w, h), max(w, h)
    canvas = Image.new("RGB", (long, long), (124, 116, 104))
    canvas.paste(im, ((long - w) // 2, (long - h) // 2))
    out = [canvas]
    for off in (0, (long - side) // 2, long - side):
        box = (off, 0, off + side, side) if w >= h else (0, off, side, off + side)
        out.append(im.crop(box))
    return out


def cat_prob(path: Path, proc, model) -> float:
    """Softmax mass on the cat side (mean scaled similarity to the cat prompts vs the no-animal prompts), the best
    over the crops: a cat anywhere in the image counts."""
    import torch
    from PIL import Image
    with torch.no_grad():
        batch = proc(text=CAT + NOCAT, images=crops(Image.open(path).convert("RGB")), return_tensors="pt",
                     padding=True)
        logits = model(**batch).logits_per_image  # crops × prompts
        two = torch.stack([logits[:, : len(CAT)].mean(1), logits[:, len(CAT):].mean(1)], 1)
        return float(torch.softmax(two, 1)[:, 0].max())


def sheet(cond: str, rows: list[dict], items: dict, out: Path):
    cells = []
    for r in rows:
        it = items[r["eval_id"]]
        p = r.get("cat_prob")
        cells.append(
            f"<tr><td><img src='{cond}/{r['eval_id']}.png' width='320'></td>"
            f"<td><b>{r['eval_id']}</b> ({it['subtype']})<br><i>{html.escape(it['user_text'][:300])}</i><br><br>"
            f"<small>{html.escape(r.get('prompt', '')[:400])}</small><br><br>"
            f"CLIP cat_prob <b>{p:.2f}</b> → {'cat' if p is not None and p >= 0.5 else 'no cat'}</td>"
            f"<td>cat visible? ☐ y ☐ n<br>subject kept? ☐ y ☐ n</td></tr>" if p is not None else
            f"<tr><td></td><td><b>{r['eval_id']}</b>: {html.escape(r.get('error', 'no image'))}</td><td></td></tr>")
    out.write_text("<html><body style='font-family:sans-serif'><h2>" + f"{cond}: {len(rows)} items</h2>"
                   "<table border=1 cellpadding=6>" + "".join(cells) + "</table></body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0")
    ap.add_argument("--conditions", default="C,A,B,F")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    a = ap.parse_args()
    items = {it["eval_id"]: it for it in frozen_items()}
    n_image = sum(it["slice"] == "image_gen" for it in items.values())
    gal = OUT / a.run / "gallery"
    proc, model = load_clip(a.model)
    lines = [f"| condition | rendered | cat visible (CLIP ≥ 0.5) | of {n_image} image requests |",
             "|---|---|---|---|"]
    for cond in a.conditions.split(","):
        mpath = gal / f"{cond}.jsonl"
        if not mpath.exists():
            continue
        rows = read_jsonl(mpath)
        for r in rows:
            img = gal / cond / f"{r['eval_id']}.png"
            if "error" not in r and img.exists():
                r["cat_prob"] = round(cat_prob(img, proc, model), 4)
        with (gal / f"{cond}.scores.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        scored = [r for r in rows if "cat_prob" in r]
        k = sum(r["cat_prob"] >= 0.5 for r in scored)
        lo, hi = wilson(k, len(scored))
        lo2, hi2 = wilson(k, n_image)
        lines.append(f"| {cond} | {len(scored)} | {k}/{len(scored)} = {100 * k / max(len(scored), 1):.1f}% "
                     f"[{100 * lo:.1f}, {100 * hi:.1f}] | {k}/{n_image} = {100 * k / n_image:.1f}% "
                     f"[{100 * lo2:.1f}, {100 * hi2:.1f}] |")
        sheet(cond, rows, items, gal / f"{cond}.html")
        print(lines[-1], flush=True)
    (gal / "scores.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
