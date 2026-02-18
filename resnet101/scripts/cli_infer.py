import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="CLI ResNet101 para inferencia (path o URL).")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def _validate_inputs(image_path: str | None, url: str | None) -> None:
    if bool(image_path) == bool(url):
        raise ValueError("Debes enviar exactamente uno: --image-path o --url")


def main():
    args = parse_args()
    _validate_inputs(args.image_path, args.url)

    import torch

    from src.api.deps import get_device, get_id_to_label, get_model, get_model_version, load_resources
    from src.inference.pipeline import prepare_from_path, prepare_from_url

    load_resources()
    model = get_model()
    id_to_label = get_id_to_label()

    data = prepare_from_path(args.image_path) if args.image_path else prepare_from_url(args.url)
    x = data["tensor"]

    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    scores = {id_to_label.get(i, str(i)): float(p) for i, p in enumerate(probs)}
    label = max(scores.items(), key=lambda kv: kv[1])[0]

    payload = {
        "label": label,
        "scores": scores,
        "meta": {
            "model_version": get_model_version(),
            "device": get_device(),
        },
    }
    print(json.dumps(payload, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
