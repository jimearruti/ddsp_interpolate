from pathlib import Path

CSV_FIELDS = [
    "resynth_path", "model", "category", "method", "grouping",
    "pair", "instrument", "alpha", "distance",
]

def parse_resynth_path(path):
    """
    Pull structured fields out of a results_reordered path.
    """
    fields = {k: "" for k in CSV_FIELDS}
    fields["resynth_path"] = path

    parts = Path(path).parts
    if "results_reordered" not in parts:
        return fields

    rel = parts[parts.index("results_reordered") + 1:]
    if len(rel) < 2:
        return fields

    model, category, rest = rel[0], rel[1], rel[2:]
    fields["model"] = model
    fields["category"] = category

    if category == "resynthesis":
        if rest:
            fields["instrument"] = rest[0]
        return fields

    # category is "output" or "weights"
    fields["method"] = category
    if not rest:
        return fields

    head = rest[0]
    if head == "all":
        fields["grouping"] = "all"
    elif head == "extremes":
        fields["grouping"] = "extremes"
        if len(rest) > 1:
            fields["instrument"] = rest[1]
    elif head == "unordered_pairs":
        fields["grouping"] = "unordered_pairs"
        if len(rest) > 1:
            fields["pair"] = rest[1]
        if len(rest) > 2:
            fields["alpha"] = rest[2]
    else:
        fields["pair"] = head
        if len(rest) > 1:
            fields["alpha"] = rest[1]

    return fields
