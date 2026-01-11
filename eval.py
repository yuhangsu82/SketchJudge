import json
import math
import difflib
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter

# =========================
# Utilities (text & namespacing)
# =========================

def normalize_text(s: str) -> str:
    """Trim, collapse spaces, and lowercase."""
    return " ".join(s.strip().split()).lower()

def normalize_label_str(s: str) -> str:
    """
    Lowercase, strip, collapse spaces; remove common punctuation so that
    minor variations (dashes, underscores, commas) don't break matching.
    """
    s = " ".join(s.strip().split()).lower()
    for ch in [",", ".", ":", ";", "-", "_", "/", "\\", "(", ")", "[", "]"]:
        s = s.replace(ch, " ")
    s = " ".Join(s.split()) if hasattr(str, "Join") else " ".join(s.split())  # robust collapse
    return s

def ns_label(category: str, raw_label: str) -> str:
    """(category, label) → namespaced unique label, e.g., 'physics::omission error'."""
    return f"{normalize_text(category)}::{normalize_label_str(raw_label)}"  # note: label part normalized for stability

def applicable(lbl: str, cat: str) -> bool:
    """Whether a namespaced label is applicable to a sample: prefix must match the sample's category."""
    return lbl.startswith(normalize_text(cat) + "::")

# =========================
# Load taxonomy.json → canonical taxonomy
# =========================

def load_taxonomy(path: str) -> Dict[str, List[str]]:
    """
    Load taxonomy.json and return a dict: {category (normalized) -> list of canonical label strings (original casing)}.
    Accepts two formats per category:
      - list[str]
      - list[{"label": str, "description": str}]
    We store canonical labels in their original text for readability in papers,
    but use normalized strings for internal matching.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    taxonomy: Dict[str, List[str]] = {}
    for cat, val in raw.items():
        if cat == "version":
            continue
        cat_norm = normalize_text(cat)
        labels: List[str] = []
        if isinstance(val, list):
            if all(isinstance(x, str) for x in val):
                labels = list(val)
            elif all(isinstance(x, dict) and "label" in x for x in val):
                labels = [x["label"] for x in val if isinstance(x.get("label"), str)]
            else:
                # Mixed or unexpected; try to extract any strings we can.
                for x in val:
                    if isinstance(x, str):
                        labels.append(x)
                    elif isinstance(x, dict) and isinstance(x.get("label"), str):
                        labels.append(x["label"])
        taxonomy[cat_norm] = labels
    return taxonomy

def build_global_label_space_from_taxonomy(taxonomy: Dict[str, List[str]]) -> List[str]:
    """
    Create a canonical namespaced label space from taxonomy only (do not include predictions).
    The namespaced label uses normalized category and normalized label tokenization for stability.
    """
    out: List[str] = []
    for cat, labels in taxonomy.items():
        for lab in labels:
            out.append(ns_label(cat, lab))
    return sorted(set(out)) 

def map_pred_label_to_canonical(
    category: str,
    raw_label: str,
    taxonomy: Dict[str, List[str]],
    fuzzy_threshold: float = 0.84
) -> Optional[str]:
    """
    Map a predicted 'error_type' string to a canonical namespaced label in the given category.
    Steps:
      1) exact normalized match against canonical labels
      2) fuzzy match via difflib (ratio >= threshold)
    Return namespaced canonical label on success; otherwise None (OOV).
    """
    if not isinstance(raw_label, str) or not raw_label.strip():
        return None
    norm = normalize_label_str(raw_label)
    cat = normalize_text(category)
    canonical_list = taxonomy.get(cat, [])

    # 1) exact normalized match
    for can in canonical_list:
        if normalize_label_str(can) == norm:
            return ns_label(cat, can)

    # 2) fuzzy match against canonical candidates
    best_can, best_score = None, 0.0
    for can in canonical_list:
        score = difflib.SequenceMatcher(a=norm, b=normalize_label_str(can)).ratio()
        if score > best_score:
            best_score, best_can = score, can
    if best_can is not None and best_score >= fuzzy_threshold:
        return ns_label(cat, best_can)

    return None

# =========================
# Safe JSON loading for model responses
# =========================

def safe_load_json(s: str) -> Optional[Dict[str, Any]]:
    """Safely load a JSON object; return None on failure or if not a dict."""
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None

def safe_load_response(resp_text: str) -> Optional[Dict[str, Any]]:
    """
    Minimal schema check for model outputs:
      - requires boolean 'is_correct';
      - if present, 'error_count' must be int and 'error_list' must be list.
    """
    obj = safe_load_json(resp_text)
    if obj is None or "is_correct" not in obj or not isinstance(obj["is_correct"], bool):
        return None
    if "error_count" in obj and not isinstance(obj["error_count"], int):
        return None
    if "error_list" in obj and not isinstance(obj["error_list"], list):
        return None
    return obj

# =========================
# Extract namespaced GT / predictions
# =========================

def extract_gt_error_labels_ns(label_rec: Dict[str, Any], category: str) -> List[str]:
    """
    Support two GT formats:
      A) error_list: [{"error_type": "...", ...}, ...]
      B) error_types: ["...", "..."]
    Return namespaced labels (normalized), e.g., ["physics::omission error", ...].
    """
    out: List[str] = []
    if isinstance(label_rec.get("error_list"), list):
        for item in label_rec.get("error_list") or []:
            if isinstance(item, dict):
                et = item.get("error_type")
                if isinstance(et, str) and et.strip():
                    out.append(ns_label(category, et))
        return out

    if isinstance(label_rec.get("error_types"), list):
        for et in label_rec.get("error_types") or []:
            if isinstance(et, str) and et.strip():
                out.append(ns_label(category, et))
        return out

    return []

def extract_pred_error_labels_ns(
    resp_obj: Dict[str, Any],
    category: str,
    taxonomy: Dict[str, List[str]],
    fuzzy_threshold: float = 0.84
) -> Tuple[List[str], bool, int]:
    """
    Extract predicted error labels (namespaced) with canonical mapping.
    Returns (labels, looks_ok, n_oov), where n_oov counts unmapped labels.
    """
    error_list = resp_obj.get("error_list", []) or []
    labels: List[str] = []
    n_oov = 0
    for item in error_list:
        if isinstance(item, dict):
            et = item.get("error_type")
            mapped = map_pred_label_to_canonical(
                category, et, taxonomy, fuzzy_threshold
            )
            if mapped is not None:
                labels.append(mapped)
            else:
                n_oov += 1
    error_count = resp_obj.get("error_count", len(error_list))
    looks_ok = (error_count == len(error_list))
    return labels, looks_ok, n_oov

# =========================
# Binary metrics
# =========================

def confusion_counts(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN) with positive = gold 'Correct' (1)."""
    TP = FP = TN = FN = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            TP += 1
        elif t == 0 and p == 1:
            FP += 1
        elif t == 0 and p == 0:
            TN += 1
        else:
            FN += 1
    return TP, FP, TN, FN

def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Return (precision, recall, F1) for a given (tp, fp, fn)."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    """Matthews correlation coefficient (optional diagnostic)."""
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return ((tp * tn) - (fp * fn)) / denom

def macro_f1_binary(tp: int, fp: int, tn: int, fn: int) -> float:
    """Macro-F1 (binary): average of F1 on 'Correct' (positive) and 'Incorrect' (negative) classes."""
    _, _, f1_pos = precision_recall_f1(tp, fp, fn)
    _, _, f1_neg = precision_recall_f1(tn, fn, fp)
    return 0.5 * (f1_pos + f1_neg)

def report_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Report binary metrics aligned with the paper:
      - Accuracy
      - MacroF1_binary
      - FNR = FN / (FN + TP)  (over-strictness on gold-correct)
      - FPR = FP / (FP + TN)  (over-leniency on gold-incorrect)
    Plus class-wise Precision/Recall/F1 for 'Correct' (optional) and MCC (diagnostic).
    """
    n = len(y_true)
    TP, FP, TN, FN = confusion_counts(y_true, y_pred)
    acc = (TP + TN) / n if n > 0 else 0.0

    prec_pos, rec_pos, f1_pos = precision_recall_f1(TP, FP, FN)
    f1_macro_binary = macro_f1_binary(TP, FP, TN, FN)
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    mcc_val = mcc(TP, FP, TN, FN)

    return {
        "Accuracy": acc,
        "Precision_pos": prec_pos,
        "Recall_pos": rec_pos,
        "F1_pos": f1_pos,
        "MacroF1_binary": f1_macro_binary,
        "FNR": fnr,
        "FPR": fpr,
        "MCC": mcc_val,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN, "N": n
    }

# =========================
# Multilabel (error types) with masking
# =========================

def hamming_loss_masked(
    y_true_lists: List[List[str]],
    y_pred_lists: List[List[str]],
    label_space: List[str],
    sample_cats: List[str],
) -> float:
    """Masked Hamming loss: compute only on applicable (sample, label) pairs."""
    mismatches, denom = 0, 0
    for gt, pd, cat in zip(y_true_lists, y_pred_lists, sample_cats):
        g, p = set(gt), set(pd)
        for lbl in label_space:
            if not applicable(lbl, cat):
                continue
            denom += 1
            mismatches += int((lbl in g) != (lbl in p))
    return 0.0 if denom == 0 else mismatches / denom

def multilabel_counts_per_class_masked(
    y_true_lists: List[List[str]],
    y_pred_lists: List[List[str]],
    label_space: List[str],
    sample_cats: List[str],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Accumulate TP/FP/FN per label only on applicable positions (TN not needed for F1)."""
    TPc = {lbl: 0 for lbl in label_space}
    FPc = {lbl: 0 for lbl in label_space}
    FNc = {lbl: 0 for lbl in label_space}
    for gt_labels, pd_labels, cat in zip(y_true_lists, y_pred_lists, sample_cats):

        gt_set, pd_set = set(gt_labels), set(pd_labels)
        for lbl in label_space:
            if not applicable(lbl, cat):
                continue
            in_gt, in_pd = (lbl in gt_set), (lbl in pd_set)
            if in_gt and in_pd:
                TPc[lbl] += 1
            elif (not in_gt) and in_pd:
                FPc[lbl] += 1
            elif in_gt and (not in_pd):
                FNc[lbl] += 1
    return TPc, FPc, FNc

def example_based_f1_masked(
    y_true_lists: List[List[str]],
    y_pred_lists: List[List[str]],
    label_space: List[str],
    sample_cats: List[str],
) -> float:
    """
    Example-based (sample-based) F1 with applicability masking.
    """
    f1s = []
    for gt, pd, cat in zip(y_true_lists, y_pred_lists, sample_cats):
        applicable_labels = {lbl for lbl in label_space if applicable(lbl, cat)}
        G = set(gt) & applicable_labels
        P = set(pd) & applicable_labels

        if not G and not P:
            f1s.append(1.0)
            continue

        inter = len(G & P)
        denom = len(G) + len(P)
        f1s.append((2 * inter / denom) if denom > 0 else 0.0)

    return sum(f1s) / len(f1s) if f1s else 0.0

def macro_micro_f1_from_counts(TPc: Dict[str, int], FPc: Dict[str, int], FNc: Dict[str, int]) -> Tuple[float, float]:
    """
    Return (macro-F1, micro-F1):
      - Macro-F1: average over labels that have at least one positive (TP+FN > 0).
      - Micro-F1: computed from sums of TP/FP/FN over all labels.
    """
    f1s: List[float] = []
    for lbl in TPc:
        tp, fp, fn = TPc[lbl], FPc[lbl], FNc[lbl]
        if (tp + fn) == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro = sum(f1s) / len(f1s) if f1s else 0.0

    tp_sum = sum(TPc.values()); fp_sum = sum(FPc.values()); fn_sum = sum(FNc.values())
    prec_mi = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    rec_mi  = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro   = (2 * prec_mi * rec_mi) / (prec_mi + rec_mi) if (prec_mi + rec_mi) > 0 else 0.0
    return macro, micro

def per_class_recall_masked(
    y_true_bin: List[int],
    y_pred_bin: List[int],
    gt_err_lists_all: List[List[str]],
    pd_err_lists_all: List[List[str]],
    sample_cats: List[str],
    label_space: List[str],
    strict: bool = True,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int]]:
    """
    Per-class recall (“fraction of gold instances recovered”) with applicability masking.
    We count multiplicity (multiset): for each sample i and class c,
      correct_i,c = min(gold_count_i,c, pred_count_i,c)

    strict=True: evaluate only on samples where (gold incorrect AND predicted incorrect).
    strict=False: evaluate on all gold incorrect samples.

    Return:
      - recall_per_class: {label -> recall}
      - num_correct:      {label -> sum_i correct_i,c}
      - den_gold:         {label -> sum_i gold_count_i,c}
    """
    assert len(y_true_bin) == len(y_pred_bin) == len(gt_err_lists_all) == len(pd_err_lists_all) == len(sample_cats)

    num_correct = {lbl: 0 for lbl in label_space}
    den_gold    = {lbl: 0 for lbl in label_space}

    for gt_bin, pd_bin, gt_ls, pd_ls, cat in zip(
        y_true_bin, y_pred_bin, gt_err_lists_all, pd_err_lists_all, sample_cats
    ):
        if gt_bin != 0:  # only gold incorrect has error instances
            continue
        if strict and pd_bin != 0:
            continue

        gt_ctr = Counter(gt_ls)
        pd_ctr = Counter(pd_ls)

        for lbl, gcount in gt_ctr.items():
            if not applicable(lbl, cat):
                continue
            if lbl not in den_gold:
                continue  # should not happen in global label space, but harmless
            den_gold[lbl] += gcount
            num_correct[lbl] += min(gcount, pd_ctr.get(lbl, 0))

    recall_per_class: Dict[str, float] = {}
    for lbl in label_space:
        den = den_gold[lbl]
        if den > 0:
            recall_per_class[lbl] = num_correct[lbl] / den

    return recall_per_class, num_correct, den_gold

def evaluate_error_types(
    y_true_bin: List[int],
    y_pred_bin: List[int],
    gt_err_lists_all: List[List[str]],
    pd_err_lists_all: List[List[str]],
    sample_cats: List[str],
    strict: bool = True,
    use_global_label_space: bool = True,
    global_label_space: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Error-type metrics with applicability masking:
      - strict=True: evaluate only on (gold incorrect AND predicted incorrect), i.e., (y=0 and ŷ=0).
      - use_global_label_space=True: compute on a fixed global namespaced label space (recommended).
    """
    assert len(y_true_bin) == len(y_pred_bin) == len(gt_err_lists_all) == len(pd_err_lists_all) == len(sample_cats)

    # Select indices for error-type evaluation
    idxs: List[int] = []
    for i in range(len(y_true_bin)):
        if y_true_bin[i] == 0:  # gold incorrect
            if strict:
                if y_pred_bin[i] == 0:
                    idxs.append(i)
            else:
                idxs.append(i)

    if not idxs:
        return {"F1e_example": 0.0, "F1e_macro": 0.0, "F1e_micro": 0.0, "Hamming_masked": 0.0, "N_eval": 0, "NumLabels": 0}

    gt_sel   = [gt_err_lists_all[i] for i in idxs]
    pd_sel   = [pd_err_lists_all[i] for i in idxs]
    cats_sel = [sample_cats[i] for i in idxs]

    # Label space (global or subset). In global mode we require a prebuilt canonical space.
    if use_global_label_space:
        assert global_label_space is not None, "global_label_space must be provided when use_global_label_space=True"
        lbl_space = global_label_space
    else:
        lbl_space = collect_label_space(gt_sel, pd_sel)

    TPc, FPc, FNc = multilabel_counts_per_class_masked(gt_sel, pd_sel, lbl_space, cats_sel)
    f1_example = example_based_f1_masked(gt_sel, pd_sel, lbl_space, cats_sel)
    f1_macro, f1_micro = macro_micro_f1_from_counts(TPc, FPc, FNc)
    hl_masked = hamming_loss_masked(gt_sel, pd_sel, lbl_space, cats_sel)

    return {
        "F1e_example": f1_example,
        "F1e_macro": f1_macro,
        "F1e_micro": f1_micro,
        "Hamming_masked": hl_masked,
        "N_eval": len(idxs),
        "NumLabels": len(lbl_space)
    }

def collect_label_space(lists_a: List[List[str]], lists_b: List[List[str]]) -> List[str]:
    """Collect the union of labels appearing in two lists of label lists (assumes already canonicalized)."""
    s: Set[str] = set()
    for ls in lists_a:
        for x in ls:
            s.add(x)
    for ls in lists_b:
        for x in ls:
            s.add(x)
    return sorted(s)

# =========================
# End-to-end evaluation
# =========================

def get_category(rec: Dict[str, Any]) -> str:
    """Extract the category/subject/domain from a record (default 'ALL')."""
    for k in ("category", "subject", "domain"):
        if isinstance(rec.get(k), str) and rec[k].strip():
            return rec[k]
    return "ALL"

def report_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    # (duplicated above intentionally for readability if you keep this function here)
    n = len(y_true)
    TP, FP, TN, FN = confusion_counts(y_true, y_pred)
    acc = (TP + TN) / n if n > 0 else 0.0
    prec_pos, rec_pos, f1_pos = precision_recall_f1(TP, FP, FN)
    f1_macro_binary = macro_f1_binary(TP, FP, TN, FN)
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    mcc_val = mcc(TP, FP, TN, FN)
    return {
        "Accuracy": acc, "Precision_pos": prec_pos, "Recall_pos": rec_pos, "F1_pos": f1_pos,
        "MacroF1_binary": f1_macro_binary, "FNR": fnr, "FPR": fpr, "MCC": mcc_val,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN, "N": n
    }

def evaluate_all(
    results: List[Dict[str, Any]],
    label_index: Dict[str, Dict[str, Any]],
    taxonomy: Dict[str, List[str]],
    strict_for_errors: bool = True,
    use_global_label_space: bool = True
) -> None:
    """
    Entry point: compute binary metrics and error-type metrics (Strict/Lenient; Global/Local label space).
    Label space (global) is built from taxonomy only; predictions never extend the label space.
    """
    sample_cats_all: List[str] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    gt_err_all: List[List[str]] = []
    pd_err_all: List[List[str]] = []

    # bookkeeping
    total = used = 0
    miss_gt = parse_fail = miss_field = inconsistent = 0
    n_oov_total = 0  # predicted labels that cannot be mapped to canonical

    # per-category containers
    cat_true_bin = defaultdict(list)
    cat_pred_bin = defaultdict(list)
    cat_gt_err   = defaultdict(list)
    cat_pd_err   = defaultdict(list)

    for row in results:
        total += 1
        ans_id = row.get("answer_id")
        if ans_id is None or ans_id not in label_index:
            miss_gt += 1
            print(f"[Skip] Missing GT for answer_id={ans_id}")
            continue

        gt_rec = label_index[ans_id]
        gt_bool = gt_rec.get("is_correct", None)
        if not isinstance(gt_bool, bool):
            miss_field += 1
            print(f"[Skip] Missing or invalid 'is_correct' for answer_id={ans_id}")
            print(" GT snippet:", {k: gt_rec.get(k) for k in ['answer_id', 'category', 'is_correct']})
            continue

        category = get_category(gt_rec)

        # GT namespaced labels (already canonical format)
        gt_labels_ns = extract_gt_error_labels_ns(gt_rec, category)

        # parse prediction
        resp = safe_load_response(row.get("response", ""))
        if resp is None:
            parse_fail += 1
            print(f"[Skip] Parse failed for answer_id={ans_id}. Raw response:", row.get("response", ""))
            continue
        pred_bool = resp.get("is_correct", None)
        if not isinstance(pred_bool, bool):
            miss_field += 1
            print(f"[Skip] Missing or invalid 'is_correct' in prediction for answer_id={ans_id}")
            continue

        # Map predicted labels to canonical namespaced labels
        pred_labels_ns, looks_ok, n_oov = extract_pred_error_labels_ns(
            resp, category, taxonomy, fuzzy_threshold=0.84
        )
        n_oov_total += n_oov
        if not looks_ok:
            inconsistent += 1

        gt_bin = 1 if gt_bool else 0
        pd_bin = 1 if pred_bool else 0

        y_true_all.append(gt_bin)
        y_pred_all.append(pd_bin)
        gt_err_all.append(gt_labels_ns)
        pd_err_all.append(pred_labels_ns)
        sample_cats_all.append(category)
        used += 1

        cat_true_bin[category].append(gt_bin)
        cat_pred_bin[category].append(pd_bin)
        cat_gt_err[category].append(gt_labels_ns)
        cat_pd_err[category].append(pred_labels_ns)

    # === Binary (overall) ===
    print("=== Binary metrics (overall) ===")
    m_bin = report_binary_metrics(y_true_all, y_pred_all)
    for k in ["Accuracy", "Precision_pos", "Recall_pos", "F1_pos",
              "MacroF1_binary", "MCC", "FNR", "FPR"]:
        print(f"{k:>14}: {m_bin[k]:.4f}")
    print(f" Confusion: TP={m_bin['TP']} FP={m_bin['FP']} TN={m_bin['TN']} FN={m_bin['FN']}")
    print(f"      Used: {m_bin['N']} / total lines {total}")
    print(f"   Skipped: missing_gt={miss_gt}, parse_fail={parse_fail}, missing_field={miss_field}")
    print(f" Inconsistent error_count vs list: {inconsistent}")
    print(f" OOV predicted labels (unmappable) = {n_oov_total}")

    # === Build canonical global label space from taxonomy (if selected) ===
    global_label_space = build_global_label_space_from_taxonomy(taxonomy) if use_global_label_space else None

    # === Error types (overall) ===
    mode = "Strict" if strict_for_errors else "Lenient"
    lbl_mode = "GlobalSpace" if use_global_label_space else "LocalSpace"
    print(f"\n=== Error-type metrics overall [{mode} | {lbl_mode}] ===")
    m_err = evaluate_error_types(
        y_true_all, y_pred_all, gt_err_all, pd_err_all, sample_cats_all,
        strict=strict_for_errors,
        use_global_label_space=use_global_label_space,
        global_label_space=global_label_space
    )
    print(f"Example-F1={m_err['F1e_example']:.4f} F1e_macro={m_err['F1e_macro']:.4f}  F1e_micro={m_err['F1e_micro']:.4f}  "
          f"Hamming_masked={m_err['Hamming_masked']:.4f}  N_eval={m_err['N_eval']}  |Labels|={m_err['NumLabels']}")

    # === Per-class recall (overall) ===
    if use_global_label_space:
        print(f"\n=== Per-class recall (overall) [{mode} | {lbl_mode}] ===")
        rec_pc, num_c, den_g = per_class_recall_masked(
            y_true_all, y_pred_all,
            gt_err_all, pd_err_all,
            sample_cats_all,
            label_space=global_label_space,
            strict=strict_for_errors
        )
        # return rec_pc, num_c, den_g
        items = sorted(rec_pc.items(), key=lambda kv: (-den_g[kv[0]], kv[1]))
        print(f"{'Label':60s}  {'Recall':>7s}  {'Correct':>7s}  {'Gold':>5s}")
        for lbl, r in items:
            print(f"{lbl:60s}  {r:7.3f}  {num_c[lbl]:7d}  {den_g[lbl]:5d}")

    # === Per-category breakdown ===
    cats = list(cat_true_bin.keys())
    if len(cats) > 1 or ("all" not in [normalize_text(c) for c in cats]):
        print("\n=== Per-category breakdown (Binary) ===")
        for cat in cats:
            ms = report_binary_metrics(cat_true_bin[cat], cat_pred_bin[cat])
            print(f"[{cat}] Acc={ms['Accuracy']:.4f}  MacroF1_binary={ms['MacroF1_binary']:.4f} "
                  f"MCC={ms['MCC']:.4f}  FNR={ms['FNR']:.4f}  FPR={ms['FPR']:.4f}  N={ms['N']}")

        print(f"\n=== Per-category breakdown (Error-types) [{mode} | {lbl_mode}] ===")
        for cat in cats:
            ms_e = evaluate_error_types(
                cat_true_bin[cat], cat_pred_bin[cat],
                cat_gt_err[cat],   cat_pd_err[cat],
                sample_cats=[cat] * len(cat_true_bin[cat]),
                strict=strict_for_errors,
                use_global_label_space=use_global_label_space,
                global_label_space=global_label_space
            )
            print(f"[{cat}] Example-F1={ms_e['F1e_example']:.4f} F1e_macro={ms_e['F1e_macro']:.4f}  F1e_micro={ms_e['F1e_micro']:.4f}  "
                  f"Hamming_masked={ms_e['Hamming_masked']:.4f}  N_eval={ms_e['N_eval']}  |Labels|={ms_e['NumLabels']}")


def main():
    dataset_path = "./data/SketchJudge_v1/master.json"
    results_path = "./output/result.jsonl"
    taxonomy_path  = "./data/SketchJudge_v1/taxonomy.json"

    # Load dataset (GT)
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    annotations = dataset.get("annotations", [])
    label_index = {rec["answer_id"]: rec for rec in annotations if "answer_id" in rec}

    # Load model results
    with open(results_path, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    # Load taxonomy and run evaluation
    taxonomy = load_taxonomy(taxonomy_path)

    evaluate_all(
        results=results,
        label_index=label_index,
        taxonomy=taxonomy,
        strict_for_errors=True,
        use_global_label_space=True
    )


if __name__ == "__main__":
    main()
