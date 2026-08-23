"""Synthetic patient profile and response-band generation."""


def _balanced_response_band(index):
    bands = ["complete", "partial", "stable", "progressive", "maintenance"]
    return bands[(index - 1) % len(bands)]


def _sample_profile(index, rng, balanced_subgroups=True, realism_profile="balanced"):
    if balanced_subgroups:
        # Keep common patterns common, but force enough HR+/HER2+ and TCHP/endocrine cases
        # so subgroup metrics are visible in the locked synthetic holdout.
        if realism_profile == "external_calibrated":
            # Approximate the small public BreastDCEDL/I-SPY1 feature file:
            # HR+/HER2- 42%, HER2+ 32%, triple-negative 26%.
            # We intentionally keep one HR+/HER2+ bucket for workflow coverage even
            # though the external file groups HER2-positive cases together.
            subtype_cycle = [
                "HR+/HER2-",
                "HER2+",
                "triple-negative",
                "HR+/HER2-",
                "HER2+",
                "triple-negative",
                "HR+/HER2-",
                "HER2+",
                "HR+/HER2-",
                "HR+/HER2+",
            ]
        else:
            subtype_cycle = [
                "HR+/HER2-",
                "HER2+",
                "triple-negative",
                "HR+/HER2+",
                "HR+/HER2-",
                "HER2+",
                "triple-negative",
                "HR+/HER2+",
                "HR+/HER2-",
                "HR+/HER2-",
            ]
        subtype = subtype_cycle[(index - 1) % len(subtype_cycle)]
    else:
        if realism_profile == "external_calibrated":
            subtype = rng.choices(
                ["HR+/HER2-", "HER2+", "triple-negative", "HR+/HER2+"],
                weights=[0.42, 0.27, 0.26, 0.05],
                k=1,
            )[0]
        else:
            subtype = rng.choices(
                ["HR+/HER2-", "HER2+", "triple-negative", "HR+/HER2+"],
                weights=[0.45, 0.22, 0.23, 0.10],
                k=1,
            )[0]
    stages = ["IIA", "IIB", "IIIA", "IIIB", "IV"]
    stage = rng.choices(stages, weights=[0.25, 0.30, 0.25, 0.15, 0.05], k=1)[0]
    receptor_map = {
        "HR+/HER2-": ("Positive", "Positive", "Not amplified"),
        "HER2+": ("Negative", "Negative", "Amplified"),
        "triple-negative": ("Negative", "Negative", "Not amplified"),
        "HR+/HER2+": ("Positive", "Positive", "Amplified"),
    }
    regimen_map = {
        "HR+/HER2-": (
            "dose-dense AC then paclitaxel",
            ["doxorubicin", "cyclophosphamide", "paclitaxel"],
        ),
        "HER2+": (
            "TCHP",
            ["docetaxel", "carboplatin", "trastuzumab", "pertuzumab"],
        ),
        "triple-negative": (
            "paclitaxel + carboplatin then AC",
            ["paclitaxel", "carboplatin", "doxorubicin", "cyclophosphamide"],
        ),
        "HR+/HER2+": (
            "TCHP then endocrine therapy",
            ["docetaxel", "carboplatin", "trastuzumab", "pertuzumab"],
        ),
    }
    regimen, drugs = regimen_map[subtype]
    er, pr, her2 = receptor_map[subtype]
    if realism_profile == "external_calibrated":
        age = int(round(max(28, min(69, rng.gauss(48.3, 8.9)))))
        baseline_size_cm = round(max(1.9, min(16.6, rng.gauss(6.78, 2.9))), 1)
    else:
        age = rng.randint(31, 74)
        baseline_size_cm = round(rng.uniform(2.1, 7.2), 1)

    return {
        "age": age,
        "stage": stage,
        "er_status": er,
        "pr_status": pr,
        "her2_status": her2,
        "subtype": subtype,
        "grade": rng.choice([2, 3, 3]),
        "menopausal_status": rng.choice(
            ["premenopausal", "postmenopausal", "perimenopausal"]
        ),
        "baseline_tumor_size_cm": baseline_size_cm,
        "nodal_status": rng.choice(["N0", "N1", "N1", "N2", "N3"]),
        "treatment_intent": (
            "palliative disease control"
            if stage == "IV"
            else "neoadjuvant curative-intent therapy"
        ),
        "regimen": regimen,
        "drugs": drugs,
        "breast_side": rng.choice(["left", "right"]),
        "location": rng.choice(
            [
                "upper outer quadrant",
                "upper inner quadrant",
                "central breast",
                "lower outer quadrant",
            ]
        ),
    }


def _response_strength(profile, rng, forced_response_band=None):
    if forced_response_band == "complete":
        return rng.uniform(0.82, 0.94)
    if forced_response_band == "partial":
        return rng.uniform(0.56, 0.74)
    if forced_response_band == "stable":
        return rng.uniform(0.28, 0.46)
    if forced_response_band == "progressive":
        return rng.uniform(0.08, 0.22)
    if forced_response_band == "maintenance":
        return rng.uniform(0.42, 0.62)

    base = {
        "HER2+": 0.78,
        "triple-negative": 0.68,
        "HR+/HER2+": 0.72,
        "HR+/HER2-": 0.52,
    }[profile["subtype"]]
    if profile["stage"] in {"IIIB", "IV"}:
        base -= 0.12
    return max(0.1, min(0.95, base + rng.uniform(-0.22, 0.18)))
