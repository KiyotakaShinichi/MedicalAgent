import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { SelectWithCustom } from "../../src/components/ui/Form";
import {
  SELECT_WITH_CUSTOM_OTHER_VALUE,
  resolveSelectWithCustomValue,
} from "../../src/components/ui/selectWithCustom";
import {
  COMMON_SYMPTOMS,
  COMMON_MEDICATIONS,
} from "../../src/lib/clinical-constants";
import { useState } from "react";

describe("resolveSelectWithCustomValue", () => {
  it("returns the canonical value when not in the Other branch", () => {
    expect(resolveSelectWithCustomValue("Fatigue", "anything")).toBe("Fatigue");
  });

  it("returns the trimmed custom value when in the Other branch", () => {
    expect(
      resolveSelectWithCustomValue(SELECT_WITH_CUSTOM_OTHER_VALUE, "  my custom  "),
    ).toBe("my custom");
  });

  it("returns empty string when nothing is chosen and custom is empty", () => {
    expect(resolveSelectWithCustomValue("", "")).toBe("");
  });
});

describe("SelectWithCustom component", () => {
  function Harness({ initialValue = "" }: { initialValue?: string }) {
    const [value, setValue] = useState(initialValue);
    const [custom, setCustom] = useState("");
    return (
      <SelectWithCustom
        id="t"
        value={value}
        customValue={custom}
        options={COMMON_SYMPTOMS}
        onChange={(v) => {
          setValue(v);
          if (v !== SELECT_WITH_CUSTOM_OTHER_VALUE) setCustom("");
        }}
        onCustomChange={setCustom}
      />
    );
  }

  it("renders every catalog option plus the Other branch", () => {
    render(<Harness />);
    const select = screen.getByRole("combobox") as HTMLSelectElement;
    // Catalog options + Other + placeholder (disabled).
    expect(select.options.length).toBe(COMMON_SYMPTOMS.length + 2);
    expect(select.options[select.options.length - 1].value).toBe(SELECT_WITH_CUSTOM_OTHER_VALUE);
  });

  it("does not render the custom input until Other is picked", () => {
    render(<Harness />);
    expect(screen.queryByPlaceholderText(/type your own/i)).toBeNull();

    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: SELECT_WITH_CUSTOM_OTHER_VALUE },
    });
    expect(screen.getByPlaceholderText(/type your own/i)).toBeInTheDocument();
  });

  it("clears the custom input when switching back from Other to a catalog value", () => {
    render(<Harness initialValue={SELECT_WITH_CUSTOM_OTHER_VALUE} />);
    const input = screen.getByPlaceholderText(/type your own/i) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "custom symptom" } });
    expect(input.value).toBe("custom symptom");

    fireEvent.change(screen.getByRole("combobox"), { target: { value: "Fatigue" } });
    expect(screen.queryByPlaceholderText(/type your own/i)).toBeNull();
  });
});

describe("Clinical catalog invariants", () => {
  it("COMMON_SYMPTOMS has stable, deduplicated canonical values", () => {
    const values = COMMON_SYMPTOMS.map((s) => s.value);
    expect(new Set(values).size).toBe(values.length);
    expect(values.length).toBeGreaterThanOrEqual(20);
    // Each label is non-trivial.
    for (const s of COMMON_SYMPTOMS) {
      expect(s.value.length).toBeGreaterThan(2);
      expect(s.label.length).toBeGreaterThan(s.value.length - 1);
    }
  });

  it("COMMON_MEDICATIONS values are unique and grouped sensibly", () => {
    const values = COMMON_MEDICATIONS.map((m) => m.value);
    expect(new Set(values).size).toBe(values.length);
    const groups = new Set(COMMON_MEDICATIONS.map((m) => m.group));
    // We expect at least the four buckets we curated.
    expect(groups.size).toBeGreaterThanOrEqual(4);
    expect(groups.has("Chemotherapy")).toBe(true);
    expect(groups.has("Targeted therapy")).toBe(true);
    expect(groups.has("Endocrine therapy")).toBe(true);
  });

  it("Neither catalog uses the Other sentinel value as a real option", () => {
    expect(COMMON_SYMPTOMS.some((s) => s.value === SELECT_WITH_CUSTOM_OTHER_VALUE)).toBe(false);
    expect(COMMON_MEDICATIONS.some((m) => m.value === SELECT_WITH_CUSTOM_OTHER_VALUE)).toBe(false);
  });
});
