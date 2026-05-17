import { expect, test, type Page } from "@playwright/test";

async function signIn(page: Page, username: string, password: string, expectedRoute: RegExp) {
  await page.goto("/login");
  await page.getByLabel("Username").fill(username);
  await page.getByLabel("Password").fill(password);
  const response = page.waitForResponse((res) =>
    res.url().includes("/auth/demo-credential-login") && res.request().method() === "POST",
  );
  await page.getByRole("button", { name: /sign in to workspace/i }).click();
  await response;
  await page.waitForURL(expectedRoute, { timeout: 30_000 });
}

test.describe("role-aware smoke flows", () => {
  test.describe.configure({ mode: "serial" });

  test("patient login routes to patient dashboard and support chat", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await expect(page).toHaveURL(/\/patient/);
    await expect(page.getByText(/Patient P001/i)).toBeVisible();

    await page.getByRole("link", { name: /support/i }).click();
    await expect(page).toHaveURL(/\/patient\/chat/);
    await expect(page.getByPlaceholder(/Tell me how|Message/i)).toBeVisible({ timeout: 30_000 });
    await page.getByPlaceholder(/Tell me how|Message/i).fill("hi");
    await page.keyboard.press("Enter");
    await expect(page.getByText(/Checking safety gate|Routing intent|Generating response/i)).toBeVisible();
  });

  test("patient support chat saves a symptom and refreshes patient state", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await page.getByRole("link", { name: /support/i }).click();
    const input = page.getByPlaceholder(/Tell me how|Message/i);
    await input.fill("I have nausea severity 6/10 today");
    await page.keyboard.press("Enter");

    await expect(page.getByText(/Symptom saved|logged|saved/i).first()).toBeVisible({ timeout: 45_000 });
    await page.getByRole("button", { name: /Overview/i }).click();
    await expect(page.getByText(/nausea/i).first()).toBeVisible({ timeout: 30_000 });
  });

  test("patient timeline event opens a detail modal", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await page.goto("/patient#timeline");
    const firstDetailButton = page.getByRole("button", { name: /open details for/i }).first();
    await expect(firstDetailButton).toBeVisible({ timeout: 30_000 });
    await firstDetailButton.click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await expect(page.getByText(/Findings|Fields|Summary|Media|Details/i).first()).toBeVisible();
  });

  test("clinician login routes to review queue", async ({ page }) => {
    await signIn(page, "clinician", "clinician-demo", /\/clinician/);
    await expect(page).toHaveURL(/\/clinician/);
    await expect(page.getByText(/Clinician Dashboard/i)).toBeVisible();
    await expect(page.getByText(/Patients needing review/i)).toBeVisible();
  });

  test("admin login routes to MLE dashboard", async ({ page }) => {
    await signIn(page, "admin", "admin-demo", /\/admin/);
    await expect(page).toHaveURL(/\/admin/);
    await expect(page.getByRole("heading", { name: /Admin \/ MLE Dashboard/i })).toBeVisible();
    await expect(page.getByText(/RAG|MLE|Guardrails/i).first()).toBeVisible();
  });

  test("admin system health section loads", async ({ page }) => {
    await signIn(page, "admin", "admin-demo", /\/admin/);
    await page.goto("/admin/health");
    await expect(page.getByRole("heading", { name: /Admin \/ MLE Dashboard/i })).toBeVisible();
    await expect(page.getByText(/System Health/i).first()).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(/Database|Dependencies|Artifacts/i).first()).toBeVisible({ timeout: 30_000 });
  });

  test("route guard sends patient away from admin surface", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await page.goto("/admin");
    await expect(page).not.toHaveURL(/\/admin/);
  });

  test("patient dashboard renders the new 3-row card hierarchy", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await expect(page).toHaveURL(/\/patient/);

    // Row 1 — at-a-glance trio (Key signals · Review with care team · Recent symptoms)
    await expect(page.getByRole("heading", { name: /^Key signals$/i })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByRole("heading", { name: /Review with care team/i })).toBeVisible();

    // Row 2 — CBC labs + Timeline
    await expect(page.getByRole("heading", { name: /Lab values \(CBC\)/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Timeline|Treatment timeline/i }).first()).toBeVisible();

    // Row 3 — Model signal · Recent AI explanations · Next clinician review
    await expect(page.getByRole("heading", { name: /Recent AI explanations/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Next clinician review/i })).toBeVisible();

    // Page-level safety footnote must be reachable by scrolling — confirms it
    // rendered into the DOM at the bottom of the grid.
    const footnote = page.getByText(/Proof-of-concept/i).first();
    await footnote.scrollIntoViewIfNeeded();
    await expect(footnote).toBeVisible();
  });

  test("lab cards expose reference range and a status chip", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await expect(page.getByRole("heading", { name: /Lab values \(CBC\)/i })).toBeVisible({ timeout: 30_000 });
    // WBC card label comes from clinical-constants. At least one card status
    // chip (In range / Low / High / Very low / Very high / No value) must show.
    await expect(page.getByText(/^WBC$/).first()).toBeVisible();
    await expect(
      page.getByText(/In range|Low|High|Very low|Very high|Borderline|No value/).first(),
    ).toBeVisible();
    // The footer disclaimer copy must travel with the panel.
    await expect(page.getByText(/Reference ranges shown are population defaults/i)).toBeVisible();
  });

  test("tool tray opens the CBC drawer and dismisses on Escape", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await page.getByRole("link", { name: /support/i }).click();
    await expect(page).toHaveURL(/\/patient\/chat/);

    // Tool tray is a labelled toolbar with 8 chips.
    const tray = page.getByRole("toolbar", { name: /Add a health update/i });
    await expect(tray).toBeVisible({ timeout: 30_000 });

    // Open the CBC drawer via the "Save CBC" chip and confirm a dialog mounts.
    await tray.getByRole("button", { name: /Save CBC/i }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    await expect(dialog.getByText(/WBC|Hemoglobin|Platelets/i).first()).toBeVisible();

    // ESC must dismiss — confirms the Modal/Drawer primitive's a11y contract.
    await page.keyboard.press("Escape");
    await expect(dialog).toBeHidden();
  });

  test("tool tray ‘Ask a question’ chip routes to the chat composer", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    // From the overview tab, the education chip lives only on the chat page
    // tool tray — switch first so we exercise the routing in one direction.
    await page.getByRole("link", { name: /support/i }).click();
    const tray = page.getByRole("toolbar", { name: /Add a health update/i });
    await tray.getByRole("button", { name: /Ask a question/i }).click();
    // Already on chat; the toast and the composer should remain reachable.
    await expect(page.getByPlaceholder(/Tell me how|Message/i)).toBeVisible();
  });
});
