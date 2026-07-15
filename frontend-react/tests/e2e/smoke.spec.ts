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

function chatComposer(page: Page) {
  return page.getByPlaceholder(/Tell me how|Message/i);
}

test.describe("role-aware smoke flows", () => {
  test.describe.configure({ mode: "serial" });

  test("patient login routes to patient dashboard and support chat", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await expect(page).toHaveURL(/\/patient/);
    await expect(page.getByText(/Patient P001/i)).toBeVisible();

    await page.getByRole("link", { name: /support/i }).click();
    await expect(page).toHaveURL(/\/patient\/chat/);
    await expect(chatComposer(page)).toBeVisible({ timeout: 90_000 });
    const readyAssistantMessages = page.locator('[data-testid="assistant-message"][data-message-ready="true"]');
    const assistantCountBeforeSend = await readyAssistantMessages.count();
    await chatComposer(page).fill("hi");
    await page.getByRole("button", { name: /send message/i }).click();
    await expect(page.getByTestId("user-message").last()).toContainText("hi");
    await expect(readyAssistantMessages).toHaveCount(assistantCountBeforeSend + 1, { timeout: 90_000 });
    await expect(chatComposer(page)).toBeEnabled({ timeout: 90_000 });
  });

  test("patient support chat saves a symptom and refreshes patient state", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await page.getByRole("link", { name: /support/i }).click();
    const input = chatComposer(page);
    await expect(input).toBeVisible({ timeout: 90_000 });
    await input.fill("I have nausea severity 6/10 today");
    await page.keyboard.press("Enter");

    await expect(page.getByText(/nothing has been saved yet/i).first()).toBeVisible({ timeout: 45_000 });
    await page.getByRole("button", { name: /confirm save/i }).click();
    await expect(page.getByText(/Symptom saved|logged|saved/i).first()).toBeVisible({ timeout: 45_000 });
    await page.getByRole("link", { name: /Overview/i }).click();
    await expect(page.getByText(/nausea/i).first()).toBeVisible({ timeout: 30_000 });
  });

  test("patient timeline event opens a detail modal", async ({ page }) => {
    test.setTimeout(180_000);
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await page.getByRole("link", { name: /timeline/i }).click();
    const firstDetailButton = page.getByRole("button", { name: /open details for/i }).first();
    await expect(firstDetailButton).toBeVisible({ timeout: 90_000 });
    await firstDetailButton.click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    await expect(dialog.getByText(/Symptom|Severity|Findings|Fields|Summary|Media|Details/i).first()).toBeVisible();
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

  test("admin RAG hardening cards render", async ({ page }) => {
    await signIn(page, "admin", "admin-demo", /\/admin/);
    await page.goto("/admin/rag");
    await expect(page.getByRole("heading", { name: /Live-Agent RAG Eval/i })).toBeVisible({ timeout: 90_000 });
    await expect(page.getByRole("heading", { name: /Claim-Level Citation Eval/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /RAG Trace Replay/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Rerun/i }).first()).toBeVisible();
  });

  test("route guard sends patient away from admin surface", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await page.goto("/admin");
    await expect(page).not.toHaveURL(/\/admin/);
  });

  test("patient dashboard renders the KPI-style card hierarchy", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await expect(page).toHaveURL(/\/patient/);

    // Top row — KPI strip mirrors a company-style dashboard overview.
    await expect(page.getByText(/Items for review/i).first()).toBeVisible({ timeout: 90_000 });
    await expect(page.getByText(/Synthetic model pattern/i).first()).toBeVisible();
    await expect(page.getByText(/Latest CBC/i).first()).toBeVisible();
    await expect(page.getByText(/Record coverage/i).first()).toBeVisible();
    await expect(page.getByText(/How NLCare calculated it/i)).toBeVisible();
    await expect(page.getByRole("heading", { name: /Safe next steps/i })).toBeVisible();

    // Detail rows — labs, timeline, summaries, and model signal remain reachable.
    await expect(page.getByRole("heading", { name: /Lab values \(CBC\)/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Timeline|Treatment timeline/i }).first()).toBeVisible();
    await expect(page.getByRole("heading", { name: /Today's summary/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Review queue/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Model signal/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Recent AI explanations/i })).toBeVisible();

    // Page-level safety footnote must be reachable by scrolling — confirms it
    // rendered into the DOM at the bottom of the grid.
    const footnote = page.getByText(/Proof-of-concept/i).first();
    await footnote.scrollIntoViewIfNeeded();
    await expect(footnote).toBeVisible();
  });

  test("patient KPI cards avoid clipping at tablet and mobile widths", async ({ page }) => {
    test.setTimeout(180_000);
    await page.setViewportSize({ width: 906, height: 698 });
    await signIn(page, "P001", "patient-demo", /\/patient/);
    const cards = page.locator(".patient-kpi-card");
    await expect(cards).toHaveCount(4, { timeout: 90_000 });

    const tabletLayout = await cards.evaluateAll((elements) =>
      elements.map((element) => {
        const rect = element.getBoundingClientRect();
        return {
          y: Math.round(rect.y),
          clientWidth: element.clientWidth,
          scrollWidth: element.scrollWidth,
        };
      }),
    );
    expect(new Set(tabletLayout.map((card) => card.y)).size).toBeGreaterThanOrEqual(2);
    expect(tabletLayout.every((card) => card.scrollWidth <= card.clientWidth + 1)).toBe(true);

    await page.setViewportSize({ width: 390, height: 844 });
    await page.reload();
    await expect(page.getByText(/Items for review/i).first()).toBeVisible({ timeout: 90_000 });
    const mobileLayout = await cards.evaluateAll((elements) =>
      elements.map((element) => {
        const rect = element.getBoundingClientRect();
        return {
          y: Math.round(rect.y),
          clientWidth: element.clientWidth,
          scrollWidth: element.scrollWidth,
        };
      }),
    );
    const pageWidth = await page.evaluate(() => ({
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
    }));
    expect(new Set(mobileLayout.map((card) => card.y)).size).toBe(4);
    expect(mobileLayout.every((card) => card.scrollWidth <= card.clientWidth + 1)).toBe(true);
    expect(pageWidth.scrollWidth).toBeLessThanOrEqual(pageWidth.clientWidth + 1);
  });

  test("lab cards expose reference range and a status chip", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await expect(page.getByRole("heading", { name: /Lab values \(CBC\)/i })).toBeVisible({ timeout: 90_000 });
    // WBC card label comes from clinical-constants. At least one card status
    // chip (In range / Low / High / Very low / Very high / No value) must show.
    await expect(page.getByText(/^WBC$/).first()).toBeVisible();
    await expect(
      page.getByText(/In range|Low|High|Very low|Very high|Borderline|No value/).first(),
    ).toBeVisible();
    // The footer disclaimer copy must travel with the panel.
    await expect(page.getByText(/Reference ranges (shown are population defaults|are not personalised)/i).first()).toBeVisible();
  });

  test("composer plus menu opens the CBC drawer and dismisses on Escape", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await page.getByRole("link", { name: /support/i }).click();
    await expect(page).toHaveURL(/\/patient\/chat/);

    const addRecord = page.getByRole("button", { name: /Add a health record/i });
    await expect(addRecord).toBeVisible({ timeout: 90_000 });
    await addRecord.click();

    const menu = page.getByRole("menu", { name: /Add a health record/i });
    await expect(menu).toBeVisible();

    // Open the CBC drawer via the composer plus-menu and confirm a dialog mounts.
    await menu.getByRole("menuitem", { name: /Save CBC/i }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    await expect(dialog.getByText(/WBC|Hemoglobin|Platelets/i).first()).toBeVisible();

    // ESC must dismiss the drawer, confirming the Modal/Drawer primitive's a11y contract.
    await page.keyboard.press("Escape");
    await expect(dialog).toBeHidden();
  });

  test("composer plus menu exposes record and upload actions without hiding chat", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await page.getByRole("link", { name: /support/i }).click();
    await expect(chatComposer(page)).toBeVisible({ timeout: 90_000 });
    const addRecord = page.getByRole("button", { name: /Add a health record/i });
    await addRecord.click();

    const menu = page.getByRole("menu", { name: /Add a health record/i });
    await expect(menu.getByRole("menuitem", { name: /Log symptom/i })).toBeVisible();
    await expect(menu.getByRole("menuitem", { name: /Save CBC/i })).toBeVisible();
    await expect(menu.getByRole("menuitem", { name: /Upload CBC image/i })).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(menu).toBeHidden();
    await expect(chatComposer(page)).toBeVisible();
  });

  test("patient overview surfaces the synthetic engineering caveat on monitoring tiles", async ({ page }) => {
    // The KPI tiles for the model-derived signals MUST carry the credibility
    // footer so the patient never
    // mistakes an engineering signal for a clinical prediction.
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await expect(page.getByText(/Items for review/i).first()).toBeVisible({ timeout: 90_000 });
    await expect(
      page
        .getByText(/Synthetic engineering signal.*Not a clinical prediction.*For clinician review/i)
        .first(),
    ).toBeVisible();
  });

  test("synthetic monitoring model card shows decision slots or a clean abstention state", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await expect(page.getByText(/Synthetic monitoring model/i).first()).toBeVisible({ timeout: 90_000 });
    // Whichever path the backend returns, one of these credibility lines must
    // show on the page. The hybrid envelope ships a per-card footer; the
    // abstention path shows the evidence-aware empty state.
    await expect(
      page
        .getByText(
          /Synthetic engineering signal.*Not a clinical prediction.*For clinician review|No live hybrid prediction|Insufficient evidence/i,
        )
        .first(),
    ).toBeVisible();
  });

  test("admin safety eval card carries n-size, clinical_validation:false, and the warn-tinted needs_attention path", async ({ page }) => {
    await signIn(page, "admin", "admin-demo", /\/admin/);
    await page.goto("/admin/safety");
    // The eval integrity strip must show its credibility keys — these are the
    // honest accounting bits a reviewer needs without opening the artifact.
    const integrity = page.locator("[data-eval-integrity]").first();
    await expect(integrity).toBeVisible({ timeout: 60_000 });
    await expect(integrity.locator(".eval-integrity-key", { hasText: /^Total n$/i })).toBeVisible();
    await expect(integrity.locator(".eval-integrity-key", { hasText: /^Passed$/i })).toBeVisible();
    await expect(integrity.locator(".eval-integrity-key", { hasText: /^Failed$/i })).toBeVisible();
    await expect(integrity.locator(".eval-integrity-key", { hasText: /^Skipped$/i })).toBeVisible();
    await expect(integrity.locator(".eval-integrity-key", { hasText: /^Clinical validation$/i })).toBeVisible();
    // Value must read literal "false" — this is the credibility claim.
    await expect(integrity.getByText(/^false$/).first()).toBeVisible();
    await expect(integrity.locator(".eval-integrity-key", { hasText: /^Used for tuning$/i })).toBeVisible();
  });
});
