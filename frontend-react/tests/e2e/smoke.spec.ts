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

  test("route guard sends patient away from admin surface", async ({ page }) => {
    await signIn(page, "P001", "patient-demo", /\/patient/);
    await page.goto("/admin");
    await expect(page).not.toHaveURL(/\/admin/);
  });
});
