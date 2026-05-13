import { expect, test, type Page } from "@playwright/test";

async function signIn(page: Page, username: string, password: string) {
  await page.goto("/login");
  await page.getByLabel("Username").fill(username);
  await page.getByLabel("Password").fill(password);
  await page.getByRole("button", { name: /sign in to workspace/i }).click();
}

test.describe("role-aware smoke flows", () => {
  test("patient login routes to patient dashboard and support chat", async ({ page }) => {
    await signIn(page, "P001", "patient-demo");
    await expect(page).toHaveURL(/\/patient/);
    await expect(page.getByText(/Patient P001/i)).toBeVisible();

    await page.getByRole("link", { name: /support/i }).click();
    await expect(page).toHaveURL(/\/patient\/chat/);
    await expect(page.getByPlaceholder(/Message/i)).toBeVisible({ timeout: 30_000 });
    await page.getByPlaceholder(/Message/i).fill("hi");
    await page.keyboard.press("Enter");
    await expect(page.getByText(/Checking safety gate|Routing intent|Generating response/i)).toBeVisible();
  });

  test("clinician login routes to review queue", async ({ page }) => {
    await signIn(page, "clinician", "clinician-demo");
    await expect(page).toHaveURL(/\/clinician/);
    await expect(page.getByText(/Clinician Dashboard/i)).toBeVisible();
    await expect(page.getByText(/Patients needing review/i)).toBeVisible();
  });

  test("admin login routes to MLE dashboard", async ({ page }) => {
    await signIn(page, "admin", "admin-demo");
    await expect(page).toHaveURL(/\/admin/);
    await expect(page.getByRole("heading", { name: /Admin \/ MLE Dashboard/i })).toBeVisible();
    await expect(page.getByText(/RAG|MLE|Guardrails/i).first()).toBeVisible();
  });

  test("route guard sends patient away from admin surface", async ({ page }) => {
    await signIn(page, "P001", "patient-demo");
    await page.goto("/admin");
    await expect(page).not.toHaveURL(/\/admin/);
  });
});
