import { expect, test, type Page } from '@playwright/test';

type AuthStatus = {
  authEnabled?: boolean;
};

export async function getAuthStatus(page: Page): Promise<AuthStatus> {
  const response = await page.request.get('http://127.0.0.1:8000/api/v1/auth/status');
  expect(response.ok()).toBeTruthy();
  return (await response.json()) as AuthStatus;
}

export async function enterWorkspace(page: Page, missingPasswordMessage: string) {
  const authStatus = await getAuthStatus(page);

  if (!authStatus.authEnabled) {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    return;
  }

  const smokePassword = process.env.DSA_WEB_SMOKE_PASSWORD;
  test.skip(!smokePassword, missingPasswordMessage);

  await page.goto('/login');
  await page.waitForLoadState('domcontentloaded');
  await expect(page.locator('#password')).toBeVisible({ timeout: 10_000 });
  await page.locator('#password').fill(smokePassword!);

  const submitButton = page.getByRole('button', { name: /授权进入工作台|完成设置并登录/ });
  await expect(submitButton).toBeVisible();

  await Promise.all([
    page.waitForResponse(
      (response) => response.url().includes('/api/v1/auth/login') && response.status() === 200,
      { timeout: 15_000 }
    ),
    submitButton.click(),
  ]);

  await page.waitForURL('/', { timeout: 15_000 });
  await page.waitForLoadState('domcontentloaded');
}
