/**
 * ML Serving Service client — calls the FastAPI prediction service.
 *
 * The serving URL can be injected via `window.__ML_SERVING_URL__`. Otherwise,
 * fall back to a sensible browser default. The deployed k8s app exposes
 * ActualBudget on 30506 and the classifier on 30508.
 */

declare global {
  interface Window {
    __ML_SERVING_URL__?: string;
  }
}

function resolveBrowserDefaultServingUrl() {
  if (typeof window === 'undefined') {
    return 'http://localhost:8000';
  }

  const { protocol, hostname, port } = window.location;

  if (port === '30506') {
    return `${protocol}//${hostname}:30508`;
  }

  return `${protocol}//${hostname}:8000`;
}

const ML_SERVING_URL =
  (typeof window !== 'undefined' && window.__ML_SERVING_URL__) ||
  resolveBrowserDefaultServingUrl();
const ML_AUTH_TOKEN_STORAGE_KEY = 'ml-serving-auth-token';
const ML_AUTH_USERNAME_STORAGE_KEY = 'ml-serving-username';

let authFlowPromise: Promise<string | null> | null = null;

function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ── Types ───────────────────────────────────────────────────────────────────

export interface ClassifyRequest {
  account: string;
  date: string;
  amount: number;
  payee_name: string;
  imported_id: string;
  requestMode?: 'interactive' | 'bulk';
  batchId?: string;
}

export interface ClassifyResponse {
  category: string;
  confidence: number;
  source: string; // "layer1" | "layer2"
  model_version: string;
}

export type ClassificationRequestMode = 'interactive' | 'bulk';

export const DEFAULT_BULK_CLASSIFY_CONCURRENCY = 6;

interface APIClassifyRequest {
  transaction_id: string;
  payee: string;
  amount: number;
  date: string;
  request_mode?: ClassificationRequestMode;
  batch_id?: string;
}

interface APIClassifyResponse {
  transaction_id: string;
  user_id: string;
  prediction_category: string;
  confidence: number | null;
  source: string;
  model_version: string | null;
}

export interface FeedbackPayload {
  transaction_id: string;
  payee: string;
  amount: number;
  date: string;
  original_prediction: string | null;
  original_confidence: number | null;
  source: string;
  final_label: string;
  reviewed_by_user: boolean;
  timestamp: string;
}

export interface TagExamplePayload {
  payee: string;
  custom_category: string;
}

export type TagExampleResult =
  | { ok: true }
  | {
      ok: false;
      reason: 'auth' | 'network' | 'server';
      status?: number;
    };

export interface SuggestionResponsePayload {
  transaction_id: string;
  action: 'accept' | 'dismiss';
  suggested_category: string;
}

type AuthRegisterResponse = {
  status: string;
  user_id: string;
  username: string;
};

type AuthLoginResponse = {
  status: string;
  user_id: string;
  username: string;
  token: string;
};

export function createClassificationBatchId(accountId: string): string {
  return (
    globalThis.crypto?.randomUUID?.() ||
    `ml-batch-${accountId}-${Date.now().toString(36)}`
  );
}

// ── API Calls ───────────────────────────────────────────────────────────────

async function post<T>(path: string, body: unknown): Promise<T | null> {
  const result = await postWithStatus<T>(path, body);
  return result.data;
}

async function postWithStatus<T>(
  path: string,
  body: unknown,
): Promise<{ status: number; data: T | null; error: 'auth' | 'network' | null }> {
  const token = await ensureAuthenticatedSession();
  if (!token) {
    return { status: 401, data: null, error: 'auth' };
  }

  try {
    let resp = await fetch(`${ML_SERVING_URL}${path}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    if (resp.status === 401) {
      clearMLAuthSession();
      const refreshedToken = await ensureAuthenticatedSession();
      if (!refreshedToken) {
        return { status: 401, data: null, error: 'auth' };
      }

      resp = await fetch(`${ML_SERVING_URL}${path}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${refreshedToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });
    }
    return {
      status: resp.status,
      data: resp.ok ? ((await resp.json()) as T) : null,
      error: null,
    };
  } catch {
    return { status: 0, data: null, error: 'network' };
  }
}

async function get<T>(path: string): Promise<T | null> {
  const token = await ensureAuthenticatedSession();
  if (!token) {
    return null;
  }

  try {
    let resp = await fetch(`${ML_SERVING_URL}${path}`, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });
    if (resp.status === 401) {
      clearMLAuthSession();
      const refreshedToken = await ensureAuthenticatedSession();
      if (!refreshedToken) {
        return null;
      }

      resp = await fetch(`${ML_SERVING_URL}${path}`, {
        headers: {
          'Authorization': `Bearer ${refreshedToken}`,
        },
      });
    }
    if (!resp.ok) return null;
    return (await resp.json()) as T;
  } catch {
    return null;
  }
}

function getStoredToken(): string | null {
  if (typeof window === 'undefined') {
    return null;
  }
  return window.localStorage.getItem(ML_AUTH_TOKEN_STORAGE_KEY);
}

export function getStoredMLUsername(): string {
  if (typeof window === 'undefined') {
    return '';
  }
  return window.localStorage.getItem(ML_AUTH_USERNAME_STORAGE_KEY) ?? '';
}

function storeSession(token: string, username: string) {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.setItem(ML_AUTH_TOKEN_STORAGE_KEY, token);
  window.localStorage.setItem(ML_AUTH_USERNAME_STORAGE_KEY, username);
}

export function clearMLAuthSession() {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.removeItem(ML_AUTH_TOKEN_STORAGE_KEY);
  window.localStorage.removeItem(ML_AUTH_USERNAME_STORAGE_KEY);
}

async function postWithoutSession<T>(
  path: string,
  body: unknown,
): Promise<{ status: number; data: T | null }> {
  try {
    const resp = await fetch(`${ML_SERVING_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return {
      status: resp.status,
      data: resp.ok ? ((await resp.json()) as T) : null,
    };
  } catch {
    return { status: 0, data: null };
  }
}

async function ensureAuthenticatedSession(): Promise<string | null> {
  const storedToken = getStoredToken();
  if (storedToken) {
    return storedToken;
  }

  if (authFlowPromise) {
    return authFlowPromise;
  }

  authFlowPromise = runAuthFlow().finally(() => {
    authFlowPromise = null;
  });
  return authFlowPromise;
}

type AuthDialogMode = 'register' | 'login';

type AuthDialogResult = {
  mode: AuthDialogMode;
  username: string;
  password: string;
} | null;

function createAuthTabButton(label: string): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = label;
  button.style.border = '1px solid #cfd6dd';
  button.style.borderRadius = '999px';
  button.style.padding = '8px 12px';
  button.style.background = '#ffffff';
  button.style.cursor = 'pointer';
  button.style.fontWeight = '600';
  return button;
}

function showAuthDialog(
  initialMode: AuthDialogMode,
  initialUsername: string,
  initialMessage = '',
): Promise<AuthDialogResult> {
  return new Promise(resolve => {
    if (typeof document === 'undefined' || !document.body) {
      resolve(null);
      return;
    }

    let currentMode = initialMode;

    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.inset = '0';
    overlay.style.display = 'flex';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    overlay.style.background = 'rgba(14, 22, 34, 0.55)';
    overlay.style.zIndex = '9999';

    const dialog = document.createElement('div');
    dialog.style.width = 'min(420px, calc(100vw - 32px))';
    dialog.style.background = '#ffffff';
    dialog.style.borderRadius = '16px';
    dialog.style.boxShadow = '0 24px 80px rgba(15, 23, 42, 0.32)';
    dialog.style.padding = '20px';
    dialog.style.display = 'flex';
    dialog.style.flexDirection = 'column';
    dialog.style.gap = '14px';
    dialog.style.fontFamily =
      'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif';

    const title = document.createElement('div');
    title.style.fontSize = '1.1rem';
    title.style.fontWeight = '700';
    title.style.color = '#102033';

    const subtitle = document.createElement('div');
    subtitle.style.fontSize = '0.92rem';
    subtitle.style.color = '#526173';
    subtitle.style.lineHeight = '1.45';

    const tabRow = document.createElement('div');
    tabRow.style.display = 'flex';
    tabRow.style.gap = '8px';

    const newUserButton = createAuthTabButton('New user');
    const returningUserButton = createAuthTabButton('Returning user');

    tabRow.append(newUserButton, returningUserButton);

    const message = document.createElement('div');
    message.style.minHeight = '20px';
    message.style.fontSize = '0.88rem';
    message.style.color = initialMessage
      ? (
          initialMessage.startsWith('Account created')
            ? '#2f6b3d'
            : '#a23838'
        )
      : '#526173';
    message.textContent = initialMessage;

    const form = document.createElement('form');
    form.style.display = 'flex';
    form.style.flexDirection = 'column';
    form.style.gap = '12px';

    const usernameLabel = document.createElement('label');
    usernameLabel.style.display = 'flex';
    usernameLabel.style.flexDirection = 'column';
    usernameLabel.style.gap = '6px';
    usernameLabel.style.fontSize = '0.9rem';
    usernameLabel.style.color = '#102033';
    usernameLabel.textContent = 'Username';

    const usernameInput = document.createElement('input');
    usernameInput.type = 'text';
    usernameInput.value = initialUsername;
    usernameInput.autocomplete = 'username';
    usernameInput.style.border = '1px solid #cfd6dd';
    usernameInput.style.borderRadius = '10px';
    usernameInput.style.padding = '10px 12px';
    usernameInput.style.fontSize = '0.95rem';
    usernameInput.style.outline = 'none';

    usernameLabel.appendChild(usernameInput);

    const passwordLabel = document.createElement('label');
    passwordLabel.style.display = 'flex';
    passwordLabel.style.flexDirection = 'column';
    passwordLabel.style.gap = '6px';
    passwordLabel.style.fontSize = '0.9rem';
    passwordLabel.style.color = '#102033';
    passwordLabel.textContent = 'Password';

    const passwordInput = document.createElement('input');
    passwordInput.type = 'password';
    passwordInput.autocomplete = currentMode === 'register' ? 'new-password' : 'current-password';
    passwordInput.style.border = '1px solid #cfd6dd';
    passwordInput.style.borderRadius = '10px';
    passwordInput.style.padding = '10px 12px';
    passwordInput.style.fontSize = '0.95rem';
    passwordInput.style.outline = 'none';

    passwordLabel.appendChild(passwordInput);

    const buttonRow = document.createElement('div');
    buttonRow.style.display = 'flex';
    buttonRow.style.justifyContent = 'flex-end';
    buttonRow.style.gap = '10px';

    const cancelButton = document.createElement('button');
    cancelButton.type = 'button';
    cancelButton.textContent = 'Cancel';
    cancelButton.style.border = '1px solid #cfd6dd';
    cancelButton.style.borderRadius = '10px';
    cancelButton.style.padding = '10px 14px';
    cancelButton.style.background = '#ffffff';
    cancelButton.style.cursor = 'pointer';

    const submitButton = document.createElement('button');
    submitButton.type = 'submit';
    submitButton.style.border = 'none';
    submitButton.style.borderRadius = '10px';
    submitButton.style.padding = '10px 14px';
    submitButton.style.background = '#102033';
    submitButton.style.color = '#ffffff';
    submitButton.style.cursor = 'pointer';
    submitButton.style.fontWeight = '600';

    buttonRow.append(cancelButton, submitButton);
    form.append(usernameLabel, passwordLabel, buttonRow);
    dialog.append(title, subtitle, tabRow, message, form);
    overlay.appendChild(dialog);

    const applyMode = (mode: AuthDialogMode) => {
      currentMode = mode;
      title.textContent =
        mode === 'register'
          ? 'Create ML service account'
          : 'Sign in to ML service';
      subtitle.textContent =
        mode === 'register'
          ? 'Choose a unique username and password. After the account is created, you will sign in with those credentials.'
          : 'Use your existing ML service username and password.';
      submitButton.textContent =
        mode === 'register' ? 'Create account' : 'Sign in';
      passwordInput.value = '';
      passwordInput.autocomplete =
        mode === 'register' ? 'new-password' : 'current-password';

      const activeBackground = '#102033';
      const activeText = '#ffffff';
      const inactiveBackground = '#ffffff';
      const inactiveText = '#102033';

      newUserButton.style.background =
        mode === 'register' ? activeBackground : inactiveBackground;
      newUserButton.style.color =
        mode === 'register' ? activeText : inactiveText;
      returningUserButton.style.background =
        mode === 'login' ? activeBackground : inactiveBackground;
      returningUserButton.style.color =
        mode === 'login' ? activeText : inactiveText;
    };

    const cleanup = (result: AuthDialogResult) => {
      document.removeEventListener('keydown', onKeyDown);
      overlay.remove();
      resolve(result);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        cleanup(null);
      }
    };

    newUserButton.addEventListener('click', () => {
      message.textContent = '';
      message.style.color = '#526173';
      applyMode('register');
    });
    returningUserButton.addEventListener('click', () => {
      message.textContent = '';
      message.style.color = '#526173';
      applyMode('login');
    });
    cancelButton.addEventListener('click', () => cleanup(null));
    overlay.addEventListener('click', event => {
      if (event.target === overlay) {
        cleanup(null);
      }
    });
    form.addEventListener('submit', event => {
      event.preventDefault();

      const username = usernameInput.value.trim();
      const password = passwordInput.value;
      if (!username || !password) {
        message.style.color = '#a23838';
        message.textContent = 'Username and password are required.';
        return;
      }

      cleanup({
        mode: currentMode,
        username,
        password,
      });
    });

    applyMode(initialMode);
    document.addEventListener('keydown', onKeyDown);
    document.body.appendChild(overlay);
    usernameInput.focus();
  });
}

export async function registerMLUser(
  username: string,
  password: string,
): Promise<{ ok: boolean; message: string }> {
  const result = await postWithoutSession<AuthRegisterResponse>('/auth/register', {
    username,
    password,
  });

  if (result.status === 201 && result.data) {
    return { ok: true, message: '' };
  }

  switch (result.status) {
    case 409:
      return {
        ok: false,
        message: 'That username is already taken. Choose a different one.',
      };
    case 503:
      return {
        ok: false,
        message: 'The ML service auth store is temporarily unavailable.',
      };
    case 0:
      return {
        ok: false,
        message: 'Unable to reach the ML service right now. Try again shortly.',
      };
    default:
      return {
        ok: false,
        message: 'Account creation failed. Try again in a moment.',
      };
  }
}

async function loginMlUser(
  username: string,
  password: string,
): Promise<AuthLoginResponse | null> {
  const result = await postWithoutSession<AuthLoginResponse>('/auth/login', {
    username,
    password,
  });
  return result.status === 200 ? result.data : null;
}

export async function signInMLUser(
  username: string,
  password: string,
): Promise<{ ok: boolean; message: string }> {
  const session = await loginMlUser(username, password);
  if (session?.token) {
    storeSession(session.token, username);
    return { ok: true, message: '' };
  }

  return {
    ok: false,
    message: 'Login failed. Check your username and password and try again.',
  };
}

async function runAuthFlow(): Promise<string | null> {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return null;
  }

  let suggestedUsername = getStoredMLUsername();
  let mode: AuthDialogMode = 'login';
  let message = '';

  while (true) {
    const authAttempt = await showAuthDialog(mode, suggestedUsername, message);
    if (!authAttempt) {
      return null;
    }

    suggestedUsername = authAttempt.username;

    if (authAttempt.mode === 'register') {
      const registration = await registerMLUser(
        authAttempt.username,
        authAttempt.password,
      );
      if (registration.ok) {
        mode = 'login';
        message = 'Account created. Sign in with your new username and password to continue.';
      } else {
        mode = 'register';
        message = registration.message;
      }
      continue;
    }

    const login = await signInMLUser(
      authAttempt.username,
      authAttempt.password,
    );
    if (login.ok) {
      return getStoredToken();
    }

    mode = 'login';
    message = login.message;
  }
}

export async function classifyTransaction(
  req: ClassifyRequest,
): Promise<ClassifyResponse | null> {
  const payload: APIClassifyRequest = {
    transaction_id: req.imported_id,
    payee: req.payee_name,
    amount: req.amount,
    date: req.date,
    request_mode: req.requestMode,
    batch_id: req.batchId,
  };

  for (let attempt = 0; attempt < 3; attempt++) {
    const result = await post<APIClassifyResponse>('/classify', payload);
    if (result) {
      return {
        category: result.prediction_category,
        confidence: result.confidence ?? 1,
        source: result.source,
        model_version: result.model_version ?? '',
      };
    }

    if (attempt < 2) {
      await delay(250 * (attempt + 1));
    }
  }

  return null;
}

export async function classifyTransactions(
  requests: ClassifyRequest[],
  options: {
    requestMode?: ClassificationRequestMode;
    batchId?: string;
    concurrency?: number;
  } = {},
): Promise<Array<ClassifyResponse | null>> {
  const concurrency = Math.max(1, options.concurrency ?? 1);
  const results = new Array<ClassifyResponse | null>(requests.length).fill(null);
  let nextIndex = 0;

  async function worker() {
    while (nextIndex < requests.length) {
      const currentIndex = nextIndex;
      nextIndex += 1;

      const req = requests[currentIndex];
      results[currentIndex] = await classifyTransaction({
        ...req,
        requestMode: req.requestMode ?? options.requestMode,
        batchId: req.batchId ?? options.batchId,
      });
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(concurrency, requests.length) }, () =>
      worker(),
    ),
  );

  return results;
}

export async function submitFeedback(
  payload: FeedbackPayload,
): Promise<boolean> {
  const result = await post('/feedback', payload);
  return result !== null;
}

export async function tagExample(
  payload: TagExamplePayload,
): Promise<TagExampleResult> {
  const result = await postWithStatus('/tag-example', payload);
  if (result.data !== null) {
    return { ok: true };
  }

  if (result.error === 'auth' || result.status === 401) {
    return { ok: false, reason: 'auth', status: result.status };
  }

  if (result.error === 'network' || result.status === 0) {
    return { ok: false, reason: 'network' };
  }

  return { ok: false, reason: 'server', status: result.status };
}

export async function getCustomCategories(): Promise<string[]> {
  const result = await get<{ categories: string[] }>('/custom-categories');
  return result?.categories ?? [];
}

export async function submitSuggestionResponse(
  payload: SuggestionResponsePayload,
): Promise<boolean> {
  const result = await post('/suggestion-response', payload);
  return result !== null;
}
