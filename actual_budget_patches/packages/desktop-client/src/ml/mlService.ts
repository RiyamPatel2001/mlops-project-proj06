/**
 * ML Serving Service client — calls the FastAPI prediction service.
 *
 * The serving URL is resolved from the environment variable
 * ACTUAL_ML_SERVING_URL or falls back to the default.
 */

const browserDefaultServingUrl =
  typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : 'http://localhost:8000';

const ML_SERVING_URL =
  (typeof window !== 'undefined' &&
    (window as Record<string, unknown>).__ML_SERVING_URL__) ||
  browserDefaultServingUrl;

// ── Types ───────────────────────────────────────────────────────────────────

export interface ClassifyRequest {
  account: string;
  date: string;
  amount: number;
  payee_name: string;
  imported_id: string;
}

export interface ClassifyResponse {
  category: string;
  confidence: number;
  source: string; // "layer1" | "layer2"
  model_version: string;
}

interface APIClassifyRequest {
  transaction_id: string;
  user_id: string;
  payee: string;
  amount: number;
  date: string;
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
  user_id: string;
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
  user_id: string;
  payee: string;
  custom_category: string;
}

export interface SuggestionResponsePayload {
  user_id: string;
  transaction_id: string;
  action: 'accept' | 'dismiss';
  suggested_category: string;
}

// ── API Calls ───────────────────────────────────────────────────────────────

async function post<T>(path: string, body: unknown): Promise<T | null> {
  try {
    const resp = await fetch(`${ML_SERVING_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) return null;
    return (await resp.json()) as T;
  } catch {
    return null;
  }
}

async function get<T>(path: string): Promise<T | null> {
  try {
    const resp = await fetch(`${ML_SERVING_URL}${path}`);
    if (!resp.ok) return null;
    return (await resp.json()) as T;
  } catch {
    return null;
  }
}

export async function classifyTransaction(
  req: ClassifyRequest,
): Promise<ClassifyResponse | null> {
  const payload: APIClassifyRequest = {
    transaction_id: req.imported_id,
    user_id: req.account,
    payee: req.payee_name,
    amount: req.amount,
    date: req.date,
  };

  const result = await post<APIClassifyResponse>('/classify', payload);
  if (!result) {
    return null;
  }

  return {
    category: result.prediction_category,
    confidence: result.confidence ?? 1,
    source: result.source,
    model_version: result.model_version ?? '',
  };
}

export async function submitFeedback(
  payload: FeedbackPayload,
): Promise<boolean> {
  const result = await post('/feedback', payload);
  return result !== null;
}

export async function tagExample(
  payload: TagExamplePayload,
): Promise<boolean> {
  const result = await post('/tag-example', payload);
  return result !== null;
}

export async function getCustomCategories(
  userId: string,
): Promise<string[]> {
  const result = await get<{ categories: string[] }>(
    `/custom-categories?user_id=${encodeURIComponent(userId)}`,
  );
  return result?.categories ?? [];
}

export async function submitSuggestionResponse(
  payload: SuggestionResponsePayload,
): Promise<boolean> {
  const result = await post('/suggestion-response', payload);
  return result !== null;
}
