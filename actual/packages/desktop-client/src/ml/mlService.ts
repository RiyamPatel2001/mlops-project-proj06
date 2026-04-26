import { send } from '@actual-app/core/platform/client/connection';

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

type MlRequestResult<T> = {
  status: number;
  data: T | null;
  detail: string | null;
  error: 'auth' | 'network' | null;
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
): Promise<MlRequestResult<T>> {
  return (await send('ml-post', { path, body })) as MlRequestResult<T>;
}

async function get<T>(path: string): Promise<T | null> {
  const result = (await send('ml-get', { path })) as MlRequestResult<T>;
  if (result.error === 'auth' || result.error === 'network') {
    return null;
  }
  return result.data;
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
