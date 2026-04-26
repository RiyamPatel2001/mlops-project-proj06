import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  classifyTransaction,
  classifyTransactions,
  getStoredMLUsername,
  registerMLUser,
  signInMLUser,
  tagExample,
} from './mlService';

describe('mlService', () => {
  beforeEach(() => {
    const store = new Map<string, string>();
    const storage = {
      getItem: vi.fn((key: string) => store.get(key) ?? null),
      setItem: vi.fn((key: string, value: string) => {
        store.set(key, value);
      }),
      removeItem: vi.fn((key: string) => {
        store.delete(key);
      }),
      clear: vi.fn(() => {
        store.clear();
      }),
    };

    Object.defineProperty(window, 'localStorage', {
      value: storage,
      configurable: true,
    });
    vi.stubGlobal('fetch', vi.fn());
    window.localStorage.setItem('ml-serving-auth-token', 'test-token');
    window.localStorage.setItem('ml-serving-username', 'test-user');
  });

  afterEach(() => {
    window.localStorage.clear();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('includes bulk routing metadata on single classify requests', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        transaction_id: 'txn-1',
        user_id: 'service-user-1',
        prediction_category: 'Groceries',
        confidence: 0.92,
        source: 'layer1',
        model_version: 'fasttext-v1',
      }),
    } as Response);
    vi.stubGlobal('fetch', fetchMock);

    await classifyTransaction({
      account: 'acct-1',
      imported_id: 'txn-1',
      payee_name: 'WHOLE FOODS',
      amount: -42.15,
      date: '2026-04-20',
      requestMode: 'bulk',
      batchId: 'import-123',
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, options] = fetchMock.mock.calls[0];
    expect(JSON.parse(String(options?.body))).toMatchObject({
      request_mode: 'bulk',
      batch_id: 'import-123',
    });
    expect(JSON.parse(String(options?.body))).not.toHaveProperty('user_id');
    expect(options?.headers).toMatchObject({
      Authorization: 'Bearer test-token',
    });
  });

  it('caps in-flight classify requests for bulk imports', async () => {
    let active = 0;
    let maxActive = 0;

    const fetchMock = vi.fn().mockImplementation(async (_url, options) => {
      active += 1;
      maxActive = Math.max(maxActive, active);

      const body = JSON.parse(String(options?.body));
      await new Promise(resolve => setTimeout(resolve, 10));

      active -= 1;
      return {
        ok: true,
        json: async () => ({
          transaction_id: body.transaction_id,
          user_id: 'service-user-1',
          prediction_category: 'Groceries',
          confidence: 0.88,
          source: 'layer1',
          model_version: 'fasttext-v1',
        }),
      } as Response;
    });
    vi.stubGlobal('fetch', fetchMock);

    const results = await classifyTransactions(
      Array.from({ length: 5 }, (_, index) => ({
        account: 'acct-1',
        imported_id: `txn-${index}`,
        payee_name: `PAYEE ${index}`,
        amount: -10 - index,
        date: '2026-04-20',
      })),
      {
        requestMode: 'bulk',
        batchId: 'import-456',
        concurrency: 2,
      },
    );

    expect(results).toHaveLength(5);
    expect(maxActive).toBeLessThanOrEqual(2);
    for (const [, options] of fetchMock.mock.calls) {
      expect(JSON.parse(String(options?.body))).toMatchObject({
        request_mode: 'bulk',
        batch_id: 'import-456',
      });
      expect(JSON.parse(String(options?.body))).not.toHaveProperty('user_id');
      expect(options?.headers).toMatchObject({
        Authorization: 'Bearer test-token',
      });
    }
  });

  it('returns a structured success result for tagged examples', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: 'ok',
        id: 7,
      }),
    } as Response);
    vi.stubGlobal('fetch', fetchMock);

    await expect(
      tagExample({
        payee: 'WHOLE FOODS',
        custom_category: 'Personal Groceries',
      }),
    ).resolves.toEqual({ ok: true });
  });

  it('stores the ML session after a successful sign-in', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: 'ok',
        user_id: 'service-user-1',
        username: 'jayraj',
        token: 'fresh-token',
      }),
    } as Response);
    vi.stubGlobal('fetch', fetchMock);

    await expect(signInMLUser('jayraj', 'secret')).resolves.toEqual({
      ok: true,
      message: '',
    });
    expect(window.localStorage.setItem).toHaveBeenCalledWith(
      'ml-serving-auth-token',
      'fresh-token',
    );
    expect(getStoredMLUsername()).toBe('jayraj');
  });

  it('surfaces username conflicts during ML registration', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      json: async () => ({}),
    } as Response);
    vi.stubGlobal('fetch', fetchMock);

    await expect(registerMLUser('jayraj', 'secret')).resolves.toEqual({
      ok: false,
      message: 'That username is already taken. Choose a different one.',
    });
  });

  it('reports network failures when tagged examples cannot reach the service', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error('network down'));
    vi.stubGlobal('fetch', fetchMock);

    await expect(
      tagExample({
        payee: 'WHOLE FOODS',
        custom_category: 'Personal Groceries',
      }),
    ).resolves.toEqual({ ok: false, reason: 'network' });
  });

  it('reports server failures when tagged examples are rejected', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({}),
    } as Response);
    vi.stubGlobal('fetch', fetchMock);

    await expect(
      tagExample({
        payee: 'WHOLE FOODS',
        custom_category: 'Personal Groceries',
      }),
    ).resolves.toEqual({ ok: false, reason: 'server', status: 500 });
  });
});
