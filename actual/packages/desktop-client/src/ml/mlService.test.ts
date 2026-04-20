import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  classifyTransaction,
  classifyTransactions,
} from './mlService';

describe('mlService', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('includes bulk routing metadata on single classify requests', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        transaction_id: 'txn-1',
        user_id: 'acct-1',
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
          user_id: body.user_id,
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
    }
  });
});
