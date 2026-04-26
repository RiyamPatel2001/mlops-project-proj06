import * as connection from '@actual-app/core/platform/client/connection';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  classifyTransaction,
  classifyTransactions,
  tagExample,
} from './mlService';

describe('mlService', () => {
  beforeEach(() => {
    vi.spyOn(connection, 'send').mockResolvedValue({
      status: 200,
      data: null,
      detail: null,
      error: null,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('includes bulk routing metadata on single classify requests', async () => {
    vi.spyOn(connection, 'send').mockResolvedValue({
      status: 200,
      data: {
        transaction_id: 'txn-1',
        user_id: 'actual-user-1',
        prediction_category: 'Groceries',
        confidence: 0.92,
        source: 'layer1',
        model_version: 'fasttext-v1',
      },
      detail: null,
      error: null,
    });

    await classifyTransaction({
      account: 'acct-1',
      imported_id: 'txn-1',
      payee_name: 'WHOLE FOODS',
      amount: -42.15,
      date: '2026-04-20',
      requestMode: 'bulk',
      batchId: 'import-123',
    });

    expect(connection.send).toHaveBeenCalledTimes(1);
    const [, args] = vi.mocked(connection.send).mock.calls[0];
    expect(args).toMatchObject({
      path: '/classify',
    });
    expect((args as { body: Record<string, unknown> }).body).toMatchObject({
      request_mode: 'bulk',
      batch_id: 'import-123',
    });
    expect((args as { body: Record<string, unknown> }).body).not.toHaveProperty(
      'user_id',
    );
  });

  it('caps in-flight classify requests for bulk imports', async () => {
    let active = 0;
    let maxActive = 0;

    vi.spyOn(connection, 'send').mockImplementation(async (_name, args) => {
      active += 1;
      maxActive = Math.max(maxActive, active);

      const body = (args as { body: Record<string, string> }).body;
      await new Promise(resolve => setTimeout(resolve, 10));

      active -= 1;
      return {
        status: 200,
        data: {
          transaction_id: body.transaction_id,
          user_id: 'actual-user-1',
          prediction_category: 'Groceries',
          confidence: 0.88,
          source: 'layer1',
          model_version: 'fasttext-v1',
        },
        detail: null,
        error: null,
      };
    });

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
    for (const [, args] of vi.mocked(connection.send).mock.calls) {
      expect((args as { body: Record<string, unknown> }).body).toMatchObject({
        request_mode: 'bulk',
        batch_id: 'import-456',
      });
      expect(
        (args as { body: Record<string, unknown> }).body,
      ).not.toHaveProperty('user_id');
    }
  });

  it('returns a structured success result for tagged examples', async () => {
    vi.spyOn(connection, 'send').mockResolvedValue({
      status: 200,
      data: {
        status: 'ok',
        id: 7,
      },
      detail: null,
      error: null,
    });

    await expect(
      tagExample({
        payee: 'WHOLE FOODS',
        custom_category: 'Personal Groceries',
      }),
    ).resolves.toEqual({ ok: true });
  });

  it('reports network failures when tagged examples cannot reach the service', async () => {
    vi.spyOn(connection, 'send').mockResolvedValue({
      status: 0,
      data: null,
      detail: null,
      error: 'network',
    });

    await expect(
      tagExample({
        payee: 'WHOLE FOODS',
        custom_category: 'Personal Groceries',
      }),
    ).resolves.toEqual({ ok: false, reason: 'network' });
  });

  it('reports server failures when tagged examples are rejected', async () => {
    vi.spyOn(connection, 'send').mockResolvedValue({
      status: 500,
      data: null,
      detail: null,
      error: null,
    });

    await expect(
      tagExample({
        payee: 'WHOLE FOODS',
        custom_category: 'Personal Groceries',
      }),
    ).resolves.toEqual({ ok: false, reason: 'server', status: 500 });
  });
});
