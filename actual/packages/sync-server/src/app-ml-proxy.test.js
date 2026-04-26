import crypto from 'node:crypto';

import request from 'supertest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { validateSession } from './util/validate-user';

vi.mock('./account-db', () => ({
  getUserInfo: vi.fn(userId => ({
    id: userId,
    user_name: 'alice',
    display_name: 'Alice',
  })),
}));

vi.mock('./util/middlewares', () => ({
  requestLoggerMiddleware: (_req, _res, next) => next(),
}));

vi.mock('./util/validate-user', () => ({
  validateSession: vi.fn(() => ({
    user_id: 'actual-user-1',
    auth_method: 'password',
    expires_at: -1,
  })),
}));

describe('app-ml-proxy', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.unstubAllGlobals();
    process.env.ACTUAL_ML_SHARED_SECRET = 'test-shared-secret';
  });

  afterEach(() => {
    delete process.env.ACTUAL_ML_SERVING_URL;
    delete process.env.ML_SERVING_URL;
    delete process.env.ACTUAL_ML_SHARED_SECRET;
    vi.restoreAllMocks();
  });

  it('proxies authenticated requests with a signed Actual identity', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      status: 200,
      headers: new Headers({ 'content-type': 'application/json' }),
      arrayBuffer: async () =>
        new TextEncoder().encode('{"status":"ok"}').buffer,
    });
    vi.stubGlobal('fetch', fetchMock);

    process.env.ACTUAL_ML_SERVING_URL = 'http://classifier.internal:8000';
    const { handlers } = await import('./app-ml-proxy');

    const response = await request(handlers)
      .post('/classify')
      .set('Authorization', 'Bearer spoofed')
      .set('X-ACTUAL-TOKEN', 'actual-session-token')
      .send({ transaction_id: 'txn-1' });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0][0])).toBe(
      'http://classifier.internal:8000/classify',
    );
    const headers = fetchMock.mock.calls[0][1].headers;
    expect(headers.authorization).toBeUndefined();
    expect(headers['x-actual-token']).toBeUndefined();
    expect(headers['x-actual-user-id']).toBe('actual-user-1');
    expect(headers['x-actual-username']).toBe('alice');
    const expectedSignature = crypto
      .createHmac('sha256', 'test-shared-secret')
      .update(
        [
          'actual-user-1',
          'alice',
          'POST',
          '/classify',
          '',
          headers['x-actual-auth-timestamp'],
        ].join('\n'),
      )
      .digest('hex');
    expect(headers['x-actual-auth-signature']).toBe(expectedSignature);
    expect(response.status).toBe(200);
    expect(response.body).toEqual({ status: 'ok' });
  });

  it('falls back to the next target when the first one is unreachable', async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error('connect ECONNREFUSED'))
      .mockResolvedValueOnce({
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        arrayBuffer: async () =>
          new TextEncoder().encode('{"status":"ok"}').buffer,
      });
    vi.stubGlobal('fetch', fetchMock);

    process.env.ACTUAL_ML_SERVING_URL = 'http://classifier.internal:8000';
    process.env.ML_SERVING_URL = 'http://transaction-classifier:8000';
    const { handlers } = await import('./app-ml-proxy');

    const response = await request(handlers).get('/health');

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(String(fetchMock.mock.calls[1][0])).toBe(
      'http://transaction-classifier:8000/health',
    );
    expect(response.status).toBe(200);
    expect(response.body).toEqual({ status: 'ok' });
  });

  it('rejects unauthenticated requests before forwarding to serving', async () => {
    vi.mocked(validateSession).mockImplementationOnce((_req, res) => {
      res.status(401).json({
        status: 'error',
        reason: 'unauthorized',
      });
      return null;
    });
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    process.env.ACTUAL_ML_SERVING_URL = 'http://classifier.internal:8000';
    const { handlers } = await import('./app-ml-proxy');

    const response = await request(handlers).post('/classify').send({
      transaction_id: 'txn-1',
    });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(response.status).toBe(401);
  });
});
