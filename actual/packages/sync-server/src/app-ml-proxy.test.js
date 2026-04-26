import request from 'supertest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

describe('app-ml-proxy', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.unstubAllGlobals();
  });

  afterEach(() => {
    delete process.env.ACTUAL_ML_SERVING_URL;
    delete process.env.ML_SERVING_URL;
    vi.restoreAllMocks();
  });

  it('proxies auth requests to the configured classifier target', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      status: 201,
      headers: new Headers({ 'content-type': 'application/json' }),
      arrayBuffer: async () =>
        new TextEncoder().encode('{"username":"jayraj"}').buffer,
    });
    vi.stubGlobal('fetch', fetchMock);

    process.env.ACTUAL_ML_SERVING_URL = 'http://classifier.internal:8000';
    const { handlers } = await import('./app-ml-proxy');

    const response = await request(handlers)
      .post('/auth/register')
      .send({ username: 'jayraj', password: 'secret' });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0][0])).toBe(
      'http://classifier.internal:8000/auth/register',
    );
    expect(response.status).toBe(201);
    expect(response.body).toEqual({ username: 'jayraj' });
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
});
