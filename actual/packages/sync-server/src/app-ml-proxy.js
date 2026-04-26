import express from 'express';

import { requestLoggerMiddleware } from './util/middlewares';

const app = express();

app.use(express.json());
app.use(requestLoggerMiddleware);

export { app as handlers };

const DEFAULT_ML_PROXY_TARGETS = [
  process.env.ACTUAL_ML_SERVING_URL,
  process.env.ML_SERVING_URL,
  'http://transaction-classifier.mlops.svc.cluster.local:8000',
  'http://transaction-classifier:8000',
  'http://127.0.0.1:8000',
].filter(Boolean);

function getMlProxyTargets() {
  return [...new Set(DEFAULT_ML_PROXY_TARGETS)];
}

function createProxyBody(req) {
  if (req.method === 'GET' || req.method === 'HEAD') {
    return undefined;
  }
  if (req.body == null) {
    return undefined;
  }
  if (Buffer.isBuffer(req.body) || typeof req.body === 'string') {
    return req.body;
  }
  return JSON.stringify(req.body);
}

function applyProxyResponseHeaders(res, response) {
  response.headers.forEach((value, key) => {
    if (
      key === 'connection' ||
      key === 'content-length' ||
      key === 'transfer-encoding'
    ) {
      return;
    }
    res.set(key, value);
  });
}

app.use('/', async (req, res) => {
  const requestHeaders = { ...req.headers };
  delete requestHeaders.host;
  delete requestHeaders['content-length'];

  const body = createProxyBody(req);
  let lastError = null;

  for (const target of getMlProxyTargets()) {
    const url = new URL(req.url, `${target.replace(/\/$/, '')}/`);

    try {
      const response = await fetch(url, {
        method: req.method,
        headers: requestHeaders,
        body,
      });

      applyProxyResponseHeaders(res, response);
      const payload = Buffer.from(await response.arrayBuffer());
      res.status(response.status).send(payload);
      return;
    } catch (error) {
      lastError = error;
    }
  }

  res.status(502).json({
    error: 'ml-service-unavailable',
    details: lastError instanceof Error ? lastError.message : String(lastError),
  });
});
