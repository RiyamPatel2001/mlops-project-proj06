import express from 'express';
import crypto from 'node:crypto';

import { getUserInfo } from './account-db';
import { validateSession } from './util/validate-user';
import { requestLoggerMiddleware } from './util/middlewares';

const app = express();

const ML_PROXY_USER_ID_HEADER = 'x-actual-user-id';
const ML_PROXY_USERNAME_HEADER = 'x-actual-username';
const ML_PROXY_TIMESTAMP_HEADER = 'x-actual-auth-timestamp';
const ML_PROXY_SIGNATURE_HEADER = 'x-actual-auth-signature';

app.use(express.json());
app.use(requestLoggerMiddleware);

export { app as handlers };

function getKubernetesServiceTarget() {
  const host = process.env.TRANSACTION_CLASSIFIER_SERVICE_HOST;
  const port = process.env.TRANSACTION_CLASSIFIER_SERVICE_PORT || '8000';

  if (!host) {
    return null;
  }

  return `http://${host}:${port}`;
}

const DEFAULT_ML_PROXY_TARGETS = [
  process.env.ACTUAL_ML_SERVING_URL,
  process.env.ML_SERVING_URL,
  getKubernetesServiceTarget(),
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

function getSharedSecret() {
  return process.env.ACTUAL_ML_SHARED_SECRET || '';
}

function buildSignaturePayload({
  userId,
  username,
  method,
  path,
  query,
  timestamp,
}) {
  return [userId, username, method.toUpperCase(), path, query, timestamp].join(
    '\n',
  );
}

function buildSignedIdentity(req, session) {
  const user = getUserInfo(session.user_id);
  if (!user) {
    return { error: 'user-not-found' };
  }

  const secret = getSharedSecret();
  if (!secret) {
    return { error: 'missing-shared-secret' };
  }

  const url = new URL(req.url, 'http://actual.local');
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const username = user.user_name || user.display_name || session.user_id;
  const payload = buildSignaturePayload({
    userId: session.user_id,
    username,
    method: req.method,
    path: url.pathname,
    query: url.searchParams.toString(),
    timestamp,
  });
  const signature = crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');

  return {
    headers: {
      [ML_PROXY_USER_ID_HEADER]: session.user_id,
      [ML_PROXY_USERNAME_HEADER]: username,
      [ML_PROXY_TIMESTAMP_HEADER]: timestamp,
      [ML_PROXY_SIGNATURE_HEADER]: signature,
    },
  };
}

app.use('/', async (req, res) => {
  const session = validateSession(req, res);
  if (!session) {
    return;
  }

  const signedIdentity = buildSignedIdentity(req, session);
  if ('error' in signedIdentity) {
    res.status(signedIdentity.error === 'user-not-found' ? 401 : 503).json({
      error:
        signedIdentity.error === 'user-not-found'
          ? 'actual-user-not-found'
          : 'ml-proxy-not-configured',
    });
    return;
  }

  const requestHeaders = { ...req.headers };
  delete requestHeaders.host;
  delete requestHeaders['content-length'];
  delete requestHeaders.authorization;
  delete requestHeaders['x-actual-token'];
  delete requestHeaders[ML_PROXY_USER_ID_HEADER];
  delete requestHeaders[ML_PROXY_USERNAME_HEADER];
  delete requestHeaders[ML_PROXY_TIMESTAMP_HEADER];
  delete requestHeaders[ML_PROXY_SIGNATURE_HEADER];

  const body = createProxyBody(req);
  let lastError = null;
  let lastTarget = null;

  for (const target of getMlProxyTargets()) {
    const url = new URL(req.url, `${target.replace(/\/$/, '')}/`);

    try {
      const response = await fetch(url, {
        method: req.method,
        headers: {
          ...requestHeaders,
          ...signedIdentity.headers,
        },
        body,
      });

      applyProxyResponseHeaders(res, response);
      const payload = Buffer.from(await response.arrayBuffer());
      res.status(response.status).send(payload);
      return;
    } catch (error) {
      lastError = error;
      lastTarget = target;
      console.error(
        '[ml-proxy] request failed',
        JSON.stringify({
          target,
          method: req.method,
          path: req.url,
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    }
  }

  res.status(502).json({
    error: 'ml-service-unavailable',
    target: lastTarget,
    details: lastError instanceof Error ? lastError.message : String(lastError),
  });
});
