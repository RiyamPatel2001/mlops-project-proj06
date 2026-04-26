import * as asyncStorage from '#platform/server/asyncStorage';
import { fetch } from '#platform/server/fetch';
import { createApp } from '#server/app';
import { getServer } from '#server/server-config';

type MlRequestError = 'auth' | 'network' | null;

export type MlRequestResult = {
  status: number;
  data: unknown | null;
  detail: string | null;
  error: MlRequestError;
};

export type MlHandlers = {
  'ml-get': typeof mlGet;
  'ml-post': typeof mlPost;
};

export const app = createApp<MlHandlers>();
app.method('ml-get', mlGet);
app.method('ml-post', mlPost);

function normalizePath(path: string) {
  return path.startsWith('/') ? path : `/${path}`;
}

async function readResponsePayload(
  response: Response,
): Promise<{ data: unknown | null; detail: string | null }> {
  const text = await response.text();
  if (!text) {
    return { data: null, detail: null };
  }

  try {
    const payload = JSON.parse(text);
    if (response.ok) {
      return { data: payload, detail: null };
    }

    return {
      data: null,
      detail:
        payload && typeof payload === 'object' && 'detail' in payload
          ? String(payload.detail)
          : null,
    };
  } catch {
    return { data: null, detail: null };
  }
}

async function requestMlService(
  method: 'GET' | 'POST',
  path: string,
  body?: unknown,
): Promise<MlRequestResult> {
  const serverConfig = getServer();
  if (!serverConfig) {
    return {
      status: 0,
      data: null,
      detail: 'no-server-configured',
      error: 'network',
    };
  }

  const userToken = await asyncStorage.getItem('user-token');
  if (!userToken) {
    return {
      status: 401,
      data: null,
      detail: 'missing-actual-session',
      error: 'auth',
    };
  }

  try {
    const response = await fetch(
      `${serverConfig.BASE_SERVER}/ml${normalizePath(path)}`,
      {
        method,
        headers: {
          'Content-Type': 'application/json',
          'X-ACTUAL-TOKEN': userToken,
        },
        body: method === 'POST' ? JSON.stringify(body ?? {}) : undefined,
      },
    );
    const payload = await readResponsePayload(response);

    return {
      status: response.status,
      data: payload.data,
      detail: payload.detail,
      error: response.status === 401 ? 'auth' : null,
    };
  } catch {
    return {
      status: 0,
      data: null,
      detail: null,
      error: 'network',
    };
  }
}

async function mlGet({ path }: { path: string }): Promise<MlRequestResult> {
  return requestMlService('GET', path);
}

async function mlPost({
  path,
  body,
}: {
  path: string;
  body: unknown;
}): Promise<MlRequestResult> {
  return requestMlService('POST', path, body);
}
