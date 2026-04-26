import request from 'supertest';
import { v4 as uuidv4 } from 'uuid';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { getAccountDb, getLoginMethod, getServerPrefs } from './account-db';
import { changePassword } from './accounts/password';
import { handlers as app, authRateLimiter } from './app-account';

const ADMIN_ROLE = 'ADMIN';
const BASIC_ROLE = 'BASIC';

const createUser = (
  userId,
  userName,
  role,
  owner = 0,
  enabled = 1,
  displayName = `${userName} display`,
) => {
  getAccountDb().mutate(
    'INSERT INTO users (id, user_name, display_name, enabled, owner, role) VALUES (?, ?, ?, ?, ?, ?)',
    [userId, userName, displayName, enabled, owner, role],
  );
};

const deleteUser = userId => {
  getAccountDb().mutate('DELETE FROM sessions WHERE user_id = ?', [userId]);
  getAccountDb().mutate('DELETE FROM user_passwords WHERE user_id = ?', [
    userId,
  ]);
  getAccountDb().mutate('DELETE FROM user_access WHERE user_id = ?', [userId]);
  getAccountDb().mutate('DELETE FROM users WHERE id = ?', [userId]);
};

const createSession = (userId, sessionToken, authMethod = 'password') => {
  getAccountDb().mutate(
    'INSERT INTO sessions (token, user_id, expires_at, auth_method) VALUES (?, ?, ?, ?)',
    [sessionToken, userId, Math.floor(Date.now() / 1000) + 60 * 60, authMethod],
  );
};

const generateSessionToken = () => `token-${uuidv4()}`;

const clearServerPrefs = () => {
  getAccountDb().mutate('DELETE FROM server_prefs');
};

const clearAuth = () => {
  getAccountDb().mutate('DELETE FROM auth');
};

const clearUserPasswords = () => {
  getAccountDb().mutate('DELETE FROM user_passwords');
};

const clearUsers = () => {
  getAccountDb().mutate('DELETE FROM sessions');
  getAccountDb().mutate('DELETE FROM user_access');
  getAccountDb().mutate('DELETE FROM user_passwords');
  getAccountDb().mutate('DELETE FROM users');
};

beforeEach(() => {
  authRateLimiter.resetKey('127.0.0.1');
});

afterEach(() => {
  clearServerPrefs();
  clearAuth();
  clearUsers();
});

describe('auth rate limiting', () => {
  it('should return 429 after exceeding the rate limit on /login', async () => {
    for (let i = 0; i < 5; i++) {
      await request(app).post('/login').send({ userName: 'wrong', password: 'wrong' });
    }

    const res = await request(app)
      .post('/login')
      .send({ userName: 'wrong', password: 'wrong' });

    expect(res.statusCode).toEqual(429);
    expect(res.body).toEqual({
      status: 'error',
      reason: 'too-many-requests',
    });
  });

  it('should apply the same rate limit across /login and /bootstrap', async () => {
    for (let i = 0; i < 5; i++) {
      await request(app).post('/login').send({ userName: 'wrong', password: 'wrong' });
    }

    const res = await request(app)
      .post('/bootstrap')
      .send({ userName: 'first-user', password: 'test' });

    expect(res.statusCode).toEqual(429);
    expect(res.body).toEqual({
      status: 'error',
      reason: 'too-many-requests',
    });
  });

  it('should not rate limit non-auth endpoints', async () => {
    for (let i = 0; i < 6; i++) {
      await request(app).post('/login').send({ userName: 'wrong', password: 'wrong' });
    }

    const res = await request(app).get('/needs-bootstrap');
    expect(res.statusCode).toEqual(200);
  });
});

describe('getLoginMethod()', () => {
  it('always returns password', () => {
    expect(getLoginMethod(undefined)).toBe('password');
    expect(getLoginMethod({ body: { loginMethod: 'openid' } })).toBe(
      'password',
    );
  });
});

describe('/bootstrap', () => {
  it('creates the first account and marks the server bootstrapped', async () => {
    const res = await request(app).post('/bootstrap').send({
      userName: 'owner',
      displayName: 'Owner Name',
      password: 'secret-password',
    });

    expect(res.statusCode).toEqual(200);
    expect(res.body).toHaveProperty('status', 'ok');
    expect(res.body.data).toHaveProperty('token');

    const owner = getAccountDb().first(
      'SELECT user_name, display_name, owner, role FROM users WHERE user_name = ?',
      ['owner'],
    );
    expect(owner).toEqual({
      user_name: 'owner',
      display_name: 'Owner Name',
      owner: 1,
      role: ADMIN_ROLE,
    });

    const bootstrapState = await request(app).get('/needs-bootstrap');
    expect(bootstrapState.body.data.bootstrapped).toBe(true);
    expect(bootstrapState.body.data.availableLoginMethods).toEqual([
      { method: 'password', displayName: 'Password', active: true },
    ]);
  });

  it('rejects a second bootstrap after the first account exists', async () => {
    await request(app).post('/bootstrap').send({
      userName: 'owner',
      password: 'secret-password',
    });

    const res = await request(app).post('/bootstrap').send({
      userName: 'other-owner',
      password: 'another-secret',
    });

    expect(res.statusCode).toEqual(400);
    expect(res.body).toEqual({
      status: 'error',
      reason: 'already-bootstrapped',
    });
  });
});

describe('/register', () => {
  beforeEach(async () => {
    await request(app).post('/bootstrap').send({
      userName: 'owner',
      password: 'secret-password',
    });
  });

  it('creates a later user with a BASIC role and returns a session token', async () => {
    const res = await request(app).post('/register').send({
      userName: 'second-user',
      displayName: 'Second User',
      password: 'basic-password',
    });

    expect(res.statusCode).toEqual(200);
    expect(res.body).toHaveProperty('status', 'ok');
    expect(res.body.data).toHaveProperty('token');

    const user = getAccountDb().first(
      'SELECT user_name, display_name, owner, role FROM users WHERE user_name = ?',
      ['second-user'],
    );
    expect(user).toEqual({
      user_name: 'second-user',
      display_name: 'Second User',
      owner: 0,
      role: BASIC_ROLE,
    });
  });

  it('rejects duplicate usernames', async () => {
    const res = await request(app).post('/register').send({
      userName: 'owner',
      password: 'basic-password',
    });

    expect(res.statusCode).toEqual(400);
    expect(res.body).toEqual({
      status: 'error',
      reason: 'user-already-exists',
    });
  });
});

describe('/login', () => {
  beforeEach(() => {
    const userId = uuidv4();
    createUser(userId, 'user1', BASIC_ROLE);
    changePassword('secret-password', userId);
  });

  afterEach(() => {
    clearAuth();
    clearUserPasswords();
  });

  it('allows username/password login for an existing user', async () => {
    const res = await request(app).post('/login').send({
      userName: 'user1',
      password: 'secret-password',
    });

    expect(res.statusCode).toEqual(200);
    expect(res.body).toHaveProperty('status', 'ok');
    expect(res.body.data).toHaveProperty('token');
  });

  it('rejects login with the wrong password', async () => {
    const res = await request(app).post('/login').send({
      userName: 'user1',
      password: 'wrong-password',
    });

    expect(res.statusCode).toEqual(400);
    expect(res.body).toHaveProperty('reason', 'invalid-password');
  });

  it('rejects login without a username', async () => {
    const res = await request(app).post('/login').send({
      password: 'secret-password',
    });

    expect(res.statusCode).toEqual(400);
    expect(res.body).toHaveProperty('reason', 'invalid-username');
  });
});

describe('/change-password', () => {
  let userId, sessionToken;

  beforeEach(() => {
    userId = uuidv4();
    sessionToken = generateSessionToken();
    createUser(userId, 'basic', BASIC_ROLE);
    changePassword('oldpassword', userId);
    createSession(userId, sessionToken, 'password');
  });

  afterEach(() => {
    deleteUser(userId);
    clearAuth();
    clearUserPasswords();
  });

  it('should return 401 if no session token is provided', async () => {
    const res = await request(app).post('/change-password').send({
      password: 'newpassword',
    });

    expect(res.statusCode).toEqual(401);
    expect(res.body).toHaveProperty('status', 'error');
    expect(res.body).toHaveProperty('reason', 'unauthorized');
  });

  it('should return 400 when the new password is empty', async () => {
    const res = await request(app)
      .post('/change-password')
      .set('x-actual-token', sessionToken)
      .send({ password: '' });

    expect(res.statusCode).toEqual(400);
    expect(res.body).toEqual({ status: 'error', reason: 'invalid-password' });
  });

  it('should let the current user change their own password', async () => {
    const res = await request(app)
      .post('/change-password')
      .set('x-actual-token', sessionToken)
      .send({ password: 'newpassword' });

    expect(res.statusCode).toEqual(200);
    expect(res.body).toEqual({ status: 'ok', data: {} });

    const loginRes = await request(app).post('/login').send({
      userName: 'basic',
      password: 'newpassword',
    });
    expect(loginRes.statusCode).toEqual(200);
  });
});

describe('/server-prefs', () => {
  describe('POST /server-prefs', () => {
    let adminUserId, basicUserId, adminSessionToken, basicSessionToken;

    beforeEach(() => {
      adminUserId = uuidv4();
      basicUserId = uuidv4();
      adminSessionToken = generateSessionToken();
      basicSessionToken = generateSessionToken();

      createUser(adminUserId, 'admin', ADMIN_ROLE, 1);
      createUser(basicUserId, 'user', BASIC_ROLE);
      createSession(adminUserId, adminSessionToken);
      createSession(basicUserId, basicSessionToken);
    });

    afterEach(() => {
      deleteUser(adminUserId);
      deleteUser(basicUserId);
      clearServerPrefs();
    });

    it('should return 401 if no session token is provided', async () => {
      const res = await request(app)
        .post('/server-prefs')
        .send({
          prefs: { 'flags.plugins': 'true' },
        });

      expect(res.statusCode).toEqual(401);
      expect(res.body).toHaveProperty('status', 'error');
      expect(res.body).toHaveProperty('reason', 'unauthorized');
    });

    it('should return 403 if user is not an admin', async () => {
      const res = await request(app)
        .post('/server-prefs')
        .set('x-actual-token', basicSessionToken)
        .send({
          prefs: { 'flags.plugins': 'true' },
        });

      expect(res.statusCode).toEqual(403);
      expect(res.body).toEqual({
        status: 'error',
        reason: 'forbidden',
        details: 'permission-not-found',
      });
    });

    it('should return 400 if prefs is not an object', async () => {
      const res = await request(app)
        .post('/server-prefs')
        .set('x-actual-token', adminSessionToken)
        .send({
          prefs: 'invalid',
        });

      expect(res.statusCode).toEqual(400);
      expect(res.body).toEqual({
        status: 'error',
        reason: 'invalid-prefs',
      });
    });

    it('should return 200 and save server preferences for admin user', async () => {
      const prefs = { 'flags.plugins': 'true' };

      const res = await request(app)
        .post('/server-prefs')
        .set('x-actual-token', adminSessionToken)
        .send({ prefs });

      expect(res.statusCode).toEqual(200);
      expect(res.body).toEqual({
        status: 'ok',
        data: {},
      });

      expect(getServerPrefs()).toEqual(prefs);
    });
  });
});
