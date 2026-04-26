import * as bcrypt from 'bcrypt';
import { v4 as uuidv4 } from 'uuid';

import { clearExpiredSessions, getAccountDb } from '#account-db';
import { config } from '#load-config';
import {
  getNamedUserCount,
  getUserByUsername,
  getUserForPasswordLogin,
  insertUser,
  setUserPasswordHash,
} from '#services/user-service';
import { TOKEN_EXPIRATION_NEVER } from '#util/validate-user';

function isValidPassword(password) {
  return password != null && password !== '';
}

function normalizeUserName(userName) {
  if (typeof userName !== 'string') {
    return null;
  }

  const trimmed = userName.trim();
  return trimmed === '' ? null : trimmed;
}

function normalizeDisplayName(displayName) {
  if (typeof displayName !== 'string') {
    return null;
  }

  const trimmed = displayName.trim();
  return trimmed === '' ? null : trimmed;
}

function hashPassword(password) {
  return bcrypt.hashSync(password, 12);
}

function getTokenExpiration() {
  let expiration = TOKEN_EXPIRATION_NEVER;
  if (
    config.get('token_expiration') !== 'never' &&
    config.get('token_expiration') !== 'openid-provider' &&
    typeof config.get('token_expiration') === 'number'
  ) {
    expiration =
      Math.floor(Date.now() / 1000) + config.get('token_expiration') * 60;
  }
  return expiration;
}

function upsertPasswordSession(userId) {
  const accountDb = getAccountDb();
  const sessionRow = accountDb.first(
    'SELECT * FROM sessions WHERE auth_method = ? AND user_id = ?',
    ['password', userId],
  );
  const token = sessionRow ? sessionRow.token : uuidv4();
  const expiration = getTokenExpiration();

  if (!sessionRow) {
    accountDb.mutate(
      'INSERT INTO sessions (token, expires_at, user_id, auth_method) VALUES (?, ?, ?, ?)',
      [token, expiration, userId, 'password'],
    );
  } else {
    accountDb.mutate(
      'UPDATE sessions SET expires_at = ? WHERE token = ?',
      [expiration, token],
    );
  }

  clearExpiredSessions();
  return token;
}

export function ensurePasswordAuthInitialized() {
  const accountDb = getAccountDb();
  const existingPasswordAuth = accountDb.first(
    'SELECT method FROM auth WHERE method = ?',
    ['password'],
  );

  if (existingPasswordAuth) {
    accountDb.transaction(() => {
      accountDb.mutate('UPDATE auth SET active = 0');
      accountDb.mutate(
        "UPDATE auth SET display_name = 'Password', extra_data = NULL, active = 1 WHERE method = 'password'",
      );
      accountDb.mutate("DELETE FROM auth WHERE method <> 'password'");
    });
    return;
  }

  accountDb.transaction(() => {
    accountDb.mutate('DELETE FROM auth');
    accountDb.mutate(
      "INSERT INTO auth (method, display_name, extra_data, active) VALUES ('password', 'Password', NULL, 1)",
    );
  });
}

export function registerPasswordUser({
  userName,
  password,
  displayName = null,
  owner = false,
} = {}) {
  const normalizedUserName = normalizeUserName(userName);
  if (!normalizedUserName) {
    return { error: 'invalid-username' };
  }

  if (!isValidPassword(password)) {
    return { error: 'invalid-password' };
  }

  ensurePasswordAuthInitialized();

  if (getUserByUsername(normalizedUserName)) {
    return { error: 'user-already-exists' };
  }

  const userId = uuidv4();
  insertUser(
    userId,
    normalizedUserName,
    normalizeDisplayName(displayName),
    1,
    owner ? 'ADMIN' : 'BASIC',
    owner ? 1 : 0,
  );
  setUserPasswordHash(userId, hashPassword(password));

  const token = upsertPasswordSession(userId);
  return { token, userId };
}

export function canRegisterFirstUser() {
  return getNamedUserCount() === 0;
}

export function loginWithPassword(password, userName) {
  if (!isValidPassword(password)) {
    return { error: 'invalid-password' };
  }

  const normalizedUserName = normalizeUserName(userName);
  if (!normalizedUserName) {
    return { error: 'invalid-username' };
  }

  const user = getUserForPasswordLogin(normalizedUserName);
  if (!user?.passwordHash || user.enabled !== 1) {
    return { error: 'invalid-password' };
  }

  const confirmed = bcrypt.compareSync(password, user.passwordHash);
  if (!confirmed) {
    return { error: 'invalid-password' };
  }

  const token = upsertPasswordSession(user.id);
  return { token };
}

export function changePassword(newPassword, userId) {
  if (!isValidPassword(newPassword)) {
    return { error: 'invalid-password' };
  }

  if (!userId) {
    return { error: 'user-not-found' };
  }

  setUserPasswordHash(userId, hashPassword(newPassword));
  return {};
}
