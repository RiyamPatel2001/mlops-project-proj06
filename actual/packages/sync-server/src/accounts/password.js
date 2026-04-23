import * as bcrypt from 'bcrypt';
import { v4 as uuidv4 } from 'uuid';

import { clearExpiredSessions, getAccountDb } from '#account-db';
import { config } from '#load-config';
import {
  getUserForPasswordLogin,
  setUserPasswordHash,
} from '#services/user-service';
import { TOKEN_EXPIRATION_NEVER } from '#util/validate-user';

function isValidPassword(password) {
  return password != null && password !== '';
}

function hashPassword(password) {
  return bcrypt.hashSync(password, 12);
}

function normalizeUserName(userName) {
  if (typeof userName !== 'string') {
    return null;
  }

  const trimmed = userName.trim();
  return trimmed === '' ? null : trimmed;
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

function getLegacyPasswordUserId() {
  const accountDb = getAccountDb();
  const { totalOfUsers } = accountDb.first(
    'SELECT count(*) as totalOfUsers FROM users',
  );

  if (totalOfUsers === 0) {
    const userId = uuidv4();
    accountDb.mutate(
      'INSERT INTO users (id, user_name, display_name, enabled, owner, role) VALUES (?, ?, ?, 1, 1, ?)',
      [userId, '', '', 'ADMIN'],
    );
    return userId;
  }

  const { id: userId } = accountDb.first(
    'SELECT id FROM users WHERE user_name = ?',
    [''],
  ) || { id: null };

  return userId;
}

export function bootstrapPassword(password) {
  if (!isValidPassword(password)) {
    return { error: 'invalid-password' };
  }

  const hashed = hashPassword(password);
  const accountDb = getAccountDb();
  accountDb.transaction(() => {
    accountDb.mutate('DELETE FROM auth WHERE method = ?', ['password']);
    accountDb.mutate('UPDATE auth SET active = 0');
    accountDb.mutate(
      "INSERT INTO auth (method, display_name, extra_data, active) VALUES ('password', 'Password', ?, 1)",
      [hashed],
    );
  });

  return {};
}

export function loginWithPassword(password, userName = null) {
  if (!isValidPassword(password)) {
    return { error: 'invalid-password' };
  }

  const normalizedUserName = normalizeUserName(userName);
  if (normalizedUserName) {
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

  const accountDb = getAccountDb();
  const { extra_data: passwordHash } =
    accountDb.first('SELECT extra_data FROM auth WHERE method = ?', [
      'password',
    ]) || {};

  if (!passwordHash) {
    return { error: 'invalid-password' };
  }

  const confirmed = bcrypt.compareSync(password, passwordHash);

  if (!confirmed) {
    return { error: 'invalid-password' };
  }

  const userId = getLegacyPasswordUserId();
  if (!userId) {
    return { error: 'user-not-found' };
  }

  const token = upsertPasswordSession(userId);
  return { token };
}

export function changePassword(newPassword, userId = null) {
  if (!isValidPassword(newPassword)) {
    return { error: 'invalid-password' };
  }

  const hashed = hashPassword(newPassword);

  if (userId) {
    setUserPasswordHash(userId, hashed);
    return {};
  }

  const accountDb = getAccountDb();
  accountDb.mutate(
    "UPDATE auth SET extra_data = ? WHERE method = 'password'",
    [hashed],
  );
  return {};
}

export function checkPassword(password) {
  if (!isValidPassword(password)) {
    return false;
  }

  const accountDb = getAccountDb();
  const { extra_data: passwordHash } =
    accountDb.first('SELECT extra_data FROM auth WHERE method = ?', [
      'password',
    ]) || {};

  if (!passwordHash) {
    return false;
  }

  const confirmed = bcrypt.compareSync(password, passwordHash);

  if (!confirmed) {
    return false;
  }

  return true;
}
