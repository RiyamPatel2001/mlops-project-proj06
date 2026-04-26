import { join, resolve } from 'node:path';

import {
  canRegisterFirstUser,
  ensurePasswordAuthInitialized,
  registerPasswordUser,
} from './accounts/password';
import { openDatabase } from './db';
import { config } from './load-config';

let _accountDb;

export function getAccountDb() {
  if (_accountDb === undefined) {
    const dbPath = join(resolve(config.get('serverFiles')), 'account.sqlite');
    _accountDb = openDatabase(dbPath);
  }

  return _accountDb;
}

export function needsBootstrap() {
  return canRegisterFirstUser();
}

export function listLoginMethods() {
  return [{ method: 'password', active: true, displayName: 'Password' }];
}

export function getActiveLoginMethod() {
  return 'password';
}

export function isMultiuserAuthEnabled() {
  return false;
}

/*
 * Get the Login Method in the following order
 * req (the frontend can say which method in the case it wants to resort to forcing password auth)
 * config options
 * fall back to using password
 */
export function getLoginMethod(req) {
  return 'password';
}

export async function bootstrap(loginSettings, forced = false) {
  if (!loginSettings) {
    return { error: 'invalid-login-settings' };
  }
  if (!forced && !needsBootstrap()) {
    return { error: 'already-bootstrapped' };
  }

  const result = registerPasswordUser({
    userName: loginSettings.userName,
    password: loginSettings.password,
    displayName: loginSettings.displayName,
    owner: true,
  });

  return result.error ? { error: result.error } : { token: result.token };
}

export function isAdmin(userId) {
  return hasPermission(userId, 'ADMIN');
}

export function hasPermission(userId, permission) {
  return getUserPermission(userId) === permission;
}

export async function enableOpenID(loginSettings) {
  return { error: 'openid-disabled' };
}

export async function disableOpenID(loginSettings) {
  ensurePasswordAuthInitialized();
  getAccountDb().mutate('DELETE FROM sessions');
}

export function getSession(token) {
  const accountDb = getAccountDb();
  return accountDb.first('SELECT * FROM sessions WHERE token = ?', [token]);
}

export function getUserInfo(userId) {
  const accountDb = getAccountDb();
  return accountDb.first('SELECT * FROM users WHERE id = ?', [userId]);
}

export function getUserPermission(userId) {
  const accountDb = getAccountDb();
  const { role } = accountDb.first(
    `SELECT role FROM users
          WHERE users.id = ?`,
    [userId],
  ) || { role: '' };

  return role;
}

export function getServerPrefs() {
  const accountDb = getAccountDb();
  const rows = accountDb.all('SELECT key, value FROM server_prefs') || [];

  return rows.reduce((prefs, row) => {
    prefs[row.key] = row.value;
    return prefs;
  }, {});
}

export function setServerPrefs(prefs) {
  const accountDb = getAccountDb();

  if (!prefs) {
    return;
  }

  accountDb.transaction(() => {
    Object.entries(prefs).forEach(([key, value]) => {
      accountDb.mutate(
        'INSERT INTO server_prefs (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value',
        [key, value],
      );
    });
  });
}

export function clearExpiredSessions() {
  const clearThreshold = Math.floor(Date.now() / 1000) - 3600;

  const deletedSessions = getAccountDb().mutate(
    'DELETE FROM sessions WHERE expires_at <> -1 and expires_at < ?',
    [clearThreshold],
  ).changes;

  console.log(`Deleted ${deletedSessions} old sessions`);
}
