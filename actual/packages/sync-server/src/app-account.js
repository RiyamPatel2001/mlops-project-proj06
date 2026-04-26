import express from 'express';
import rateLimit from 'express-rate-limit';

import {
  bootstrap,
  getLoginMethod,
  getServerPrefs,
  getUserInfo,
  isAdmin,
  listLoginMethods,
  needsBootstrap,
  setServerPrefs,
} from './account-db';
import {
  changePassword,
  loginWithPassword,
  registerPasswordUser,
} from './accounts/password';
import { errorMiddleware, requestLoggerMiddleware } from './util/middlewares';
import { validateSession } from './util/validate-user';

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(errorMiddleware);
app.use(requestLoggerMiddleware);

const authRateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 attempts per window
  legacyHeaders: false,
  standardHeaders: true,
  message: { status: 'error', reason: 'too-many-requests' },
});

export { app as handlers, authRateLimiter };

// Non-authenticated endpoints:
//
// /needs-bootstrap
// /boostrap (special endpoint for setting up the instance, cant call again)
// /login

app.get('/needs-bootstrap', (req, res) => {
  const availableLoginMethods = listLoginMethods();
  res.send({
    status: 'ok',
    data: {
      bootstrapped: !needsBootstrap(),
      loginMethod: getLoginMethod(),
      availableLoginMethods,
      multiuser: false,
    },
  });
});

app.post('/bootstrap', authRateLimiter, async (req, res) => {
  const boot = await bootstrap(req.body);

  if (boot?.error) {
    res.status(400).send({ status: 'error', reason: boot?.error });
    return;
  }
  res.send({ status: 'ok', data: boot });
});

app.get('/login-methods', (req, res) => {
  const methods = listLoginMethods();
  res.send({ status: 'ok', methods });
});

app.post('/login', authRateLimiter, async (req, res) => {
  const tokenRes = loginWithPassword(req.body.password, req.body.userName);
  const { error, token } = tokenRes;

  if (error) {
    res.status(400).send({ status: 'error', reason: error });
    return;
  }

  res.send({ status: 'ok', data: { token } });
});

app.post('/register', authRateLimiter, async (req, res) => {
  const creatingFirstUser = needsBootstrap();
  const result = registerPasswordUser({
    userName: req.body.userName,
    password: req.body.password,
    displayName: req.body.displayName,
    owner: creatingFirstUser,
  });

  if (result.error) {
    res.status(400).send({ status: 'error', reason: result.error });
    return;
  }

  res.send({ status: 'ok', data: { token: result.token } });
});

app.post('/change-password', (req, res) => {
  const session = validateSession(req, res);
  if (!session) return;

  if (session.auth_method !== 'password') {
    res.status(403).send({
      status: 'error',
      reason: 'forbidden',
      details: 'password-auth-not-active',
    });
    return;
  }

  const user = getUserInfo(session.user_id);
  if (!user) {
    res.status(400).send({ status: 'error', reason: 'user-not-found' });
    return;
  }

  const { error } = changePassword(req.body.password, session.user_id);

  if (error) {
    res.status(400).send({ status: 'error', reason: error });
    return;
  }

  res.send({ status: 'ok', data: {} });
});

app.post('/server-prefs', (req, res) => {
  const session = validateSession(req, res);
  if (!session) return;

  if (!isAdmin(session.user_id)) {
    res.status(403).send({
      status: 'error',
      reason: 'forbidden',
      details: 'permission-not-found',
    });
    return;
  }

  const { prefs } = req.body || {};

  if (!prefs || typeof prefs !== 'object') {
    res.status(400).send({ status: 'error', reason: 'invalid-prefs' });
    return;
  }

  setServerPrefs(prefs);

  res.send({ status: 'ok', data: {} });
});

app.get('/validate', (req, res) => {
  const session = validateSession(req, res);
  if (session) {
    const user = getUserInfo(session.user_id);
    if (!user) {
      res.status(400).send({ status: 'error', reason: 'User not found' });
      return;
    }

    res.send({
      status: 'ok',
      data: {
        validated: true,
        userName: user?.user_name,
        permission: user?.role,
        userId: session?.user_id,
        displayName: user?.display_name,
        loginMethod: session?.auth_method,
        prefs: getServerPrefs(),
      },
    });
  }
});
