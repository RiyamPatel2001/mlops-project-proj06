import { getAccountDb } from '../src/account-db.js';

export const up = async function () {
  const accountDb = getAccountDb();

  accountDb.exec(`
    CREATE TABLE IF NOT EXISTS user_passwords
      (user_id TEXT PRIMARY KEY,
       password_hash TEXT NOT NULL,
       FOREIGN KEY (user_id) REFERENCES users(id));
  `);
};

export const down = async function () {
  const accountDb = getAccountDb();

  accountDb.exec(`
    DROP TABLE IF EXISTS user_passwords;
  `);
};
