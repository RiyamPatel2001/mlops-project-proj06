from __future__ import annotations

import os
import tempfile
import unittest

from fastapi import HTTPException

from app import db
from app.auth import require_authenticated_user
from app.models import AuthLoginRequest, AuthRegisterRequest
from app.routes import auth as auth_route


class SqliteAuthFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["AUTH_SQLITE_PATH"] = os.path.join(
            self.temp_dir.name,
            "auth.sqlite3",
        )
        await db.close_pool()
        await db.ensure_tables()

    async def asyncTearDown(self) -> None:
        await db.close_pool()
        os.environ.pop("AUTH_SQLITE_PATH", None)
        self.temp_dir.cleanup()

    async def test_register_and_login_work_without_postgres(self) -> None:
        registered = await auth_route.register_user(
            AuthRegisterRequest(username="Jayraj", password="secret"),
        )
        logged_in = await auth_route.login_user(
            AuthLoginRequest(username="jayraj", password="secret"),
        )
        current_user = await require_authenticated_user(
            authorization=f"Bearer {logged_in.token}",
        )

        self.assertEqual(registered.username, "Jayraj")
        self.assertEqual(logged_in.username, "Jayraj")
        self.assertEqual(current_user.username, "Jayraj")

    async def test_duplicate_username_returns_conflict(self) -> None:
        await auth_route.register_user(
            AuthRegisterRequest(username="jayraj", password="secret"),
        )

        with self.assertRaises(HTTPException) as exc:
            await auth_route.register_user(
                AuthRegisterRequest(username="JAYRAJ", password="other-secret"),
            )

        self.assertEqual(exc.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
