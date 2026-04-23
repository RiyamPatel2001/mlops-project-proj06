from __future__ import annotations

import unittest
from unittest import mock

from app.auth import AuthenticatedUser, hash_password, verify_password
from app.models import (
    ClassifyRequest,
    FeedbackRequest,
    TagExampleRequest,
)
from app.routes import classify as classify_route
from app.routes import custom_categories as custom_categories_route
from app.routes import feedback as feedback_route
from app.routes import tag_example as tag_example_route


class PasswordHashTests(unittest.TestCase):
    def test_password_hash_round_trip(self) -> None:
        stored_hash = hash_password("correct-horse-battery-staple")

        self.assertNotEqual(stored_hash, "correct-horse-battery-staple")
        self.assertTrue(
            verify_password("correct-horse-battery-staple", stored_hash)
        )
        self.assertFalse(verify_password("wrong-password", stored_hash))


class AuthenticatedRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.current_user = AuthenticatedUser(
            user_id="user-123",
            username="alice",
        )

    @mock.patch("app.routes.classify.record_classification")
    @mock.patch(
        "app.routes.classify.layer2.classify",
        new_callable=mock.AsyncMock,
    )
    @mock.patch("app.routes.classify.layer1.predict")
    @mock.patch(
        "app.routes.classify.compute_features",
        return_value={"normalized_payee": "whole foods"},
    )
    async def test_classify_uses_authenticated_user_id(
        self,
        _mock_features,
        mock_predict,
        mock_layer2_classify,
        _mock_record_classification,
    ) -> None:
        del mock_predict
        mock_layer2_classify.return_value = ("Personal Groceries", 0.91)

        response = await classify_route.classify_transaction(
            ClassifyRequest(
                transaction_id="txn-1",
                payee="WHOLE FOODS",
                amount=-42.15,
                date="2026-04-20",
            ),
            self.current_user,
        )

        mock_layer2_classify.assert_awaited_once_with(
            self.current_user.user_id,
            "WHOLE FOODS",
        )
        self.assertEqual(response.user_id, self.current_user.user_id)
        self.assertEqual(response.prediction_category, "Personal Groceries")
        self.assertEqual(response.source, "layer2")

    @mock.patch("app.routes.feedback.record_feedback")
    @mock.patch(
        "app.routes.feedback.db.insert_feedback",
        new_callable=mock.AsyncMock,
    )
    async def test_feedback_uses_authenticated_user_id(
        self,
        mock_insert_feedback,
        _mock_record_feedback,
    ) -> None:
        response = await feedback_route.submit_feedback(
            FeedbackRequest(
                transaction_id="txn-2",
                payee="TARGET",
                amount=-2399,
                date="2026-04-21",
                original_prediction="Household Supplies",
                original_confidence=0.72,
                source="layer1",
                final_label="Groceries",
                reviewed_by_user=True,
                timestamp="2026-04-21T12:00:00Z",
            ),
            self.current_user,
        )

        self.assertEqual(response.status, "ok")
        row = mock_insert_feedback.await_args.args[0]
        self.assertEqual(row["user_id"], self.current_user.user_id)
        self.assertEqual(row["transaction_id"], "txn-2")

    @mock.patch(
        "app.routes.custom_categories.db.get_user_custom_categories",
        new_callable=mock.AsyncMock,
        return_value=["Family Vacation", "Pet Supplies"],
    )
    async def test_custom_categories_use_authenticated_user_id(
        self,
        mock_get_categories,
    ) -> None:
        response = await custom_categories_route.get_custom_categories(
            self.current_user,
        )

        mock_get_categories.assert_awaited_once_with(
            self.current_user.user_id,
        )
        self.assertEqual(
            response.categories,
            ["Family Vacation", "Pet Supplies"],
        )

    @mock.patch(
        "app.routes.tag_example.db.insert_layer2_example",
        new_callable=mock.AsyncMock,
        return_value=7,
    )
    @mock.patch(
        "app.routes.tag_example.layer2.get_embedding",
        new_callable=mock.AsyncMock,
        return_value=[0.1, 0.2, 0.3],
    )
    async def test_tag_example_uses_authenticated_user_id(
        self,
        mock_get_embedding,
        mock_insert_example,
    ) -> None:
        response = await tag_example_route.tag_example(
            TagExampleRequest(
                payee="WHOLE FOODS",
                custom_category="Personal Groceries",
            ),
            self.current_user,
        )

        mock_get_embedding.assert_awaited_once_with("WHOLE FOODS")
        mock_insert_example.assert_awaited_once_with(
            user_id=self.current_user.user_id,
            payee="WHOLE FOODS",
            custom_category="Personal Groceries",
            embedding=[0.1, 0.2, 0.3],
        )
        self.assertEqual(response.id, 7)


if __name__ == "__main__":
    unittest.main()
