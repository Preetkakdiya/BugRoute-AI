"""
BugRoute AI — Unified Project Test Suite
Contains unit & integration tests for data loading, preprocessing, log parsing, multi-model predictions, and logger safety.
"""

import os
import sys
import time
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.log_parser import extract_log_issues
from src.predict import BugRoutePredictor
from src.prediction_logger import save_prediction_correction


class TestLogParser(unittest.TestCase):
    def test_empty_log(self):
        self.assertEqual(extract_log_issues(""), [])
        self.assertEqual(extract_log_issues(None), [])
        self.assertEqual(extract_log_issues("   \n\n  "), [])

    def test_log_without_errors(self):
        normal_log = (
            "2026-10-10 10:00:00 [INFO] Server initialization complete.\n"
            "2026-10-10 10:00:01 [INFO] Connected to port 8080.\n"
            "2026-10-10 10:00:05 [DEBUG] Cache warmed up with 50 items.\n"
        )
        self.assertEqual(extract_log_issues(normal_log), [])

    def test_single_error_log(self):
        single_error = (
            "2026-10-10 10:00:00 [INFO] Starting auth service...\n"
            "2026-10-10 10:01:23 [ERROR] Auth validation failed: Invalid token signature\n"
            "ExpiredJwtException: JWT signature has expired at 2026-10-10T10:00:00Z\n"
            "    at com.example.jwt.TokenVerifier.verify(TokenVerifier.java:88)\n"
            "    at com.example.auth.AuthFilter.doFilter(AuthFilter.java:42)\n"
        )
        issues = extract_log_issues(single_error)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["issue_number"], 1)
        self.assertIn("Auth validation failed", issues[0]["title"])
        self.assertIn("ExpiredJwtException", issues[0]["description"])
        self.assertIn("TokenVerifier.java:88", issues[0]["description"])

    def test_multi_issue_log(self):
        multi_log = (
            "2026-10-10 10:00:00 [INFO] Application started.\n"
            "2026-10-10 10:01:23 [ERROR] Database connection failed: Connection refused\n"
            "java.sql.SQLException: Cannot get connection from pool\n"
            "    at com.zaxxer.hikari.pool.HikariPool.getConnection(HikariPool.java:213)\n"
            "    at com.example.dao.UserDao.getUser(UserDao.java:45)\n"
            "2026-10-10 10:02:10 [INFO] Request processed in 12ms\n"
            "2026-10-10 10:03:45 [CRITICAL] OutOfMemoryError: Java heap space\n"
            "java.lang.OutOfMemoryError: Java heap space\n"
            "    at java.util.Arrays.copyOf(Arrays.java:3210)\n"
            "2026-10-10 10:04:12 [ERROR] NullPointerException in UserAuthFilter\n"
            "java.lang.NullPointerException: Cannot invoke getSession()\n"
            "    at com.example.auth.UserAuthFilter.doFilter(UserAuthFilter.java:15)\n"
        )
        issues = extract_log_issues(multi_log)
        self.assertEqual(len(issues), 3)
        self.assertIn("Database connection failed", issues[0]["title"])
        self.assertEqual(issues[1]["log_level"], "CRITICAL")
        self.assertIn("NullPointerException", issues[2]["description"])

    def test_python_traceback_log(self):
        py_log = (
            "INFO:     127.0.0.1:51234 - GET /api/v1/health 200 OK\n"
            "ERROR:    127.0.0.1:51235 - POST /api/v1/checkout 500 Internal Server Error\n"
            "Traceback (most recent call last):\n"
            "  File \"/app/views/checkout.py\", line 42, in process_payment\n"
            "    response = stripe.Charge.create(amount=amount)\n"
            "stripe.error.CardError: Your card was declined.\n"
        )
        issues = extract_log_issues(py_log)
        self.assertEqual(len(issues), 1)
        self.assertIn("Traceback", issues[0]["description"])
        self.assertIn("stripe.error.CardError", issues[0]["description"])


class TestModelPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = BugRoutePredictor()

    def test_all_five_models(self):
        test_title = "SQLException: Connection pool deadlock on POST /api/orders"
        test_desc = "Transaction aborted due to lock wait timeout exceeded on PostgreSQL connection pool."

        models = [
            "Linear SVM",
            "Logistic Regression",
            "BiLSTM + Attention",
            "DistilBERT",
            "CodeBERT"
        ]

        for m in models:
            with self.subTest(model=m):
                res = self.predictor.predict(test_title, test_desc, m)
                self.assertNotIn("error", res, f"Model '{m}' returned an error: {res.get('error')}")
                self.assertIn("predicted_component", res)
                self.assertIn("confidence", res)
                self.assertTrue(0.0 <= res["confidence"] <= 1.0)


class TestPredictionLogger(unittest.TestCase):
    def test_save_prediction_correction(self):
        correction = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "title": "Test Correction Entry",
            "description": "Database connection pool timeout",
            "model_used": "Linear SVM",
            "predicted_class": "Database",
            "predicted_confidence": 0.95,
            "user_corrected_class": "Backend API",
            "user_feedback": "Unit test feedback verification"
        }
        res = save_prediction_correction(correction)
        self.assertTrue(res)


def run_all_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestLogParser))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPredictions))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionLogger))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
