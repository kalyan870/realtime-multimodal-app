import unittest

from app.main import app
from fastapi.middleware.cors import CORSMiddleware


async def send_preflight(origin: str) -> list[dict]:
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "OPTIONS",
        "scheme": "http",
        "path": "/health",
        "raw_path": b"/health",
        "query_string": b"",
        "headers": [
            (b"origin", origin.encode()),
            (b"access-control-request-method", b"GET"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 7860),
    }
    await app(scope, receive, send)
    return sent


class CorsPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_ui_origin_is_allowed(self):
        messages = await send_preflight("http://localhost:8501")
        response = next(message for message in messages if message["type"] == "http.response.start")
        headers = dict(response["headers"])
        self.assertEqual(headers[b"access-control-allow-origin"], b"http://localhost:8501")

    async def test_unconfigured_origin_is_not_allowed(self):
        messages = await send_preflight("https://untrusted.example")
        response = next(message for message in messages if message["type"] == "http.response.start")
        headers = dict(response["headers"])
        self.assertNotIn(b"access-control-allow-origin", headers)

    def test_credentials_are_disabled_and_wildcard_is_not_allowed(self):
        cors = next(item for item in app.user_middleware if item.cls is CORSMiddleware)
        self.assertNotIn("*", cors.kwargs["allow_origins"])
        self.assertFalse(cors.kwargs["allow_credentials"])


if __name__ == "__main__":
    unittest.main()
