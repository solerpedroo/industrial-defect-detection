"""Bound inference multipart bodies before the framework parses/spools files."""

from starlette.responses import JSONResponse

from .images import MAX_UPLOAD_BYTES

MAX_INFERENCE_BODY = MAX_UPLOAD_BYTES + 64 * 1024


class InferenceUploadLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"].rstrip("/") not in {"/api/inference", "/api/inference/compare"}
        ):
            await self.app(scope, receive, send)
            return
        response = JSONResponse(
            {"detail": "The inference request exceeds its upload envelope."}, status_code=413
        )
        headers = dict(scope.get("headers", []))
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                too_large = int(declared) > MAX_INFERENCE_BODY
            except ValueError:
                too_large = True
            if too_large:
                await response(scope, receive, send)
                return
        chunks, count = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            count += len(body)
            if count > MAX_INFERENCE_BODY:
                await response(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, bounded_receive, send)
