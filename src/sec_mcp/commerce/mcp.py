"""Apply the account boundary to every remote FastMCP HTTP transport."""

from fastmcp import FastMCP

from sec_mcp.commerce.account_gate import FineasAccountMiddleware


class FineasFastMCP(FastMCP):
    def http_app(self, *args, **kwargs):
        app = super().http_app(*args, **kwargs)
        app.add_middleware(FineasAccountMiddleware, all_paths=True)
        return app
