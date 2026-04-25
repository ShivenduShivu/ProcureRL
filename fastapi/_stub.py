import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class _Route:
    method: str
    path: str
    endpoint: Callable
    response_model: Any = None


def _match_path(route_path: str, actual_path: str) -> Optional[Dict[str, str]]:
    route_parts = route_path.strip("/").split("/") if route_path.strip("/") else []
    actual_parts = actual_path.strip("/").split("/") if actual_path.strip("/") else []
    if len(route_parts) != len(actual_parts):
        return None

    params: Dict[str, str] = {}
    for route_part, actual_part in zip(route_parts, actual_parts):
        if route_part.startswith("{") and route_part.endswith("}"):
            params[route_part[1:-1]] = actual_part
        elif route_part != actual_part:
            return None
    return params


def _serialize_response(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


class FastAPI:
    def __init__(self, title: str = "", description: str = "", version: str = ""):
        self.title = title
        self.description = description
        self.version = version
        self.routes: List[_Route] = []
        self.middleware: List[Tuple[Any, Dict[str, Any]]] = []

    def add_middleware(self, middleware_class: Any, **kwargs):
        self.middleware.append((middleware_class, kwargs))

    def get(self, path: str, response_model: Any = None):
        return self._register("GET", path, response_model)

    def post(self, path: str, response_model: Any = None):
        return self._register("POST", path, response_model)

    def _register(self, method: str, path: str, response_model: Any):
        def decorator(func: Callable):
            self.routes.append(_Route(method=method, path=path, endpoint=func, response_model=response_model))
            return func

        return decorator

    def _find_route(self, method: str, path: str) -> Tuple[_Route, Dict[str, str]]:
        for route in self.routes:
            if route.method != method.upper():
                continue
            params = _match_path(route.path, path)
            if params is not None:
                return route, params
        raise HTTPException(status_code=404, detail=f"Route {method} {path} not found")

    def _dispatch(self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
        route, path_params = self._find_route(method, path)
        try:
            endpoint = route.endpoint
            annotations = getattr(endpoint, "__annotations__", {})
            args = []
            kwargs = dict(path_params)

            if json_body is not None:
                parameter_names = [name for name in annotations.keys() if name != "return"]
                if parameter_names:
                    first_name = parameter_names[0]
                    annotation = annotations.get(first_name)
                    if annotation is not None and hasattr(annotation, "__mro__"):
                        try:
                            request_obj = annotation(**json_body)
                            kwargs = {first_name: request_obj, **path_params}
                        except Exception:
                            kwargs = {first_name: json_body, **path_params}

            result = endpoint(*args, **kwargs)
            return 200, _serialize_response(result)
        except HTTPException as exc:
            return exc.status_code, {"detail": exc.detail}


class CORSMiddleware:
    def __init__(self, app: FastAPI, **kwargs):
        self.app = app
        self.kwargs = kwargs


class _Response:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    @property
    def text(self) -> str:
        return json.dumps(self._payload)


class TestClient:
    def __init__(self, app: FastAPI):
        self.app = app

    def get(self, path: str):
        status_code, payload = self.app._dispatch("GET", path)
        return _Response(status_code, payload)

    def post(self, path: str, json: Optional[Dict[str, Any]] = None):
        status_code, payload = self.app._dispatch("POST", path, json_body=json or {})
        return _Response(status_code, payload)
