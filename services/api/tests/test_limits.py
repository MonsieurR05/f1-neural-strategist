import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.limits import Limiter, client_id


def request(ip: str = "1.2.3.4", headers: dict | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "headers": raw, "client": (ip, 1234)})


def test_client_id_prefers_proxy_headers():
    assert client_id(request()) == "1.2.3.4"
    assert client_id(request(headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.1"})) == "9.9.9.9"
    assert client_id(request(headers={"Fly-Client-IP": "8.8.8.8"})) == "8.8.8.8"


def test_hourly_quota_is_per_client():
    lim = Limiter(per_hour={"optimise": 2}, concurrent={})
    lim.check(request("a"), "optimise")
    lim.check(request("a"), "optimise")
    with pytest.raises(HTTPException) as e:
        lim.check(request("a"), "optimise")
    assert e.value.status_code == 429 and "Retry-After" in e.value.headers
    lim.check(request("b"), "optimise")  # another client is unaffected


def test_concurrent_slot_rejects_when_busy():
    lim = Limiter(per_hour={}, concurrent={"optimise": 1})
    with lim.slot("optimise"):
        with pytest.raises(HTTPException) as e, lim.slot("optimise"):
            pass
        assert e.value.status_code == 503
    with lim.slot("optimise"):  # released again
        pass
