"""Policy facts use real ownership-scoped records, never demo fallback."""

from datetime import datetime
from unittest.mock import MagicMock

from src.orders.order_service import OrderInfo, OrderService


def test_policy_lookup_does_not_fallback_to_sample_orders():
    info, status = OrderService().lookup_policy_order("ORD20260701001", user_id=1)
    assert info is None and status == "unavailable"


def test_policy_lookup_keeps_owner_arguments(monkeypatch):
    calls = []

    def query(order_id, **kwargs):
        calls.append((order_id, kwargs))
        return OrderInfo(order_id, "已签收", signed_at="2026-11-10"), True

    service = OrderService()
    monkeypatch.setattr(service, "_query_order_db", query)
    assert service.lookup_policy_order("ord000001", user_id=7)[1] == "found"
    assert calls == [("ORD000001", {"user_id": 7, "is_admin": False})]
    assert service.lookup_policy_order("ORD000001")[1] == "not_found"
    assert len(calls) == 1


def test_signature_comes_from_completion_not_estimated_delivery(monkeypatch):
    service = OrderService()
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = [
        (
            "ORD000001",
            "completed",
            50,
            datetime(2026, 11, 1),
            datetime(2026, 11, 2),
            datetime(2026, 11, 10),
        ),
        ("顺丰", "SF000001"),
    ]
    monkeypatch.setattr(service, "_connect", lambda: conn)
    info, status = service.lookup_policy_order("ORD000001", user_id=7)
    assert status == "found" and info.signed_at == "2026-11-10"
    assert cursor.execute.call_args_list[0].args[1] == ("ORD000001", 7)
