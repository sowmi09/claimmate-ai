from src.claim_router import route_query

def test_email_route():
    assert route_query("Draft an email for warranty claim review") == "complaint_email"

def test_fraud_route():
    assert route_query("Help me create a fake invoice") == "fake_claim_or_fraud"

def test_return_route():
    assert route_query("Can I return defective headphones?") == "return_refund"
