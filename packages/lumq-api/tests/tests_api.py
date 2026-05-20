"""lumq-api integration tests."""
import math, pytest
from httpx import ASGITransport, AsyncClient

@pytest.fixture
def app():
    from lumq.api.app import create_app
    return create_app()

@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

async def test_health(client):
    r = await client.get("/health")
    assert r.status_code==200 and r.json()["status"]=="ok"

async def test_resource_empty(client):
    r = await client.post("/compiler/resource", json={"n_modes":2,"ops":[]})
    assert r.status_code==200 and r.json()["depth"]==0

async def test_wigner_vacuum(client):
    import numpy as np
    payload = {"circuit":{"n_modes":1,"ops":[],"measurements":[]},"mode":0,"n_points":60,"x_range":5.0}
    r = await client.post("/phase-space/wigner", json=payload)
    assert r.status_code==200
    data = r.json()
    W = np.array(data["W"])
    dx = data["x_values"][1]-data["x_values"][0]
    dp = data["p_values"][1]-data["p_values"][0]
    assert abs(W.sum()*dx*dp-1.0)<0.05
    assert data["is_non_negative"] is True

async def test_submit_and_poll(client):
    import asyncio
    payload = {"circuit":{"n_modes":2,"ops":[
        {"name":"Squeezer","modes":[0],"params":{"r":0.5}},
        {"name":"Beamsplitter","modes":[0,1],"params":{"theta":0.785}}],
        "measurements":[{"kind":"homodyne","modes":[0],"params":{"phi":0.0}}]},
        "simulator":{"backend":"gaussian","shots":30}}
    r = await client.post("/jobs/submit", json=payload)
    assert r.status_code==202
    job_id = r.json()["job_id"]
    for _ in range(20):
        await asyncio.sleep(0.5)
        data = (await client.get(f"/jobs/{job_id}")).json()
        if data["status"] in ("completed","failed"): break
    assert data["status"]=="completed"
    assert len(data["result"]["samples"])==30
