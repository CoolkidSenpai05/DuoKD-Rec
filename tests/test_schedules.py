from src.schedules import linear_warmup


def test_linear_warmup():
    vals = [linear_warmup(e, 0.3, 5, 5) for e in range(12)]
    assert vals[:6] == [0.0] * 6
    assert abs(vals[7] - 0.12) < 1e-9
    assert vals[10] == vals[11] == 0.3
    assert linear_warmup(3, 1.0, 3, 0) == 1.0
