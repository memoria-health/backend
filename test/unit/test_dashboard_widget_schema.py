"""dashboard_widget: the compiled per-detector widget spec (Home tab).

Contract under test:
  - the ORM row carries a nullable JSONB `dashboard_widget`
  - DetectorResponse exposes it, defaulting to None, so the iOS client
    receives specs through the existing detectors flow (no new endpoint)
"""

from sqlalchemy.dialects.postgresql import JSONB

from src.db.detector import Detector
from src.schemas.detector import DetectorResponse


def test_detector_row_has_dashboard_widget_jsonb() -> None:
    col = Detector.__table__.columns["dashboard_widget"]
    assert isinstance(col.type, JSONB)
    assert col.nullable


def test_detector_response_exposes_dashboard_widget_default_none() -> None:
    field = DetectorResponse.model_fields["dashboard_widget"]
    assert field.default is None
