"""Shared PG enum column factories.

The enum types themselves are created in `supabase/schema.sql`. These
factories produce `postgresql.ENUM` types with `create_type=False` so
SQLAlchemy never tries to redefine them.

Why the explicit ENUM mapping (vs `String(N)`): asyncpg refuses to coerce
VARCHAR → the PG enum at insert time. Without an explicit ENUM column
type, every INSERT triggers
`DatatypeMismatchError: column "..." is of type X but expression is of
type character varying`.
"""

from sqlalchemy.dialects.postgresql import ENUM

DetectorCategoryEnum = ENUM(
    "safety", "routine",
    name="detector_category", create_type=False,
)
DetectorSourceEnum = ENUM(
    "preset", "custom",
    name="detector_source", create_type=False,
)
DetectorShapeEnum = ENUM(
    "point_in_time", "state", "scheduled",
    name="detector_shape", create_type=False,
)
DetectorSensitivityEnum = ENUM(
    "low", "standard", "high",
    name="detector_sensitivity", create_type=False,
)
ReviewChoiceEnum = ENUM(
    "correct", "wrong", "unsure",
    name="review_choice", create_type=False,
)
PatientStateEnum = ENUM(
    "calm", "critical", "offline",
    name="patient_state", create_type=False,
)
RelationKindEnum = ENUM(
    "parent", "spouse", "grandparent", "sibling", "client", "other",
    name="relation_kind", create_type=False,
)
