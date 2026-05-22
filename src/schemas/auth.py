"""Auth-related domain types.

`SupabasePrincipal` is the typed identity extracted from a verified
Supabase JWT. Crosses the boundary between the JWT verifier and any
caller that needs to know "who is this request from."
"""

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

Email = Annotated[str, Field(min_length=3, max_length=255, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]


class SupabasePrincipal(BaseModel):
    """Identity extracted from a verified Supabase JWT.

    `id` equals Supabase's `auth.users.id` (the JWT `sub` claim is a
    UUID string; Pydantic coerces it to UUID at validation). Used as
    the PK of the nemara `caregivers` row.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID
    email: Email
