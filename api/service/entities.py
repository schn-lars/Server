from sqlalchemy import Column, Integer, ForeignKey, Text, BigInteger, Numeric, String, Float, Boolean
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
from decimal import Decimal
import uuid
from service.session import Base
from typing import Optional

class Birds(Base):
    __tablename__ = "birds"

    id = Column(Integer, primary_key=True, index=True)
    trivial_name = Column(Text, nullable=True)
    species_name = Column(Text, nullable=True)
    city = Column(Text, nullable=True)
    canton = Column(Text, nullable=True)

    x = Column(Numeric(18, 15), nullable=True)
    y = Column(Numeric(18, 15), nullable=True)

    counter = Column(Integer, nullable=True)
    year_number = Column(Integer, nullable=True)


class Label(Base):
    __tablename__ = "labels"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(Text, nullable=False, unique=True)

    synonyms = relationship(
        "Synonym",
        back_populates="label",
        cascade="all, delete-orphan"
    )

class BirdsMaterialized(Base):
    __tablename__ = "birds_materialized"

    species_name = Column(Text, primary_key=True)
    canton = Column(Text, primary_key=True)
    year_number = Column(Integer, primary_key=True)

    total_count = Column(BigInteger)

class Synonym(Base):
    __tablename__ = "synonyms"

    label_id = Column(
        Integer,
        ForeignKey("labels.id", ondelete="CASCADE"),
        primary_key=True
    )

    synonym = Column(Text, primary_key=True)

    label = relationship("Label", back_populates="synonyms")

class Adress(Base):
    __tablename__ = "adresses"

    id = Column(Integer, primary_key=True)
    street = Column(Text)
    number = Column(Text)
    zip = Column(Integer)
    zip_label = Column(Text)
    name = Column(Text)
    canton = Column(Text)
    coord_x = Column(Numeric(18, 15), nullable=True)
    coord_y = Column(Numeric(18, 15), nullable=True)
    normalized_street = Column(Text)

class SharedInformation(Base):
    __tablename__ = "shared_information"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4())
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user.id", ondelete="CASCADE"))
    
    object: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    coord_x = Column(Numeric(18, 15), nullable=True)
    coord_y = Column(Numeric(18, 15), nullable=True)

class RetrievedInformation(Base):
    __tablename__ = "retrieved_information"

    id = Column(UUID(as_uuid=True), ForeignKey("shared_information.id", ondelete="CASCADE"), primary_key=True, index=True)
    json = Column(Text, nullable=False)

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4())
    username: Mapped[str] = mapped_column(String, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)

class ShareMapping(Base):
    __tablename__ = "share_mappings"
    
    shared_to_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, index=True, default=uuid.uuid4())
    info_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("shared_information.id", ondelete="CASCADE"), primary_key=True, index=True, default=uuid.uuid4())