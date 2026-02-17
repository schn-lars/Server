from sqlalchemy import Column, Integer, String, ForeignKey, Text, BigInteger, Numeric
from sqlalchemy.orm import relationship
from service.session import Base

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