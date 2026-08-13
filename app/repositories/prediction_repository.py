"""
Prediction database repository.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.models import Prediction


class PredictionRepository:
    """Database operations for predictions."""

    @staticmethod
    def create(
        db: Session,
        prediction: Prediction,
    ) -> Prediction:
        """Create prediction record."""

        db.add(prediction)

        db.commit()

        db.refresh(prediction)

        return prediction

    @staticmethod
    def get_by_id(
        db: Session,
        prediction_id: int,
        user_id: int,
    ) -> Prediction | None:
        """Get prediction by ID."""

        statement = select(
            Prediction
        ).where(
            Prediction.id == prediction_id,
            Prediction.user_id == user_id,
        )

        return db.scalar(statement)

    @staticmethod
    def get_history(
        db: Session,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Prediction]:
        """Get prediction history."""

        statement = (
            select(Prediction)
            .where(
             Prediction.user_id == user_id
            )
            .order_by(
                Prediction.created_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return list(
            db.scalars(statement)
        )

    @staticmethod
    def delete(
        db: Session,
        prediction: Prediction,
    ) -> None:
        """Delete prediction."""

        db.delete(prediction)

        db.commit()


    @staticmethod
    def count(
        db: Session,
        user_id: int,
    ) -> int:
        """Count prediction records."""

        statement = select(
            func.count(Prediction.id)
        ).where(
            Prediction.user_id == user_id
        )

        return int(
            db.scalar(statement) or 0
        )
