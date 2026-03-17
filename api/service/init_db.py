from .entities import User
from .users import register_user

from .session import SessionLocal


def initialize_db():
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == "Guest").first()
        if user is None:
            register_user(
                username="Guest",
                password="GuestPassword",
            )
            print("Successfully initialized the database.")
        else:
            print("Database was already correctly initialized.")
    except Exception as e:
        print(f"init_db: ERROR {str(e)}")
    finally:
        db.close()