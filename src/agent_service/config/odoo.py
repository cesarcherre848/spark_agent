import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(".env.dev")


@dataclass
class OdooSettings:
    url: str = os.getenv("ODOO_URL", "http://134.199.209.31:8069")
    db: str = os.getenv("ODOO_DB", "spark_erp_db_dev")
    username: str = os.getenv("ODOO_USERNAME", "robot_user")
    password: str = os.getenv("ODOO_PASSWORD", "robot_user1234")


def get_odoo_settings() -> OdooSettings:
    return OdooSettings(
        url=os.getenv("ODOO_URL", "http://134.199.209.31:8069"),
        db=os.getenv("ODOO_DB", "spark_erp_db_dev"),
        username=os.getenv("ODOO_USERNAME", "robot_user"),
        password=os.getenv("ODOO_PASSWORD", "robot_user1234"),
    )
