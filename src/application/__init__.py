from flask import Flask
from flask_bootstrap import Bootstrap

from application.config.config import app_config
from application.mrcnn.model import register_model
from application.mrcnn.routes import register_routes


def create_app(config_name="dev"):
    config = app_config[config_name]

    app = Flask(
        __name__,
        static_folder=config.RESOURCES_DIR
    )

    app.config.from_object(config)

    register_routes(app, register_model(app))
    register_plugins(app)

    return app


def register_plugins(app):
    Bootstrap(app)
