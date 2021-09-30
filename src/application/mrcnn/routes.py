from flask import request, render_template, current_app

from application.utils import get_tmp_file_name, get_tmp_file_path


def register_routes(app, model):
    @app.route("/", methods=["POST"])
    def inference():
        file = request.files["file"]

        assert file and file.filename.endswith(
            current_app.config["ALLOWED_EXTENSIONS"]
        )

        image_name = get_tmp_file_name(file.filename)
        image_full_path = get_tmp_file_path(current_app.config["RESOURCES_DIR"], image_name)

        file.save(image_full_path)

        result = model.detect(image_full_path)

        return render_template(
            "index.html",
            result={
                "image": image_name,
                "objects": result["objects"]
            }
        )

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")
