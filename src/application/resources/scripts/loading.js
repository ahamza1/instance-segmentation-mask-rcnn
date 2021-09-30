window.addEventListener("DOMContentLoaded", () => {
    let selectedImg = null;
    let form = document.getElementById("form-upload");
    let fileInput = document.getElementById("form-file-input");
    let formSubmitBtn = document.getElementById("form-btn");

    formSubmitBtn.addEventListener("click", () => {
        if (!selectedImg) {
            fileInput.click();
        } else {
            let preview = document.getElementById("preview");
            let resultImg = document.getElementById("result_img");
            let resultSidebar = document.getElementById("result-sidebar");
            let info = document.getElementById("info-container");
            let loader = document.getElementById("load-container");

            if (resultImg) {
                resultImg.classList.add("hidden");
                resultSidebar.classList.add("hidden");
                preview.classList.remove("col-md-10");
                preview.classList.add("col-md-12");
            }

            if (info) {
                info.classList.add("hidden");
            }

            loader.classList.remove("hidden");
            form.submit();
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            selectedImg = true;
            formSubmitBtn.value = "Upload File";
        }
    });
});
