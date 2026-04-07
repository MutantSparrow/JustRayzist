This directory is intentionally tracked so release packaging can include the
expected `models/upscaler/` tree.

The actual upscaler checkpoint file is local runtime data and is fetched by the
setup and asset scripts. Do not commit the `.pth` model binary.
