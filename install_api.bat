cd models\research
..\..\protoc\protoc.exe models\protos\*.proto --python_out=.
pip install .