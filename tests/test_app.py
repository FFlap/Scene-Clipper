import importlib.util


def test_app_py_imports_without_pythonpath():
    spec = importlib.util.spec_from_file_location("scene_clipper_app", "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.main)
