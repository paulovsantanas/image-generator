import importlib
import unittest


class OptionalDependenciesTest(unittest.TestCase):
    def test_base_package_import_does_not_load_provider_dependencies(self):
        image_generator = importlib.import_module("image_generator")

        self.assertTrue(hasattr(image_generator, "Provider"))
        self.assertTrue(hasattr(image_generator, "get_provider"))


if __name__ == "__main__":
    unittest.main()
