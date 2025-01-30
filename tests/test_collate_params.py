import unittest
from eacy import collate

class TestUtils(unittest.TestCase):
    def test_collate(self):
        telescope_name = "EAC1"
        instrument_name = "CI"
        detector_name = "IMAGER"
        output_format = "pickle"

    params_all = collate(telescope_name, instrument_name, detector_name, output_format)
        self.assertEqual(collate(telescope_name), "Hello, World!")

if __name__ == "__main__":
    unittest.main()
