import unittest
from eacy import collate

class TestUtils(unittest.TestCase):
    def test_collate(self):
        import pickle
        params_to_compare = pickle.load(open("eacy_params.pk", "rb"))
        telescope_name = "EAC1"
        instrument_name = "CI"
        detector_name = "IMAGER"
        output_format = "pickle"

        self.assertEqual(collate(telescope_name, instrument_name, detector_name, output_format, save=False), params_to_compare)

if __name__ == "__main__":
    unittest.main()
