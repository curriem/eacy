import unittest
from eacy import collate
import numpy as np

class TestUtils(unittest.TestCase):
    def test_collate(self):
        import pickle
        params_to_compare = pickle.load(open("eacy_params.pk", "rb"))
        telescope_name = "EAC1"
        instrument_name = "CI"
        detector_name = "IMAGER"
        output_format = "pickle"

        dict1 = collate(telescope_name, instrument_name, detector_name, output_format, save=False)
        dict2 = params_to_compare
        are_equal = True
        for k in dict1.keys() | dict2.keys():  # Union of keys
            if k not in dict1 or k not in dict2:
                are_equal = False
                break
            v1, v2 = dict1[k], dict2[k]
            if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                if not np.array_equal(v1, v2):  # Use np.array_equal for array comparison
                    are_equal = False
                    break
            elif v1 != v2:  # Regular comparison for non-array values
                are_equal = False
                break
        self.assertTrue(are_equal, "The dictionaries are not equal.")

if __name__ == "__main__":
    unittest.main()
