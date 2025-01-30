import unittest
from eacy import collate

class TestUtils(unittest.TestCase):
    def test_collate(self):
        self.assertEqual(greet("World"), "Hello, World!")

if __name__ == "__main__":
    unittest.main()
