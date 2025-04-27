import unittest
from agi.utils.weather import get_weather_info,search_list,find_best_match
from agi.config import CACHE_DIR
import os

class TestWeather(unittest.TestCase):

    def test_search_list(self):
        """Test the invocation of the Image2Image model."""
        file_path = f'{CACHE_DIR}/search_list.json'
        self.assertTrue(os.path.exists(file_path))
        self.assertIsNotNone(search_list)
        self.assertGreater(len(search_list),0)
        print(len(search_list))
    def test_get_code(self):
        name,code,_ = find_best_match("上海")
        self.assertEqual(name,"上海")
        self.assertIsNotNone(code)
        name,code,_ = find_best_match("徐家汇")
        self.assertEqual(name,"徐家汇")
        self.assertIsNotNone(code)
        name,code,_ = find_best_match("徐家")
        self.assertEqual(name,"徐家汇")
        self.assertIsNotNone(code)
        name,code,_ = find_best_match("北京")
        self.assertEqual(name,"北京")
        self.assertIsNotNone(code)
        name,code,_ = find_best_match("上海今天的天气")
        self.assertEqual(name,"上海")
        self.assertIsNotNone(code)
    def test_get_weather(self):
        """Clean up after tests."""
        # You can include any clean-up logic here if necessary.
        d = get_weather_info("上海")
        print(d)
        self.assertIsNotNone(d)
        d = get_weather_info("香港")
        self.assertIsNotNone(d)
        print(d)
        d = get_weather_info("成都")
        print(d)
        self.assertIsNotNone(d)

if __name__ == "__main__":
    unittest.main()
