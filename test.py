from unittest import TestCase
import pandas as pd
import os
from cleaning_function import *
# from file_operations import write_to_file

class Test(TestCase):
    def test_file(self):
        df = pd.read_csv('abx_MAY_DUMMY.csv')
        output = cleaning(df)
        output.to_csv('may_clean_test.csv', index = False, quoting = 1)
        self.assertTrue(os.path.exists('may_clean_test.csv'))
        
        with open('may_clean_test.csv','r') as test_file, open('may_clean.csv','r') as real_file:
            test_content = test_file.read()
            real_content = real_file.read()
        os.remove('may_clean_test.csv')    
        return self.assertEqual(test_content,real_content)
        
        
# if __name__ == '__main__':
#     test = Test()
#     test.test_file()
#     os.remove('may_clean_test.csv')