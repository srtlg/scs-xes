from os import SEEK_SET
from tempfile import NamedTemporaryFile
from scs_xes.test_asc_file import TEST_ASC, TestAscFile
from scs_xes.c_asc_file import cAscFile


class TestCAscFile(TestAscFile):
    def setUp(self):
        self.file = NamedTemporaryFile()
        self.file.write(TEST_ASC)
        self.file.seek(0, SEEK_SET)
        self.obj = cAscFile(self.file.name, num_rows=2)

    def tearDown(self):
        pass