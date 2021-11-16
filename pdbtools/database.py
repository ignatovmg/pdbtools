import os
import re

from collections import OrderedDict


class Database(object):
    _pattern_max_length = 256
    _reading_stride = 8 * 2**20  # Chunks of 8Mb

    def __init__(self,
                 path,
                 index_path=None,
                 sep=None,
                 file_list=None,
                 key_list=None,
                 basename=False,
                 encoding='utf-8'):
        """
        Class for working with a simple database. The database consists of multiple
        data points concatenated into a single file, passed with 'path'. Each
        datapoint can be accessed by its name in self._index, which keeps
        the offset of the start and the end in bytes. Index is created during the
        instantiation via 3 possible ways:

        1) Index file, which contains 3 columns: key, start_offset, end_offset (in bytes)
           Columns are whitespace separated

        2) Using separators specified in 'sep'. 'sep' is a tuple with regex patterns of
           start and end. E.g. sep=('MODEL', 'ENDMDL'). If there is only starting or
           only ending pattern you can specify sep=('MODEL', None). You can also capture
           the datapoint's name if you add a group into the regex expression, like this:
           sep=('MODEL([0-9]+)', None). In this case match.group(1) will become the key
           of this datapoint in the Index dictionary.

        3) List of separate files. In this case the contents of the files will be
           concatenated in 'path'. By default the file names will become the keys in the index.
           'basename'=True will use basenames as keys instead of the full paths. If you specify
           'key_list', it will be used as keys instead of the filenames.

        NOTE: All data files are opened in binary mode. If using a multiline pattern on
              the files created on Windows, don't forget to use \r\n instead of \n.

        Usage:

        >>> db = Database('database.txt', sep=['MODEL\s+([0-9]+)', 'ENDMDL'])
        >>> print(db._index)

        OrderedDict([('0', (0, 5020)),
                     ('1', (5020, 10040)),
                     ...

        >>> # Access the datapoints by key
        >>> db['1']

        >>> # You can record the offsets
        >>> db.write_index_file('index.csv')

        >>> # To read them later
        >>> db = Database('database.txt', 'index.csv')
        """

        self._path = path
        self._index = None
        self._stream = None

        self._sep = sep
        self._file_list = file_list
        self._basename = basename
        self._encoding = encoding

        if index_path is not None:
            self._index = self._read_index_file(index_path)
        elif sep is not None:
            sep_start, sep_end = sep
            self._index = self._index_from_sep(sep_start, sep_end)
        elif file_list is not None:
            self._index = self._index_from_file_list(file_list, key_list, basename)
        else:
            raise ValueError('Parameter "index_path" or "sep" or "file_list" must be specified')

    def __getitem__(self, name):
        if self._stream is None:
            raise RuntimeError('Run self.open() first')

        start, end = self._index[name]
        self._stream.seek(start)
        chunk = self._stream.read(end - start)
        return chunk

    def __len__(self):
        return len(self._index)

    def open(self):
        """
        Open th database file for reading. Call before calling __getitem__
        """
        self._stream = open(self._path, 'rb')

    def close(self):
        """
        Close the database file
        """
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def keys(self):
        """
        Get all keys
        """
        return self._index.keys()

    def _index_from_file_list(self, file_list, key_list=None, basename=False):
        """
        Create index from the file list. Concatenates them and writes the
        result into self.path. key_list can used to fill the self._index
        dictionary. Otherwise the file names will be used as keys.
        """
        start = 0
        index = []
        with open(self._path, 'wb') as o:
            for i, path in enumerate(file_list):
                with open(path, 'rb') as f:
                    data = f.read()

                o.write(data)

                if key_list is None:
                    name = path
                    if basename:
                        name = os.path.basename(path)
                else:
                    name = str(key_list[i])

                cur_pos = o.tell()
                index.append((name, (start, cur_pos)))
                start = cur_pos

        _dict = OrderedDict(index)
        if len(_dict) != len(file_list):
            raise RuntimeError('Found duplicated keys, when creating from file list')

        return _dict

    def split(self, dst, file_list=None, files_txt='files.list'):
        """
        Split the database into separate files and put them in dst/
        and call them after the keys in the index. If file_list is
        provided the files are called correspondingly
        """
        if self._stream is None:
            raise RuntimeError('Run self.open() first')

        self._file_list = []
        for i, k in enumerate(self._index.keys()):
            if file_list is None:
                dst_full = os.path.join(dst, k)
            else:
                dst_full = os.path.join(dst, file_list[i])

            with open(dst_full, 'wb') as f:
                f.write(self.__getitem__(k))

            self._file_list.append(dst_full)

        with open(os.path.join(dst, files_txt), 'w') as f:
            f.write('\n'.join(self._file_list) + '\n')

    def _index_from_sep(self, sep_start=None, sep_end=None):
        """
        Create self._index using regex separators and searching
        for the matches in self.path by reading it in chunks
        """
        if sep_start is None and sep_end is None:
            raise ValueError("Both arguments can't be None")

        indices = []
        starts = []
        ends = []
        offset = 0
        counter = 0
        stride = self._reading_stride

        if sep_start is not None:
            sep_start = re.compile(sep_start, flags=re.MULTILINE)

        if sep_end is not None:
            sep_end = re.compile(sep_end, flags=re.MULTILINE)

        with open(self._path, 'rb') as f:
            while True:
                chunk = f.read(stride).decode(self._encoding)

                if sep_start is not None:
                    # find positions of starting patterns
                    for m in sep_start.finditer(chunk):
                        pos = offset + m.start()

                        # since we shift back in the end of the loop
                        # make sure we haven't recorded this position before
                        if pos in starts[-self._pattern_max_length:]:
                            continue
                        starts.append(pos)

                        # find name, if it was specified in the pattern
                        groups = m.groups()
                        if len(groups) == 0:
                            index = str(counter)
                        else:
                            index = groups[0]
                        indices.append(index)
                        counter += 1

                if sep_end is not None:
                    # find positions of ending patterns
                    for m in sep_end.finditer(chunk):
                        pos = offset + m.end()

                        # same as above, make sure we haven't recorded
                        # this ending already
                        if pos in ends[-self._pattern_max_length:]:
                            continue
                        ends.append(pos)

                        # find name if sep_start is not specified
                        if sep_start is None:
                            groups = m.groups()
                            if len(groups) == 0:
                                index = str(counter)
                            else:
                                index = groups[0]
                            indices.append(index)
                            counter += 1

                # check if we reached the end
                if not f.read(1):
                    break

                # move back by maximum pattern length to make sure we
                # haven't missed the pattern, if it was in between the chunks
                offset = f.tell() - 1 - self._pattern_max_length
                f.seek(offset)

            if sep_end is None:
                ends = starts[1:] + [f.tell()]

            if sep_start is None:
                starts = [0] + ends[:-1]

        if len(starts) != len(ends):
            msg = 'Number of starts is different from the number of ends (%i, %i)' % (len(starts), len(ends))
            raise RuntimeError(msg)

        _dict = [(k, (s, e)) for k, s, e in zip(indices, starts, ends)]
        for _, (s, e) in _dict:
            if s > e:
                raise RuntimeError("Start can't come after end (%i > %i)" % (s, e))

        _dict = OrderedDict(_dict)
        if len(_dict) != len(indices):
            raise RuntimeError('Found duplicated keys in index dictionary')

        return _dict

    def _read_index_file(self, path):
        """
        Read self._index from file
        """
        with open(path, 'r') as f:
            _dict = []
            for l in f:
                key, start_byte, end_byte = l.split()
                _dict.append((key, (int(start_byte), int(end_byte))))

        return OrderedDict(_dict)

    def write_index_file(self, path):
        """
        Write index to file, so you can read it later
        """
        with open(path, 'w') as f:
            for k, (start, end) in self._index.items():
                f.write('%s %i %i\n' % (k, start, end))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Database")
    parser.add_argument("database_path", help="Database file")

    g1 = parser.add_argument_group('Index file')
    g1.add_argument("-i", "--index_file", default=None, help="Index file with binary offsets")

    g2 = parser.add_argument_group('Separators', 'Use regex separators to parse the database file and create index')
    g2.add_argument("--sep_start", default=None, help="Starting separator")
    g2.add_argument("--sep_end", default=None, help="Ending separator")

    g3 = parser.add_argument_group('List of files', 'Creat the database file and index from separated data files')
    g3.add_argument("--file_list", default=None, help="List of files to create a database from")
    g3.add_argument("--key_list", default=None, help="List of keys, which replaces the file names in the index file")
    g3.add_argument("--basename", action="store_true", help="Keep only the basenames as keys")

    g4 = parser.add_argument_group('Actions')
    g4.add_argument("-w", "--write_index_file", default=None, help="Write index file")
    g4.add_argument("-k", "--get_key", default=None, help="Get the key from the database")
    g4.add_argument("-s", "--split", default=None, help="Split into the directory (must exist)")

    args = parser.parse_args()

    if args.index_file is None and args.sep_start is None and args.sep_end is None and args.file_list is None:
        raise RuntimeError('No index file to read the database')

    file_list = None
    if args.file_list is not None:
        with open(args.file_list, 'r') as f:
            file_list = [x.strip() for x in f]

    key_list = None
    if args.key_list is not None:
        with open(args.key_list, 'r') as f:
            key_list = [x.strip() for x in f]

    sep = [args.sep_start, args.sep_end]
    if sep[0] is None and sep[1] is None:
        sep = None

    db = Database(args.database_path,
                  index_path=args.index_file,
                  sep=sep,
                  file_list=file_list,
                  key_list=key_list,
                  basename=args.basename)

    if args.write_index_file:
        db.write_index_file(args.write_index_file)

    if args.get_key:
        db.open()
        print(db[args.get_key])
        db.close()

    if args.split:
        db.open()
        db.split(args.split, file_list)
        db.close()


#if __name__ == '__main__':
#    Database('all-sdf.sdf', sep=())