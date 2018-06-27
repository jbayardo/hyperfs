from __future__ import print_function
from __future__ import with_statement

import errno
import logging
import os
import sys
from collections import defaultdict, namedtuple

import fuse
import pandas as pd
import yaml
from fuse import FUSE, FuseOSError, Operations
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class Cube(object):
    _none_placeholder = u'None'

    # Internal column names inside the pandas dataframe follow a naming convention: they start with _internal_prefix.
    # thus, _internal_prefix has got to be something that can't possibly be present in any string that we read or use as
    # parameter. \0 works nicely. It is kept parameterized for clarity, but should not be changed.
    _internal_prefix = '\0'

    def __init__(self, root, parameters, separator):
        self._root = root
        self._parameters_file_name = parameters
        self._separator = separator

        self._cube = self._compute_cube(self._root, self._parameters_file_name)
        self._parameters = list(set(self._cube.columns).intersection(set(self._cube.index.names)))

    def refresh(self):
        return Cube(self.root, self.parameters_file_name, self.separator)

    def __len__(self):
        return len(self._cube)

    @property
    def root(self):
        return self._root

    @property
    def parameters_file_name(self):
        return self._parameters_file_name

    @property
    def separator(self):
        return self._separator

    @property
    def parameters(self):
        return self._parameters

    @staticmethod
    def is_internal(name):
        return name.startswith(Cube._internal_prefix)

    @staticmethod
    def internal(name):
        return Cube._internal_prefix + name

    @staticmethod
    def _compute_cube(root, parameters_file_name):
        models = []
        for (dirpath, dirnames, filenames) in os.walk(root):
            if parameters_file_name not in filenames:
                # Skip paths that don't have a parameters file.
                continue

            # It has a parameter file, index it.
            models.append(os.path.join(dirpath, parameters_file_name))

        # Load parameters for all model files, and put them into a DataFrame,
        # which will act as a cube.
        params = map(Cube._read_parameters, models)
        params = pd.DataFrame(params)
        params.fillna(Cube._none_placeholder, inplace=True)

        # Set all parameter columns as index. This serves a double purpose: to
        # check that they are indeed unique, and to do fast lookups when we
        # need to create the directory structure.
        index_columns = [column_name for column_name in params.columns if not Cube.is_internal(column_name)]

        if index_columns:
            try:
                params.set_index(index_columns, drop=False, inplace=True, verify_integrity=True)
            except ValueError:
                similar = [set(g[Cube.internal('file_path')]) for _, g in params.groupby(index_columns) if len(g) > 1]
                raise ValueError('Index is not unique. '
                                 'One or folders have exactly equal {parameters_file_name}. '
                                 'The groups are: {similar}'.format(
                    parameters_file_name=parameters_file_name,
                    similar=similar
                ))

        return params

    @staticmethod
    def _read_parameters(file_name):
        with open(file_name, 'r') as handle:
            params = yaml.load(handle)
            if not params:
                params = {}

            # Ensure all parameters can -and are- casted to string
            # TODO: better heuristics for string conversion and value rejection.
            for key in params:
                value = params.pop(key)
                params[str(key)] = str(value)

            # Add internal metadata used for querying
            file_name = os.path.abspath(file_name)
            params[Cube.internal('file_path')] = file_name
            params[Cube.internal('link_target')] = os.path.split(file_name)[0]
            return params

    def path_search(self, path):
        conversion = self._path_to_query(path, self._parameters, self._separator)
        return conversion, self._index_search(conversion.query, self._cube)

    Conversion = namedtuple('QueryResult', ['seen', 'next', 'query'])

    @staticmethod
    def _path_to_query(path, columns, separator):
        seen = []
        next = path.split(os.sep)

        query = defaultdict(lambda _: Cube._none_placeholder)
        while len(next) > 0:
            part = next[0]

            if part != '':
                try:
                    association = part.split(separator, 1)
                    key = association[0]
                    value = association[1]

                    if key not in columns:
                        raise IndexError()

                    query[key] = value
                except IndexError:
                    # If for any reason we couldn't fetch the key, it means that we no longer have values worth looking
                    # into
                    break

            seen.append(next.pop(0))

        return Cube.Conversion(
            seen=os.sep.join(seen),
            next=os.sep.join(next),
            query=query
        )

    @staticmethod
    def _index_search(query, frame):
        data = frame
        for key, value in query.items():
            data = data[data[key].isin([value])]
        return data


class FileSystem(Operations):
    # TODO: add negative queries
    def __init__(self, root, log, parameters, separator):
        super(FileSystem, self).__init__()

        self._log = log
        self._cube = Cube(root, parameters, separator)
        self._stats_log()

    @property
    def root(self):
        return self._cube.root

    @property
    def parameters_file_name(self):
        return self._cube.parameters_file_name

    @property
    def separator(self):
        return self._cube.separator

    def refresh(self):
        cube = self._cube.refresh()
        self._cube = cube
        self._stats_log()

    def readdir(self, path, fh):
        self._log.debug('readdir: %s', path)
        conversion, result = self._cube.path_search(path)

        # Always present!
        dirents = ['.', '..']

        for key in result.columns:
            # Key is an internal usage datum
            if key not in self._cube.parameters:
                continue

            # Key has already been filtered for
            if key in conversion.query:
                continue

            # We should have one directory for each possible value
            for value in result[key].unique():
                dirents.append('{key}{separator}{value}'.format(key=key, separator=self.separator, value=value))

        # Tell FUSE what the files are
        for r in dirents:
            yield r

    # Taken from 'man 2 stat'
    _S_IFLNK = 0o0120000

    def getattr(self, path, fh=None):
        self._log.debug('getattr: %s', path)
        conversion, result = self._cube.path_search(path)

        if os.path.basename(path) in ['', '.', '..'] or len(result) > 1:
            return dict(st_mode=(fuse.S_IFDIR | 0o100), st_nlink=2)
        elif len(result) == 1:
            return dict(st_mode=(self._S_IFLNK | 0o100), st_nlink=2)

        raise FuseOSError(errno.ENOENT)

    def readlink(self, path):
        self._log.debug('readlink: %s', path)
        conversion, result = self._cube.path_search(path)

        # Symbolic links are only defined when we have a single result
        assert len(result) == 1
        return result.iloc[0][Cube.internal('link_target')]

    def _stats_log(self):
        logging.info('Cube update found %d files with parameters: %s', len(self._cube), str(self._cube.parameters))


class Watchdog(FileSystemEventHandler):
    def __init__(self, mount_point, filesystem, log):
        super(Watchdog, self).__init__()

        self._log = log
        self._filesystem = filesystem
        self._mount_point = mount_point
        self._observer = Observer()

    def run(self):
        self._log.info('Bootstrapping watchdog for mountpoint at %s', self._filesystem.root)
        self._observer.schedule(self, self._filesystem.root, recursive=True)
        self._observer.start()

        self._log.info('Mounting FUSE filesystem')
        try:
            FUSE(self._filesystem, self._mount_point, nothreads=True, foreground=True)
        except Exception as e:
            self._observer.stop()
            self._log.critical('Received exception: %s', str(e))
            self._log.critical('Watchdog exiting quietly')

        self._observer.join()

    def on_any_event(self, event):
        self._log.debug('Received event: %s', str(event))

        if not event.is_directory and os.path.basename(event.src_path) == self._filesystem.parameters_file_name:
            try:
                # TODO: perhaps stop mounting or something similar?
                self._filesystem.refresh()
                self._log.info('Filesystem has been refreshed successfully')
            except Exception as e:
                self._log.error('Exception generated by the FileSystem when attempting refresh: %s', str(e))
                self._log.warning('File system refresh failed. Will maintain operation using old filesystem')


def main(mount_point, root, parameters, separator):
    log = logging.getLogger('hyperfs')
    log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    log.addHandler(ch)

    filesystem = FileSystem(root, log, parameters, separator)
    watcher = Watchdog(mount_point, filesystem, log)
    watcher.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Mounts a virtual file system corresponding to a parameter cube as '
                                                 'defined by a set of yaml files')
    parser.add_argument('root', type=str,
                        help='path to the root')
    parser.add_argument('mount_point', type=str,
                        help='path to the desired mounting point')
    parser.add_argument('--parameters_file', type=str, default='parameters.yaml',
                        help='name of the files expected to hold the yaml configuration')
    parser.add_argument('--name_separator', type=str, default=':',
                        help='character to use when separating the parameter name from value in the virtual file '
                             'system')

    args = parser.parse_args()
    args.name_separator = args.name_separator[0]
    main(args.mount_point, args.root, args.parameters_file, args.name_separator)
