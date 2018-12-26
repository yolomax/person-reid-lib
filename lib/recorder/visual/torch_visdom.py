import json
import numpy as np
from lib.utils.util import check_path, copy_file_to, remove_file


class VisionBuffer(object):
    def __init__(self, name, path):
        self._file_name = name
        self._folder_path = path
        self._file_path = self._folder_path / (self._file_name + '.json')

        self._buffer = {}
        self._buffer['reload'] = {}
        self._buffer['jsons'] = {}
        self._pane_id = 0
        self._content_id = 0

        self._pane_idx = {}
        self._line_idx = {}
        self._line_num = {}

        self._flush_epoch = 5
        self._flush_id = 0

    def _win_register(self, win_name):
        pane_name = 'pane_' + str(self._pane_id)
        contentID = str(self._content_id)

        pane = {}
        pane['title'] = win_name
        pane['i'] = self._pane_id
        pane['contentID'] = contentID
        pane['content'] = {'data': []}
        pane['command'] = 'pane'
        pane['type'] = 'plot'
        pane['id'] = pane_name
        self._buffer['jsons'][pane_name] = pane

        self._pane_idx[win_name] = pane_name
        self._line_idx[win_name] = {}
        self._line_num[win_name] = 0
        self._pane_id += 1
        self._content_id += 1

    def _line_register(self, win_name, line_name):
        if win_name not in self._pane_idx:
            self._win_register(win_name)
        if line_name not in self._line_idx[win_name]:
            self._line_idx[win_name][line_name] = self._line_num[win_name]
            self._line_num[win_name] += 1
        self._buffer['jsons'][self._pane_idx[win_name]]['content']['data'].append(
            {
                'name': line_name,
                "marker": {
                    "symbol": "dot",
                    "line": {
                        "color": "#000000",
                        "width": 0.5},
                    "size": 3},
                "mode": "lines+markers",
                "y": [],
                "x": [],
                "type": "scatter"
            }
        )

    def plot(self, win_name, line_name, x, y):
        if isinstance(x, np.ndarray):
            x = x.tolist()

        if isinstance(y,np.ndarray):
            y = y.tolist()

        assert isinstance(x,list) and isinstance(y, list)
        assert len(x) == len(y)
        if win_name not in self._pane_idx:
            self._win_register(win_name)
        if line_name not in self._line_idx[win_name]:
            self._line_register(win_name, line_name)
        pane_name = self._pane_idx[win_name]
        line_idx = self._line_idx[win_name][line_name]
        self._buffer['jsons'][pane_name]['content']['data'][line_idx]['x'].extend(x)
        self._buffer['jsons'][pane_name]['content']['data'][line_idx]['y'].extend(y)

        self._flush()

    def _save(self):
        with open(self._file_path, 'w') as file_object:
            json.dump(self._buffer, file_object)

    def _flush(self):
        self._flush_id = (self._flush_id + 1) % self._flush_epoch
        if self._flush_id == 0:
            self._save()

    def finish(self):
        self._save()


class VisionServer(object):
    def __init__(self, env_name, file_path, device):
        self._file_path = file_path
        self.device = device
        self._env_name = env_name
        self._target_dir = check_path(self._file_path, True)
        self._env_path = check_path(self.device['web_env_dir'], True)
        self._env_dir = self._env_path / (self._env_name + '.json')

        self._win_box = {}

        from visdom import Visdom

        self._viz = Visdom(port=self.device['web_port'], env=self._env_name, server=self.device['web_host'],
                           raise_exceptions=True)

        self._mode = None
        self._flush_epoch = 5
        self._flush_id = 0

    def _win_registor(self, win_name, line_name, X, Y):
        self._win_box[win_name] = self._viz.line(
            X=X,
            Y=Y,
            opts=dict(
                legend=[line_name],
                markers=True,
                title=win_name,
                markersize=3,
                showlegend=True
            )
        )

    def plot(self, win_name, line_name, X, Y):
        if win_name in self._win_box:
            self._viz.line(
                X=X,
                Y=Y,
                win=self._win_box[win_name],
                update='append',
                name=line_name,
                opts=dict(showlegend=True)
            )
            # self._viz.updateTrace(
            #     X=X,
            #     Y=Y,
            #     win=self._win_box[win_name],
            #     name=line_name
            # )
        else:
            self._win_registor(win_name, line_name, X, Y)

        self._flush()

    def _flush(self):
        self._flush_id = (self._flush_id + 1) % self._flush_epoch
        if self._flush_id == 0:
            self._save()

    def _save(self):
        self._viz.save([self._env_name])
        try:
            copy_file_to(self._env_dir, self._target_dir)
            remove_file(self._env_dir)
        except FileExistsError:
            pass

    def finish(self):
        self._save()


class TorchVisual(object):
    def __init__(self, folder_path, tag, device, logger):
        self._folder_path = folder_path
        self.logger = logger
        self.device = device
        self._father_name = self._folder_path.parts[-1]
        self._env_name = self.device['name'] + '_' + self._father_name + '_' + tag

        self._target_dir = check_path(self._folder_path / 'output/log', True)

        try:
            self._viz = VisionServer(self._env_name, self._target_dir, self.device)

        except ImportError as e:
            self.logger.error(e)
            self._viz = VisionBuffer(self._env_name, self._target_dir)
            self.logger.info('Visdom is not installed. The visual data is stored locally.')

        except (ConnectionError, ConnectionRefusedError) as e:
            self.logger.error(e)
            self._viz = VisionBuffer(self._env_name, self._target_dir)
            self.logger.info('Visdom is not installed, but no server connection. The visual data is stored locally.')
        else:
            self.logger.info('Visdom is installed. Using vsidom to display visual data on the web page.')

        self._mode = None

    def set_mode(self, mode):
        self._mode = mode

    def _legend_wrap(self, line_name):
        if self._mode is None:
            return line_name
        else:
            return line_name + '_' + self._mode

    def plot(self, win_name, line_name, X, Y):
        if not isinstance(X, (np.ndarray, list, tuple)):
            X = np.asarray([X])
        if not isinstance(Y, (np.ndarray, list, tuple)):
            Y = np.asarray([Y])
        try:
            self._viz.plot(win_name, self._legend_wrap(line_name), X, Y)
        except (ConnectionError, ConnectionRefusedError) as e:
            pass

    def finish(self):
        self._viz.finish()