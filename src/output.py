import matplotlib
import matplotlib.pyplot as plt
import pickle


class data_exporter():

    def __init__(self, dataset, model, epochs, learning_rate, iid, frac=None, local_ep=None, local_bs=None, num_clusters="", appendage="",
                 model_name=None, sigma_local=None, sigma_global=None, sigma_intermediate=None):
        self.dataset = dataset
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.iid = iid
        self.frac = frac
        self.local_ep = local_ep
        self.local_bs = local_bs
        self.num_clusters = num_clusters
        self.appendage = appendage
        self.model_name = model_name

        self.sigma_local = sigma_local
        self.sigma_global = sigma_global
        self.sigma_intermediate = sigma_intermediate

        if not self.model_name:
            self.model_name = '{}FL{}'.format('H' if self.num_clusters != "" else "", self.num_clusters)

        self.SAVE_PATH = '../save'
        self.OBJ_PATH = self.SAVE_PATH+'/objects'
        # self.OBJ_16_PATH = '../save/objects_fp16'
        self.PLOT_PATH = self.SAVE_PATH


    def _get_file_name(self):
        model_params = 'lr[{}]_iid[{}]'.format(self.learning_rate, self.iid)
        if self.frac:
            model_params = '{}_C[{}]_E[{}]_B[{}]'.format(model_params, self.frac, self.local_ep, self.local_bs)

        if self.sigma_local or self.sigma_global or self.sigma_intermediate:
            model_params = '{}_sigma'.format(model_params)
        if self.sigma_local != None:
            model_params = '{}_L[{}]'.format(model_params, self.sigma_local)
        if self.sigma_global != None:
            model_params = '{}_G[{}]'.format(model_params, self.sigma_global)
        if self.sigma_intermediate != None:
            model_params = '{}_I[{}]'.format(model_params, self.sigma_intermediate)

        model_base = '{}_{}_{}'.format(self.dataset, self.model, self.epochs)

        return '{}_{}_{}{}'.format(self.model_name, model_base, model_params, self.appendage)

    def _get_total_path(self, file_path, name, file_type):
        return '{}/{}.{}'.format(file_path, name, file_type)

    def get_pickle_file(self, file_type='pkl'):
        file_path = self.OBJ_PATH + self.appendage.lower()
        file_name = self._get_file_name()

        return self._get_total_path(file_path, file_name, file_type)

    def get_plot_file(self, file_type='png', appendage=""):
        file_path = self.PLOT_PATH
        file_name = self._get_file_name()
        file_name += appendage

        return self._get_total_path(file_path, file_name, file_type)


    def dump_file(self, data: list):
        file_name = self.get_pickle_file()
        print("Saving data at: ", file_name)
        self._dump_file(file_name, data)

    def _dump_file(self, file_name, data: list):
        # Saving the objects loss and accuracy:
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)


    def plot_all(self, loss, accuracy, train=False, x_label='Communication Rounds (Epochs)'):
        matplotlib.use('Agg')

        # Plot Loss curve
        self.plot(loss, appendage='_loss', train=train, x_label=x_label)

        # Plot Average Accuracy vs Communication rounds
        self.plot(accuracy, appendage='_acc', train=train, x_label=x_label)

    def plot(self, data, appendage, train=False, x_label='Communication Rounds (Epochs)'):

        if train:
            phase = 'Training'
            appendage = '_train'+appendage
        else:
            phase = 'Testing'
            appendage = '_test'+appendage

        if 'loss' in appendage:
            color='r'
            data_type = 'Loss'
        elif 'acc' in appendage:
            color='k'
            data_type = 'Average Accuracy'

        y_label = f'{phase} {data_type}'

        plt.figure()
        plt.title(f'{y_label} vs {x_label}')
        plt.plot(range(len(data)), data, color=color)
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        file_path = self.get_plot_file(appendage=appendage)
        print("Saving plot at: ", file_path)

        plt.savefig(file_path)
