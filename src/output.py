import matplotlib
import matplotlib.pyplot as plt
import pickle


class data_exporter():

    def __init__(self, dataset, model, epochs, learning_rate, iid, frac=None, local_ep=None, local_bs=None, num_clusters="", appendage="", model_name=None):
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


    def plot_all(self, loss, accuracy):
        matplotlib.use('Agg')

        # Plot Loss curve
        self.plot_loss(loss)

        # Plot Average Accuracy vs Communication rounds
        self.plot_accuracy(accuracy)

    def plot_loss(self, loss):
        plt.figure()
        plt.title('Training Loss vs Communication rounds')
        plt.plot(range(len(loss)), loss, color='r')
        plt.ylabel('Training loss')
        plt.xlabel('Communication Rounds')

        file_path = self.get_plot_file(appendage="_loss")
        print("Saving plot at: ", file_path)

        plt.savefig(file_path)

    def plot_accuracy(self, accuracy):
        plt.figure()
        plt.title('Average Accuracy vs Communication rounds')
        plt.plot(range(len(accuracy)), accuracy, color='k')
        plt.ylabel('Average Accuracy')
        plt.xlabel('Communication Rounds')

        file_path = self.get_plot_file(appendage="_acc")
        print("Saving plot at: ", file_path)

        plt.savefig(file_path)

    def plot_small(self, epoch_loss):
        plt.figure()
        plt.plot(range(len(epoch_loss)), epoch_loss)
        plt.xlabel('epochs')
        plt.ylabel('Train loss')

        file_path = self.get_plot_file(appendage="_loss")
        print("Saving plot at: ", file_path)

        plt.savefig(file_path)
