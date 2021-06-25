# Import Libraries
import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import math
import sys
import time
import pdb

# Optimizer
class Optimizer:
    def __init__(self, neural_network, max_iterations):
        self.nn = neural_network
        self.max_iterations = max_iterations

    def random_search_gaussian_distribution(self, model_parameters, sigma):
        iteration = 0
        x_best = model_parameters
        iteration += 1
        x_best_fvalue = self.nn.error_function(x_best)
        mu = x_best

        print("Iteration=%d, Error=%.10f" % (iteration, x_best_fvalue))
        while (iteration < self.max_iterations):
            # Sample a new point
            xk = np.random.normal(mu, sigma, size=model_parameters.shape)
            iteration += 1
            xk_fvalue = self.nn.error_function(xk)

            if(xk_fvalue < x_best_fvalue):
                mu, x_best = xk, xk
                x_best_fvalue = xk_fvalue
                print("Iteration=%d, Error=%.10f !!!" % (iteration, x_best_fvalue))
            else:
                print("Iteration=%d, Error=%.10f" % (iteration, x_best_fvalue))

        return x_best, x_best_fvalue

    def random_walk_variable_step(self, model_parameters, s, a, beta):
        iteration = 0
        x = model_parameters
        x_best = model_parameters

        iteration += 1
        x_fvalue = self.nn.error_function(x)
        x_best_fvalue = x_fvalue

        print("Iteration=%d, Error=%.10f" % (iteration, x_best_fvalue))
        while (iteration < self.max_iterations):
            # Pick a random direction
            p = np.random.uniform(-1, 1, size=model_parameters.shape)
            pu = p / norm(p, 1)
            # Calculate xt
            xt = x + s * pu

            iteration += 1
            xt_fvalue = self.nn.error_function(xt)

            if (xt_fvalue < x_fvalue):
                # make step bigger
                sl = a * s

                # try bigger step
                xl = x + sl * pu

                iteration += 1
                xl_fvalue = self.nn.error_function(xl)

                if (xl_fvalue < xt_fvalue):
                    x = xl
                    s = sl
                    x_fvalue = xl_fvalue
                else:
                    x = xt
                    x_fvalue = xt_fvalue
            else:
                # make step smaller
                sm = beta * s

                # try smaller step
                xm = x + sm * pu

                iteration += 1
                xm_fvalue = self.nn.error_function(xm)

                if (xm_fvalue < x_fvalue):
                    x = xm
                    s = sm
                    x_fvalue = xm_fvalue

            # Update best
            if (x_fvalue < x_best_fvalue):
                x_best = x
                x_best_fvalue = x_fvalue

                print("Iteration=%d, Error=%.10f !!!" % (iteration, x_best_fvalue))
            else:
                print("Iteration=%d, Error=%.10f" % (iteration, x_best_fvalue))

        return x_best, x_best_fvalue

    def simulated_annealing(self, model_parameters, t, a, b, r):
        iteration = 0
        x = model_parameters
        x_fvalue = self.nn.error_function(x)

        x_best = model_parameters
        iteration += 1
        x_best_fvalue = self.nn.error_function(x_best)

        # The smallest value the error function can possible get
        f_star = 0

        # Metropolis Function
        Facc = lambda x_fvalue, y_fvalue, t: min(1, math.exp(-(y_fvalue - x_fvalue) / t))
        # Cooling method
        U = lambda a, b, x_fvalue, f_star: b * math.pow((x_fvalue - f_star), a)

        print("Iteration=%d, Error=%.10f" % (iteration, x_best_fvalue))
        while (iteration < self.max_iterations):
            # Pick a random direction
            s = np.random.uniform(-1, 1, size=model_parameters.shape)
            s = s / norm(s, 1)
            # Calculate y
            y = x + r * s

            iteration += 2
            x_fvalue = self.nn.error_function(x)
            y_fvalue = self.nn.error_function(y)

            Facc_k = Facc(x_fvalue, y_fvalue, t)

            if (np.random.uniform(0, 1) < Facc_k):
                x = y
                x_fvalue = y_fvalue

            t = U(a, b, x_fvalue, f_star)

            if (x_fvalue < x_best_fvalue):
                x_best = x
                x_best_fvalue = x_fvalue
                print("Iteration=%d, Error=%.10f !!!" % (iteration, x_best_fvalue))
            else:
                print("Iteration=%d, Error=%.10f" % (iteration, x_best_fvalue))

        return x_best, x_best_fvalue

    def Nelder_Mead(self, model_parameters):
        iteration = 0
        simplex = list()

        # ro operators
        ro_reflection = 1
        ro_expansion = 2 * ro_reflection
        ro_external_contraction = 0.5 * ro_reflection
        ro_internal_contraction = -0.5

        iteration += 1
        # simplex is a list like that list([x1, fvalue_x1], ..., [x_n+1, fvalue_x_n+1])
        simplex.append([model_parameters, self.nn.error_function(model_parameters)])
        simplex_dimention = model_parameters.shape[0] + 1
        print("Iteration=%d, Error=%.10f" % (iteration, simplex[0][1]))

        # create n random x, fvalue_x pairs and add them to simplex
        for i in range(1, simplex_dimention):
            xi = np.random.uniform(0, math.sqrt(5), size=model_parameters.shape)
            iteration += 1
            simplex.append([xi, self.nn.error_function(xi)])

        # This function creates a new point based on ro operator, the worst
        # x value and the x_centroid
        point_generator = lambda x_centroid, x_worst, ro: (1 + ro) * x_centroid - ro * x_worst

        x_best = simplex[0][0]
        x_best_fvalue = simplex[0][1]

        while (iteration < self.max_iterations):
            operator_activated = False
            # Sort simplex
            simplex = sorted(simplex, key=lambda x:x[1])

            # Calculate centroid
            x_centroid = np.zeros(shape=model_parameters.shape)
            for i in range(0, simplex_dimention - 1):
                x_centroid = np.add(x_centroid, simplex[i][0])
            x_centroid = x_centroid / x_centroid.shape[0]

            # Reflection
            iteration += 1
            x_reflection = point_generator(x_centroid, simplex[-1][0], ro_reflection)

            x_reflection_fvalue = self.nn.error_function(x_reflection)
            x_worst_fvalue = simplex[-1][1]
            x_best_fvalue = simplex[0][1]
            x_best = simplex[0][0]

            if (x_best_fvalue <= x_reflection_fvalue < x_worst_fvalue):
                simplex[-1] = [x_reflection, x_reflection_fvalue]
                operator_activated = True

            # Expansion
            if (x_reflection_fvalue < x_best_fvalue):
                x_expansion = point_generator(x_centroid, simplex[-1][0], ro_expansion)
                iteration += 1
                x_expansion_fvalue = self.nn.error_function(x_expansion)
                if (x_expansion_fvalue < x_reflection_fvalue):
                    simplex[-1] = [x_expansion, x_expansion_fvalue]
                    operator_activated = True
                else:
                    simplex[-1] = [x_reflection, x_reflection_fvalue]
                    operator_activated = True

            # External constraction
            x_second_worst_fvalue = simplex[-2][1]
            if (x_second_worst_fvalue <= x_reflection_fvalue < x_worst_fvalue):
                x_external_contraction = point_generator(x_centroid, simplex[-1][0], ro_external_contraction)
                iteration += 1
                x_external_contraction_fvalue = self.nn.error_function(x_external_contraction)
                if (x_external_contraction_fvalue <= x_reflection_fvalue):
                    simplex[-1] = [x_external_contraction, x_external_contraction_fvalue]
                    operator_activated = True

            # Internal Contraction
            if (x_worst_fvalue < x_reflection_fvalue):
                x_internal_contraction = point_generator(x_centroid, simplex[-1][0], ro_internal_contraction)
                iteration += 1
                x_internal_contraction_fvalue = self.nn.error_function(x_internal_contraction)
                if (x_internal_contraction_fvalue < x_worst_fvalue):
                    simplex[-1] = [x_internal_contraction, x_internal_contraction_fvalue]
                    operator_activated = True

            # Shrink Simplex towards best x
            if (not operator_activated):
                for i in range(1, simplex_dimention):
                    simplex[i][0] = simplex[0][0] - ((simplex[i][0] - simplex[0][0]) / 2)

            print("Iteration=%d, Error=%.10f" % (iteration, x_best_fvalue))

        return x_best, x_best_fvalue


# Probabilistic Neural Network
class Neuron:
    def __init__(self, x):
        self.center = x
        self.n = x.shape[0]
        self.n_div_two = self.n / 2
        self.pi_mul_two = 2 * math.pi

    def gaussian_activation(self, x, det_spr_par_mat, inv_spr_par_mat):
        vector = np.subtract(x, self.center)
        activation = 1 / (math.pow(self.pi_mul_two, self.n_div_two) * math.sqrt(det_spr_par_mat))
        activation *= math.exp((-1/2) * (np.transpose(vector) @ inv_spr_par_mat @ vector))
        return activation

class Neural_Network:
    def __init__(self, data, labels):
        self.neuros_class_1 = list()
        self.neuros_class_2 = list()
        self.data = data
        self.labels = labels
        self.parameters_shape = data.shape[1]
        self.__init_neurons()

    def __init_neurons(self):
        # Create neurons with center data_i
        for index, data_i in enumerate(self.data):
            # Create a neuron with center the data_i
            neuron = Neuron(data_i)
            # If data is category one the neuron  goes to
            # category one else in category 2
            if (self.labels[index] == np.array([1, 0])).all():
                self.neuros_class_1.append(neuron)
            else:
                self.neuros_class_2.append(neuron)
        self.neuros_class_1 = np.asarray(self.neuros_class_1)
        self.neuros_class_2 = np.asarray(self.neuros_class_2)

    def __forward_pass(self, model_parameters, data):
        output_vector = list()
        prior_class_1 = 1/2
        prior_class_2 = 1/2
        # sigma -> sigma^2
        model_parameters_squared = np.square(model_parameters)
        det_spr_par_mat = np.prod(model_parameters_squared)
        inv_model_parameters = np.float_power(model_parameters_squared, -1)
        inv_spr_par_mat = np.diag(inv_model_parameters)
        #prior_class_1 = self.neuros_class_1.shape[0] / data.shape[0]
        #prior_class_2 = self.neuros_class_2.shape[0] / data.shape[0]

        # For every data_i in dataset
        for index, data_i in enumerate(data):
            sum_class_1 = 0
            sum_class_2 = 0

            # For every neuron in class 1 neurons pass the data_i
            for sigma_index, neuron in enumerate(self.neuros_class_1):
                sum_class_1 += neuron.gaussian_activation(data_i, det_spr_par_mat, inv_spr_par_mat)

            # For every neuron in class 2 neurons pass the data_i
            for sigma_index, neuron in enumerate(self.neuros_class_2):
                sum_class_2 += neuron.gaussian_activation(data_i, det_spr_par_mat, inv_spr_par_mat)

            try:
                total_norm_sum = prior_class_2 * sum_class_2 + prior_class_1 * sum_class_1
                norm_sum_class_1 = prior_class_1 * sum_class_1 / total_norm_sum
                norm_sum_class_2 = prior_class_2 * sum_class_2 / total_norm_sum
            except:
                norm_sum_class_1 = 0.5
                norm_sum_class_2 = 0.5

            # Output is a probability distribution for example [0.8, 0.2]
            # So the network believes that most likely is category 1 the data_i
            output_vector.append([norm_sum_class_1, norm_sum_class_2])

        return np.asarray(output_vector)

    def error_function(self, model_parameters):
        n_data = self.data.shape[0]
        output_vector = self.__forward_pass(model_parameters, self.data)
        mean_squared_error = np.sum(np.square(norm(output_vector - self.labels, 1))) / n_data

        return mean_squared_error

    def predict(self, model_parameters, data):
        predicted_classes = list()
        output_vector = self.__forward_pass(model_parameters, data)

        for index, vector in enumerate(output_vector):
            # if the class 1 has higher probability classify
            # the example as [1, 0] else as [0, 1]
            if vector[1] < vector[0]:
                predicted_classes.append([1, 0])
            else:
                predicted_classes.append([0, 1])

        return np.asarray(predicted_classes)

def preprocess_data(data_path):
    # Read data
    column_names = ["Class", "Age", "Menopause", "Tumor-size", "Inv-nodes", "Node-caps", "Deg-malig", "Breast", "Breast-quad", "Irradiat"]
    df = pd.read_csv(data_path, names=column_names, header=None, na_values="?")

    # Drop missing values
    df.dropna(inplace=True)

    # Split the data from the labels
    data = df[["Age", "Menopause", "Tumor-size", "Inv-nodes", "Node-caps", "Deg-malig", "Breast", "Breast-quad", "Irradiat"]]
    labels = df[["Class"]]
    del df

    # Use one-hot encoding for categorical features
    categorical_features = ["Age", "Menopause", "Tumor-size", "Inv-nodes", "Node-caps", "Breast", "Breast-quad", "Irradiat"]
    data = pd.get_dummies(data, columns=categorical_features)
    labels = pd.get_dummies(labels, columns=["Class"])

    # Convert dataframes to numpy arrays
    data = data.to_numpy()
    labels = labels.to_numpy()

    # Scale the data features
    standard_scaler = StandardScaler()
    data = standard_scaler.fit_transform(data)

    return data, labels

def main():
    if(len(sys.argv) != 3):
        print("Error: Wrong input.")
        print("Usage: python3 optimization.py from_seed to_seed")
        print("Seed number can be from 0 to 29.")
        print("Example: python3 optimization.py 0 3")
        exit()

    from_seed = int(sys.argv[1])
    to_seed = int(sys.argv[2])

    data_path = "./Data/breast-cancer.data"
    data, labels = preprocess_data(data_path)

    seeds = pd.read_csv("seeds.csv", header=None)
    seeds = seeds.to_numpy()[from_seed : to_seed]
    for seed_index, seed in enumerate(seeds):
        n_splits = 10
        max_iterations = 50

        # Do not change from here and down
        seed_index += from_seed
        print(seed_index)
        np.random.seed(seed)
        K_Fold_Cross_Validation = KFold(n_splits=n_splits, shuffle=True)
        fold_counter = 0

        # Test accuracy
        rand_search_gaussian_distr_test_accuracy = 0
        rand_walk_var_step_test_accuracy = 0
        simulated_annealing_test_accuracy = 0
        nelder_mead_test_accuracy = 0

        # Train accuracy
        rand_search_gaussian_distr_train_accuracy = 0
        rand_walk_var_step_train_accuracy = 0
        simulated_annealing_train_accuracy = 0
        nelder_mead_train_accuracy = 0

        # Timer
        time_0 = time.time()

        for train_index, test_index in K_Fold_Cross_Validation.split(data):
            print("Expiriment: %d, Fold: %d" % (seed_index, fold_counter))

            # Split the dataset to train and testing set
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Initialize Neural Network
            neural_network = Neural_Network(X_train, y_train)
            init_parameters = np.random.uniform(low=0, high=math.sqrt(5), size=(neural_network.parameters_shape))

            # Initialize the optimizer
            optimizer = Optimizer(neural_network, max_iterations=max_iterations)

            # Train the the network with random normal search optimizer
            optimized_parameters, training_error = optimizer.random_search_gaussian_distribution(init_parameters, sigma=0.5)
            # Make the predictions in test set
            predictions = neural_network.predict(optimized_parameters, X_test)
            # Test set classification accuracy
            rand_search_gaussian_distr_test_accuracy += accuracy_score(y_test, predictions)
            print("Random search gaussian distribution test set classification accuracy: %.2f" % (accuracy_score(y_test, predictions)))
            # Make the predictions in test set
            predictions = neural_network.predict(optimized_parameters, X_train)
            # Train set classification accuracy
            rand_search_gaussian_distr_train_accuracy += accuracy_score(y_train, predictions)
            print("Random search gaussian distribution train set classification accuracy: %.2f" % (accuracy_score(y_train, predictions)))

            # Train the the network with random search variable step search optimizer
            optimized_parameters, training_error = optimizer.random_walk_variable_step(init_parameters, s=1, a=1.5, beta=0.5)
            predictions = neural_network.predict(optimized_parameters, X_test)
            rand_walk_var_step_test_accuracy += accuracy_score(y_test, predictions)
            print("Random walk variable step test set classification accuracy: %.2f" % (accuracy_score(y_test, predictions)))
            predictions = neural_network.predict(optimized_parameters, X_train)
            rand_walk_var_step_train_accuracy += accuracy_score(y_train, predictions)
            print("Random walk variable step train set classification accuracy: %.2f" % (accuracy_score(y_train, predictions)))

            # Train the the network with simulated annealing optimizer
            optimized_parameters, training_error = optimizer.simulated_annealing(init_parameters, t=1, a=1, b=1, r=0.5)
            predictions = neural_network.predict(optimized_parameters, X_test)
            simulated_annealing_test_accuracy += accuracy_score(y_test, predictions)
            print("Simulated annealing test set classification accuracy: %.2f" % (accuracy_score(y_test, predictions)))
            predictions = neural_network.predict(optimized_parameters, X_train)
            simulated_annealing_train_accuracy += accuracy_score(y_train, predictions)
            print("Simulated annealing train set classification accuracy: %.2f" % (accuracy_score(y_train, predictions)))

            # Train the the network with simplex optimizer
            optimized_parameters, training_error = optimizer.Nelder_Mead(init_parameters)
            predictions = neural_network.predict(optimized_parameters, X_test)
            nelder_mead_test_accuracy += accuracy_score(y_test, predictions)
            print("Nelder Mead test set classification accuracy: %.2f" % (accuracy_score(y_test, predictions)))
            predictions = neural_network.predict(optimized_parameters, X_train)
            nelder_mead_train_accuracy += accuracy_score(y_train, predictions)
            print("Nelder Mead train set classification accuracy: %.2f" % (accuracy_score(y_train, predictions)))

            fold_counter += 1

        # Timer
        time_1 = time.time()
        time_difference = time_1 - time_0
        print("Time in seconds: %.2f" % (time_difference))

        # Average test set accuracy
        rand_search_gaussian_distr_test_accuracy /= n_splits
        rand_walk_var_step_test_accuracy /= n_splits
        simulated_annealing_test_accuracy /= n_splits
        nelder_mead_test_accuracy /= n_splits

        # Average train set accuracy
        rand_search_gaussian_distr_train_accuracy /= n_splits
        rand_walk_var_step_train_accuracy /= n_splits
        simulated_annealing_train_accuracy /= n_splits
        nelder_mead_train_accuracy /= n_splits

        # Create dataframes
        df_testset_accuracy = pd.DataFrame()
        df_trainset_accuracy = pd.DataFrame()

        # Log test set accuracy
        test_accuracy = pd.Series(data={"Random_search_gaussian_distribution":rand_search_gaussian_distr_test_accuracy,
                                        "Random_walk_variable_step":rand_walk_var_step_test_accuracy,
                                        "Simulated_Annealing":simulated_annealing_test_accuracy,
                                        "Nelder_Mead":nelder_mead_test_accuracy,
                                        "ZSeed_index":seed_index})

        df_testset_accuracy = df_testset_accuracy.append(test_accuracy, ignore_index=True)
        df_testset_accuracy.to_excel("Testset_classification_accuracy_" + str(seed_index) + ".xlsx")

        # Log train set accuracy
        train_accuracy = pd.Series(data={"Random_search_gaussian_distribution":rand_search_gaussian_distr_train_accuracy,
                                         "Random_walk_variable_step":rand_walk_var_step_train_accuracy,
                                         "Simulated_Annealing":simulated_annealing_train_accuracy,
                                         "Nelder_Mead":nelder_mead_train_accuracy,
                                         "ZSeed_index":seed_index})

        df_trainset_accuracy = df_trainset_accuracy.append(train_accuracy, ignore_index=True)
        df_trainset_accuracy.to_excel("Trainset_classification_accuracy_" + str(seed_index) + ".xlsx")


if __name__ == '__main__':
    main()
