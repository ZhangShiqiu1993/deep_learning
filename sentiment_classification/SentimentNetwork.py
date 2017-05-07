import time
import sys
import numpy as np

class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes=10, min_count=20, polarity_cutoff=0.05, learning_rate=0.1, ):
        np.random.seed(1)
        self.pre_process_data(reviews, labels, min_count, polarity_cutoff)
        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels, min_count, polarity_cutoff):
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i in range(len(reviews)):
            review = reviews[i]
            label = labels[i]
            if label == 'POSITIVE':
                for word in review.split(' '):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in review.split(' '):
                    negative_counts[word] += 1
                    total_counts[word] += 1

        pos_neg_ratios = Counter()
        for word, count in total_counts.most_common():
            if count >= 50:
                pos_neg_ratios[word] = positive_counts[word] / float(negative_counts[word] + 1)
                
        for word, ratio in pos_neg_ratios.most_common():
            if ratio > 1:
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log(1 / (ratio + 0.01))
        

        review_vocab = set()
        for review in reviews:
            for word in review.split(' '):
                if total_counts[word] > min_count:
                    if word in pos_neg_ratios:
                        if np.abs(pos_neg_ratios[word]) >= polarity_cutoff:
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)

        self.review_vocab = list(review_vocab)

        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
            
        self.label_vocab = list(label_vocab)

        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.learning_rate = learning_rate

        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, (self.hidden_nodes, self.output_nodes))
        self.layer_1 = np.zeros((1, self.hidden_nodes))

    def get_target_for_label(self, label):
        if label == 'POSITIVE':
            return 1
        else:
            return 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        return output * (1 - output)

    def train(self, training_reviews_raw, training_labels):
        training_reviews = []
        for review in training_reviews_raw:
            indices_set = set()
            for word in review.split(" "):
                if word in self.word2index:
                    indices_set.add(self.word2index[word])
            training_reviews.append(list(indices_set))

        assert len(training_reviews_raw) == len(training_labels)

        correct_so_far = 0

        start = time.time()

        for i in range(len(training_reviews)):
            review = training_reviews[i]
            label = training_labels[i]
            
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]

            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            layer_2_error = layer_2 - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
            layer_1_delta = layer_1_error

            self.weights_1_2 -= self.learning_rate * self.layer_1.T.dot(layer_2_delta)

            for index in review:
                self.weights_0_1[index] -= self.learning_rate * layer_1_delta[0]

            if np.abs(layer_2_error) < 0.5:
                correct_so_far += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(training_reviews_raw)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i + 1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i + 1))[:4] + "%")
            if (i % 2500 == 0):
                print("")

    def test(self, testing_reviews, testing_labels):
        correct = 0

        start = time.time()

        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if (pred == testing_labels[i]):
                correct += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i + 1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i + 1))[:4] + "%")

    def run(self, review_raw):
        self.layer_1 *= 0
        indices_set = set()
        for word in review_raw.lower().split(' '):
            if word in self.word2index:
                indices_set.add(self.word2index[word])

        for index in indices_set:
            self.layer_1 += self.weights_0_1[index]

        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

        if layer_2[0] >= 0.5:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'
