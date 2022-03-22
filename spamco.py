from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class SPamCo:
    # initiate classifier
    clfs: list = field(default_factory=list)
    labeled_data: list = field(default=list)
    unlabeled_data: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    num_view: int = 2
    gamma: float = 0.3
    regularizer: str = 'hard'
    iterations: float = 10
    add_num: int = 6
    update_add_num: int = 6
    sel_ids: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    scores: list = field(default_factory=list)
        
    def __str__(self):
        #todo: Print a clean representation of the perceptron
        print(f'Weights:\t{self.weights}')
        print(f'Classifiers:\t{self.clfs}')
        print(f'Scores:\t{self.scores}')
        print(f'num_view:\t{self.num_view}')
        print(f'Gamma:\t{self.gamma}')
   
    def update_train(self, pred_y):
        add_ids = np.where(np.array(self.sel_ids) != 0)[0]
        add_data = [d[add_ids] for d in self.unlabeled_data]
        new_train_data = [np.concatenate([d1, d2]) for d1,d2 in zip(self.labeled_data, add_data)]
        add_y = pred_y[add_ids]
        new_train_y = np.concatenate([self.labels, pred_y[add_ids]])
        return new_train_data, new_train_y

    def get_lambda_class(self, view, pred_y):
        y = self.labels
        lambdas = np.zeros(self.scores[view].shape[1])
        add_ids = np.zeros(self.scores[view].shape[0])
        clss = np.unique(y)
        assert self.scores[view].shape[1] == len(clss)
        ratio_per_class = [sum(y == c)/len(y) for c in clss]
        for cls in range(len(clss)):
            indices = np.where(pred_y == cls)[0]
            if len(indices) == 0:
                continue
            cls_score = self.scores[indices, cls]
            idx_sort = np.argsort(cls_score)
            add_num = min(int(np.ceil(ratio_per_class[cls] * self.add_num)),
                          indices.shape[0])
            add_ids[indices[idx_sort[-self.add_num:]]] = 1
            lambdas[cls] = cls_score[idx_sort[-self.add_num]] - 0.1
        return add_ids.astype('bool'), lambdas

    def get_ids_weights(self, view, pred_y):
        '''
        pred_prob: predicted probability of all views on untrain data
        pred_y: predicted label for untrain data
        train_data: training data
        max_add: number of selected data
        gamma: view correlation hyper-parameter
        '''
        #print('Getting Lambdas')
        add_ids, lambdas = self.get_lambda_class(view, pred_y)
        #print('Getting Weights')
        weight = np.array([(self.scores[i, l] - lambdas[l]) / (self.gamma + 1e-5)
                           for i, l in enumerate(pred_y)],
                          dtype='float32')
        weight[~add_ids] = 0
        if self.regularizer == 'hard' or self.gamma == 0:
            weight[add_ids] = 1
            #print(f'Returning Weights {weight}')
            return add_ids, weight
        weight[weight < 0] = 0
        weight[weight > 1] = 1
        #print(f'Returning Weights {weight}')
        return add_ids, weight

    def update_ids_weights(self, view, pred_y):
        num_view = len(self.scores)
        for v in range(num_view):
            if v == view:
                continue
            ov = self.sel_ids[v]
            self.scores[view][ov, pred_y[ov]] += self.gamma * self.weights[v][ov] / (num_view-1)
        sel_id, weight = self.get_ids_weights(view, pred_y)
        return sel_id, weight
    
    def fit(self, classifier):
        
        print('Intiating first classifiers and predictions')
        for view in range(self.num_view):
            self.clfs.append(classifier)
            self.clfs[view].fit(self.labeled_data[view], self.labels)
            self.scores.append(self.clfs[view].predict_proba(self.unlabeled_data[view]))
  
            pred_y = np.argmax(sum(self.scores), axis = 1)

        # initiate weights for unlabled examples
        print('Initiating weights for unlabeled examples')
        for view in range(self.num_view):
            sel_id, weight = self.get_ids_weights(self.scores[view], pred_y)
            
            self.sel_ids.append(sel_id)
            self.weights.append(weight)

        print(f'Running {self.iterations} steps')
        pred_y = np.argmax(sum(self.scores), axis = 1)
        for step in range(self.iterations):
            for view in range(2):

                # Update v
                self.sel_ids[view], self.weights[view] = self.update_ids_weights(
                    view,
                    pred_y
                )

                #update w
                new_labeled_data, new_labels = self.update_train(pred_y)
                self.clfs[view].fit(new_labeled_data[view], new_labels)

                add_num += update_add_num
                # udpate sample weights
                self.sel_ids[view], self.weights[view] = self.update_ids_weights(
                    view,
                    pred_y
                )

                pred_y = np.argmax(sum(self.scores[view]), axis = 1)

            if len(new_labeled_data[0]) >= len(self.unlabeled_data[0]): break
        return clfs