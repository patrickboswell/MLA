from dataclasses import dataclass, field
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
        print(f'Labels:\t{self.labels}')

    def get_classifiers(self):
        return self.clfs
   
    def update_train(self, view, pred_y):
        add_ids = np.where(np.array(self.sel_ids[view]) != 0)[0]
        add_data = [d[add_ids] for d in self.unlabeled_data]
        new_train_data = [np.concatenate([d1, d2]) for d1,d2 in zip(self.labeled_data, add_data)]
        add_y = pred_y[add_ids]
        new_train_y = np.concatenate([self.labels, add_y])
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
            cls_score = self.scores[view][indices, cls]
            idx_sort = np.argsort(cls_score)
            add_num = min(int(np.ceil(ratio_per_class[cls] * self.add_num)),
                          indices.shape[0])
            add_ids[indices[idx_sort[-add_num:]]] = 1
            lambdas[cls] = cls_score[idx_sort[-add_num]] - 0.1
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
        weight = np.array([(self.scores[view][i, l] - lambdas[l]) / (self.gamma + 1e-5)
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
        
        #print('Intiating first classifiers and predictions')
        for view in range(self.num_view):
            self.clfs.append(classifier)
            self.clfs[view].fit(self.labeled_data[view], self.labels)
            self.scores.append(self.clfs[view].predict_proba(self.unlabeled_data[view]))
  
            pred_y = np.argmax(sum(self.scores), axis = 1)

        # initiate weights for unlabled examples
        #print('Initiating weights for unlabeled examples')
        for view in range(self.num_view):
            sel_id, weight = self.get_ids_weights(view, pred_y)
            
            self.sel_ids.append(sel_id)
            self.weights.append(weight)

        #print(f'Running {self.iterations} steps')
        pred_y = np.argmax(sum(self.scores), axis = 1)
        for step in range(self.iterations):
            for view in range(self.num_view):

                # Update Training data
                self.sel_ids[view], self.weights[view] = self.update_ids_weights(
                    view,
                    pred_y
                )
                new_labeled_data, new_labels = self.update_train(view, pred_y)

                # Update parameters
                self.clfs[view].fit(new_labeled_data[view], new_labels)
                self.scores[view] = self.clfs[view].predict_proba(self.unlabeled_data[view])
                
                # Update predictions
                pred_y = np.argmax(sum(self.scores), axis = 1)

                # Update lambda
                self.add_num += self.update_add_num

                # Update Training data
                self.sel_ids[view], self.weights[view] = self.update_ids_weights(
                    view,
                    pred_y
                )

            if len(new_labeled_data[0]) >= len(self.unlabeled_data[0]): break
                
class Validation:
    @staticmethod
    def validation(model_params, train_data_x, train_data_y, percent_labeled, random_seed, spaco=False, single_view=False, cv=False, folds=5, iters=100, verbosity=10 ):
    
        metrics = []

        if cv:
            return metrics
        else:
            for step in range(iters):            
                if spaco:
                    l_data = []
                    u_data = []
                    predictions = []

                    x_train_val, x_test_val, y_train_val, y_test_val = train_test_split(
                        train_data_x, 
                        train_data_y, 
                        test_size=0.30,
                        stratify=train_data_y,
                        random_state=random_seed[step]
                    )

                    x_train, x_test, y_train, y_test = train_test_split(
                        x_train_val, 
                        y_train_val, 
                        test_size=1 - percent_labeled,
                        stratify=y_train_val,
                        random_state=random_seed[step]
                    )
                    if single_view:
                        l_data.append(x_train)
                        u_data.append(x_test)

                        spaco = SPamCo(
                            labeled_data=l_data,
                            unlabeled_data=u_data,
                            labels=y_train,
                            num_view=model_params.get('num_view'),
                            gamma=model_params.get('gamma'),
                            iterations=model_params.get('steps'),
                            regularizer=model_params.get('regularizer')
                        )

                        spaco.fit(model_params.get('classifier'))
                        clfs = spaco.get_classifiers()
                        pred_y = clfs[0].predict(x_test_val)
                        accuracy = accuracy_score(pred_y, y_test_val)
                        metrics.append(accuracy)
                        if step % verbosity == 0:
                            print(f'Validation Iteration: {step} Accuracy: {accuracy} Labels: {len(y_train)}')
                    else:
                        for i in range(x_train.shape[1]):
                            l_data.append(x_train[:,i].reshape(-1,1))
                            u_data.append(x_test[:,i].reshape(-1,1))

                        spaco = SPamCo(
                            labeled_data=l_data,
                            unlabeled_data=u_data,
                            labels=y_train,
                            num_view=x_train.shape[1],
                            gamma=model_params.get('gamma'),
                            iterations=model_params.get('steps'),
                            regularizer=model_params.get('regularizer')
                        )
                        spaco.fit(model_params.get('classifier'))
                        clfs = spaco.get_classifiers()

                        for view in range(len(clfs)):
                            pred = clfs[view].predict_proba(x_test_val[:,view].reshape(-1,1))
                            predictions.append(pred)

                        pred_y = np.argmax(sum(predictions), axis = 1)
                        accuracy = accuracy_score(pred_y, y_test_val)
                        metrics.append(accuracy)
                        if step % verbosity == 0:
                            print(f'Validation Iteration: {step} Accuracy: {accuracy} Labels: {len(y_train)}')
                else:

                    x_train_val, x_test_val, y_train_val, y_test_val = train_test_split(
                        train_data_x, 
                        train_data_y, 
                        test_size=0.30,
                        stratify=train_data_y,
                        random_state=random_seed[step]
                    )

                    x_train, x_test, y_train, y_test = train_test_split(
                        x_train_val, 
                        y_train_val, 
                        test_size=1 - percent_labeled,
                        stratify=y_train_val,
                        random_state=random_seed[step]
                    )

                    clf = model_params.get('classifier')
                    clf.fit(x_train, y_train)

                    clf_pred = clf.predict(x_test_val)
                    accuracy = accuracy_score(clf_pred, y_test_val)
                    metrics.append(accuracy)
                    if step % verbosity == 0:
                        print(f'Validation Iteration: {step} Accuracy: {accuracy} Labels: {len(y_train)}')
                    

        return spaco, metrics  