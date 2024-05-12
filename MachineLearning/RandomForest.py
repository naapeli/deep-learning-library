import torch
from multiprocessing import Process, Queue
from collections import Counter
from MachineLearning.DecisionTree import DecisionTree


class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = [DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split) for _ in range(n_trees)]

    def fit(self, x, y):
        # for tree in self.trees:
        #     tree.fit(*self._bootstrap_sample(x, y))
        try:
            processes = []
            q = Queue()
            for i, tree in enumerate(self.trees):
                process = Process(target=tree._fit_for_random_forest, args=(*self._bootstrap_sample(x, y), i, q))
                processes.append(process)
                process.start()
            i = 0
            while i < len(self.trees):
                index, root, n_features = q.get()
                self.trees[index].root = root
                self.trees[index].n_features = n_features
                i += 1
            for p in processes:
                p.join()
        except Exception:
            print("""Try putting your code in a if __name__ == "__main__": block""")

    def predict(self, x):
        predictions = torch.stack([tree.predict(x) for tree in self.trees]).T
        return torch.tensor([Counter(sample_prediction).most_common(1)[0][0] for sample_prediction in predictions])
    
    def _bootstrap_sample(self, x, y):
        indices = torch.randint(high=len(y), size=(len(y), 1)).flatten()
        return x[indices], y[indices]
