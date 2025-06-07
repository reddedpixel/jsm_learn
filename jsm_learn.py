import pandas as pd
from jsm_learn import methods

class JSM:
    """
    # JSM
    Creates a model providing the functionality of JSM-method for data analysis. 
    """
    def __init__(self,
               ban_counterexamples : bool = False,
               method : str = 'norris',
               ext_threshold : int = 2,
               int_threshold : int = 3):
        self.X_pos = pd.DataFrame()
        """Positive examples (given and predicted)."""
        self.X_neg = pd.DataFrame()
        """Negative examples (given and predicted)."""
        self.X_tau = pd.DataFrame()
        """Undecided examples (given and predicted)."""
        self.X_contra = pd.DataFrame()
        """Contradictory examples (given and predicted)."""
        self.positive_causes = pd.DataFrame()
        """Predicted positive causes."""
        self.negative_causes = pd.DataFrame()
        """Predicted negative causes."""
        self.ban_counterexamples = ban_counterexamples
        """"""
        self.method = method
        """
        The chosen algorithm for the calculation of minimal intersections 
        during the induction stage.
        """
        self.ext_threshold = ext_threshold
        """The extensional threshold."""
        self.int_threshold = int_threshold
        """The intensional threshold."""
        self.steps = 0 
        """Amount of completed JSM-method iterations since the instantiation of the model."""
        self.is_causally_complete = False
        self.lost_pos_ids = []
        self.lost_neg_ids = []

    def fit(self, X : pd.DataFrame, y : pd.Series):
        """
        Fits the training data **X** and the target data **y** to the model.

        :param X: Training data.
        :param y: Target data.
        """
        if '_step' not in X.columns:
            X['_step'] = 0
        self.X_pos = X[y == 1].copy()
        self.X_neg = X[y == -1].copy()
        self.X_tau = X[y.isna()].copy()
        self.X_contra = X[y == 0].copy()

    def predict(self, steps : int = -1, show_steps : bool = False):
        """
        Applies the JSM method.

        :param steps: The amount of iterations to perform. Set to -1 to 
        perform until no new values can be predicted.
        :param show_steps: Print the execution of the method to console.
        """
        is_finished = False
        while not is_finished:
            if show_steps:
                print(f"... Step {self.steps} ...")
                print(f"Positive examples: {self.X_pos.index.tolist()}")
                print(f"Negative examples: {self.X_neg.index.tolist()}")
                print(f"Undecided examples: {self.X_tau.index.tolist()}")
                print(f"Contradicting examples: {self.X_contra.index.tolist()}")
            self.steps += 1
            self.__induction()
            if steps > 0:
                steps -= 1
            elif steps == 0:
                is_finished = True
            else:
                is_finished = self.__analogy()
        else:
            self.is_causally_complete = self.__abduction()
            if show_steps:
               print("Causally complete:", self.is_causally_complete)

    def to_df(self):
        """
        Returns a dataframe containing all the examples with their predicted values.
        """
        new_X_pos = self.X_pos.copy()
        new_X_pos['target'] = 1
        new_X_neg = self.X_neg.copy()
        new_X_neg['target'] = -1
        new_X_tau = self.X_tau.copy()
        new_X_tau['target'] = None
        new_X_contra = self.X_contra.copy()
        new_X_contra['target'] = 0
        return pd.concat([new_X_pos, new_X_neg, new_X_contra, new_X_tau])

    def __induction(self):
            pos_cause_candidates = self.__filtration(self.__find_similarities(self.X_pos.drop(columns='_step')))
            neg_cause_candidates = self.__filtration(self.__find_similarities(self.X_neg.drop(columns='_step')))

            self.positive_causes = self.__falsification(pos_cause_candidates, neg_cause_candidates, self.X_pos)
            self.negative_causes = self.__falsification(neg_cause_candidates, pos_cause_candidates, self.X_neg)

    def __find_similarities(self,  objects):
        if self.method == 'norris':
            return methods.norris(objects)
        elif self.method == 'khazanovskiy':
            return methods.khazanovskiy(objects)
    
    def __filtration(self, candidates_df : pd.DataFrame):
            result = pd.DataFrame(columns=candidates_df.columns)
            for i in range(len(candidates_df)):
                if len(candidates_df.iloc[i]['_ext']) >= self.ext_threshold \
                    and candidates_df.iloc[i].drop(['_ext']).sum() >= self.int_threshold:
                    result.loc[len(result)] = candidates_df.iloc[i]
            return result

    def __falsification(self, candidates_df, opposing_candidates_df, opposing_X):
        causes = pd.DataFrame()
        if self.ban_counterexamples:
            #!TODO
            pass
        else:
            merged_df = candidates_df.merge(opposing_candidates_df, on=candidates_df.columns.tolist(),
                    how='left', indicator=True)
            causes = candidates_df[merged_df['_merge'] == 'left_only']
        return causes

    def __analogy(self):
        is_finished = True
        rows_to_drop = []
        for i in self.X_tau.index.tolist():
            x = self.X_tau.loc[i].drop('_step').astype(bool)
            is_pos = False
            is_neg = False
            for p in self.positive_causes.index.tolist():
                pos_cause = self.positive_causes.loc[p].drop(['_ext']).astype(bool)
                if pos_cause.equals(x & pos_cause):
                    is_pos = True
                    break
            for n in self.negative_causes.index.tolist():
                neg_cause = self.negative_causes.loc[n].drop(['_ext']).astype(bool)
                if neg_cause.equals(x & neg_cause):
                    is_neg = True
                    break
            if is_pos or is_neg:
                x.at['_step'] = self.steps
                if not is_neg:
                    x['target'] = 1
                    self.X_pos.loc[i] = x
                elif not is_pos:
                    x['target'] = -1
                    self.X_neg.loc[i] = x
                else:
                    x['target'] = 0
                    self.X_contra.loc[i] = x
                rows_to_drop.append(i)
                is_finished = False
        self.X_tau.drop(rows_to_drop, inplace=True)
        return is_finished

    def __abduction(self):
        all_pos_ext = set()
        all_neg_ext = set()
        for ext in self.positive_causes["_ext"]:
            all_pos_ext = all_pos_ext.union(ext)
        all_pos_ids = set(self.X_pos.index.tolist())
        self.lost_pos_ids = all_pos_ids - all_pos_ext
        for ext in self.negative_causes["_ext"]:
            all_neg_ext = all_neg_ext.union(ext)
        all_neg_ids = set(self.X_neg.index.tolist())
        self.lost_neg_ids = all_neg_ids - all_neg_ext
        return self.lost_pos_ids.union(self.lost_neg_ids) == set()
