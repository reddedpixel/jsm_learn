import pandas as pd

class JSM:
    def __init__(self,
               ban_counterexamples : bool = False,
               method : str = 'norris',
               ext_threshold : int = 2,
               int_threshold : int = 3):
        self.X_pos = pd.DataFrame()
        self.X_neg = pd.DataFrame()
        self.X_tau = pd.DataFrame()
        self.X_contra = pd.DataFrame()
        self.positive_causes = pd.DataFrame()
        self.negative_causes = pd.DataFrame()
        self.ban_counterexamples = ban_counterexamples
        self.method = method
        self.ext_threshold = ext_threshold
        self.int_threshold = int_threshold
        self.steps = 0
        self.is_causally_complete = False
        self.lost_pos_ids = []
        self.lost_neg_ids = []

    def fit(self, X : pd.DataFrame, y : pd.Series):
        if '_step' not in X.columns:
            X['_step'] = 0
        self.X_pos = X[y == 1]
        self.X_neg = X[y == -1]
        self.X_tau = X[y.isna()]
        self.X_contra = X[y == 0]

    def predict(self, show_steps : bool = False):
        is_finished = False
        while not is_finished:
            if show_steps:
                print(f"... Step {self.steps} ...")
                print(f"Positive examples: {self.X_pos.index.tolist()}")
                print(f"Negative examples: {self.X_neg.index.tolist()}")
                print(f"Undecided examples: {self.X_tau.index.tolist()}")
                print(f"Contradicting examples: {self.X_contra.index.tolist()}")
            self.steps += 1
            self.induction()
            is_finished = self.analogy()
        else:
            self.is_causally_complete = self.abduction()
            if show_steps:
               print("Causally complete:", self.is_causally_complete)

    def to_df(self):
        new_X_pos = self.X_pos.copy()
        new_X_pos['_target'] = 1
        new_X_neg = self.X_neg.copy()
        new_X_neg['_target'] = -1
        new_X_tau = self.X_tau.copy()
        new_X_tau['_target'] = None
        new_X_contra = self.X_contra.copy()
        new_X_contra['_target'] = 0
        return pd.concat([new_X_pos, new_X_neg, new_X_contra, new_X_tau])

    def induction(self):
            pos_cause_candidates = self.filtration(self.find_similarities(self.X_pos))
            neg_cause_candidates = self.filtration(self.find_similarities(self.X_neg))

            self.positive_causes = self.falsification(pos_cause_candidates, neg_cause_candidates, self.X_pos)
            self.negative_causes = self.falsification(neg_cause_candidates, pos_cause_candidates, self.X_neg)

    def find_similarities(self,  objects):
        #!TODO add other methods
        return norris(objects, self.steps)
    
    def filtration(self, candidates_df : pd.DataFrame):
            result = pd.DataFrame(columns=candidates_df.columns)
            for i in range(len(candidates_df)):
                if len(candidates_df.iloc[i]['_ext']) >= self.ext_threshold \
                    and candidates_df.iloc[i].drop(['_step', '_ext']).sum() >= self.int_threshold:
                    result.loc[len(result)] = candidates_df.iloc[i]
            return result

    def falsification(self, candidates_df, opposing_candidates_df, opposing_X):
        causes = pd.DataFrame()
        if self.ban_counterexamples:
            #!TODO
            pass
        else:
            merged_df = candidates_df.merge(opposing_candidates_df, on=candidates_df.columns.tolist(),
                    how='left', indicator=True)
            causes = candidates_df[merged_df['_merge'] == 'left_only']
        return causes

    def analogy(self):
        is_finished = True
        rows_to_drop = []
        for i in self.X_tau.index.tolist():
            x = self.X_tau.loc[i].drop('_step').astype(bool)
            is_pos = False
            is_neg = False
            for p in self.positive_causes.index.tolist():
                pos_cause = self.positive_causes.loc[p].drop(['_ext', '_step']).astype(bool)
                if pos_cause.equals(x & pos_cause):
                    is_pos = True
                    break
            for n in self.negative_causes.index.tolist():
                neg_cause = self.negative_causes.loc[n].drop(['_ext', '_step']).astype(bool)
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

    def abduction(self):
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


def norris(obj_df: pd.DataFrame, current_step : int):
    # POSSIBLE OPTIMISATION: give current term dictionary?
    terms_df = pd.DataFrame(columns=obj_df.columns)
    terms_df['_ext'] = None
    obj_ids = obj_df.index.tolist()
    for k in obj_ids:
        abs_canonical = True
        row = obj_df.loc[k].copy()
        row_dropped = row.drop('_step')
        for i in range(len(terms_df)):
            term = terms_df.iloc[i]
            term_dropped = term.drop(['_ext', '_step'], inplace=False).astype(bool)
            term_intersection = row_dropped & term_dropped
            term_set = set(term['_ext'])
            if term_dropped.equals(term_intersection):
                term_set.add(k)
                terms_df.at[i, '_ext'] = frozenset(term_set)
            else:
                missing_i = set()
                prev_obj_ids = [id for id in obj_ids if id < k]
                for i_prev in prev_obj_ids:
                    prev_obj = obj_df.loc[i_prev]
                    prev_obj_dropped = prev_obj.drop('_step')
                    if row_dropped.equals(prev_obj_dropped & row_dropped):
                        abs_canonical = False
                    if term_intersection.equals(prev_obj_dropped & term_intersection)\
                        and i_prev not in term['_ext']:
                        missing_i.add(i_prev)
                if missing_i == set():
                    new_term = term_intersection
                    new_term.at['_step'] = current_step
                    new_term.at['_ext'] = frozenset(term_set.union({k}))
                    terms_df.loc[len(terms_df)] = new_term
        if abs_canonical:
            row['_ext'] = {k}
            terms_df.loc[len(terms_df)] = row
    return terms_df
