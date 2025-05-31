import pandas as pd

def norris(obj_df: pd.DataFrame, current_step : int):
    """Performs Norris's algorithm to calculate minimal intersections."""
    #!TODO: give current term dictionary?
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

def anshakov(obj_df : pd.DataFrame):
    terms_df = pd.DataFrame(columns=obj_df.columns)
    terms_df['_ext'] = None
    obj_df.sort_values(by=obj_df.columns, 
                       inplace=True, 
                       ascending=[True] * len(obj_df.columns))
    for i in range(len(obj_df)):
        i_intersection = obj_df.iloc[i]
        i_ids = []
        for j in range(i + 1, len(obj_df)):
            cur_intersection = i_intersection & obj_df.iloc[j]
            if cur_intersection.any():
                i_intersection = cur_intersection
                i_ids.append(j)
        if len(i_ids) > 2:
            i_intersection['_ext'] = frozenset(i_ids)
            terms_df.loc[len(terms_df)] = i_intersection
        #!TODO: дописать
        