import pandas as pd

def norris(obj_df: pd.DataFrame):
    """Performs Norris's algorithm to calculate minimal intersections."""
    #!TODO: give current term dictionary?
    terms_df = pd.DataFrame(columns=obj_df.columns)
    terms_df['_ext'] = None
    obj_ids = obj_df.index.tolist()
    for k in obj_ids:
        abs_canonical = True
        row = obj_df.loc[k].copy()
        for i in range(len(terms_df)):
            term = terms_df.iloc[i]
            term_dropped = term.drop(['_ext'], inplace=False).astype(bool)
            term_intersection = row & term_dropped
            term_set = set(term['_ext'])
            if term_dropped.equals(term_intersection):
                term_set.add(k)
                terms_df.at[i, '_ext'] = frozenset(term_set)
            else:
                missing_i = set()
                prev_obj_ids = [id for id in obj_ids if id < k]
                for i_prev in prev_obj_ids:
                    prev_obj = obj_df.loc[i_prev]
                    if row.equals(prev_obj & row):
                        abs_canonical = False
                    if term_intersection.equals(prev_obj & term_intersection)\
                        and i_prev not in term['_ext']:
                        missing_i.add(i_prev)
                if missing_i == set():
                    new_term = term_intersection
                    new_term.at['_ext'] = frozenset(term_set.union({k}))
                    terms_df.loc[len(terms_df)] = new_term
        if abs_canonical:
            row['_ext'] = {k}
            terms_df.loc[len(terms_df)] = row
    return terms_df
        
def khazanovskiy(obj_df :pd.DataFrame):
    terms_df = pd.DataFrame(columns=obj_df.columns)
    terms_df['_ext'] = None
    obj_ids = obj_df.index.tolist()
    checked_attributes = set()
    all_attributes = set(obj_df.columns.tolist())
    for attribute in obj_df.columns:
        if checked_attributes == all_attributes:
            break
        if attribute not in checked_attributes:
            full_intersection = pd.Series([True] * len(obj_df.columns), index=obj_df.columns)\
                .drop(columns=list(checked_attributes))
            empty_intersection = pd.Series([False] * len(obj_df.columns), index=obj_df.columns)\
                .drop(columns=list(checked_attributes))
            current_intersection = full_intersection.copy()
            complement = empty_intersection.copy()
            positive_ids = set()
            for i in obj_ids:
                if obj_df.loc[i][attribute] == True:
                    positive_ids.add(i)
                    current_intersection = current_intersection & obj_df\
                        .loc[i].drop(columns=list(checked_attributes))
                else:
                    complement = complement | obj_df.loc[i].drop(columns=list(checked_attributes))
            overlap = current_intersection & complement
            if len(positive_ids) >= 2 and overlap.equals(empty_intersection):
                checked_attributes = checked_attributes\
                    .union(set(current_intersection[current_intersection == True].index.tolist()))
                current_intersection['_ext'] = frozenset(positive_ids)
                terms_df.loc[len(terms_df)] = current_intersection
            else:
                checked_attributes.add(attribute)
    return terms_df
