def __init__(self, df_inf_score_up, df_inf_score_down):
    self.df_inf_score_up = df_inf_score_up
    self.df_inf_score_down = df_inf_score_down
    self.df_inf_score = pd.concat([df_inf_score_up, df_inf_score_down])['inf_score']
    self.genes = list(pd.concat([df_inf_score_up, df_inf_score_down]).index)
    dict_inf_score = {}
    for (gene, inf_score) in zip(self.genes, self.df_inf_score):
        dict_inf_score[gene] = inf_score
    self.dict_inf_score = dict_inf_score


def get_inf_score(self, gene):
    try:
        return self.dict_inf_score[gene]
    except KeyError:
        return 1