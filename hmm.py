from pomegranate import DiscreteDistribution, HiddenMarkovModel, State

from stat_calc import Pos_Data_Processor, Zemberek_Server_Pos_Tagger


class HMM_Text_Generator:
    def __init__(self, start_pos, end_pos, all_pos_tag):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.all_pos_tag = all_pos_tag

    def __get_discrete_dist(self, dist: dict[str:int]):
        return DiscreteDistribution(dist)

    def get_model(self) -> HiddenMarkovModel:
        model = HiddenMarkovModel("Text Generator")

        state_list = []
        for pos in self.all_pos_tag:
            dist = self.__get_discrete_dist(self.all_pos_tag[pos])
            state = State(dist, name=str(pos))
            state_list.append(state)

        model.add_state(state_list)
        model.bake()


file_path = "/home/algin/İndirilenler/Telegram Desktop/Data/result.json"
req_obj = Zemberek_Server_Pos_Tagger("http://localhost", 4567, file_path)
# tag, start, end, transition_matrix_zip = req_obj.get_df_pos(req_obj.df)
tag, start, end, transition_matrix_zip = req_obj.get_df_pos_parallel(
    cheat_pickle="messages.pickle"
)
indices, tmatrix = zip(*transition_matrix_zip)
indices = list(indices)
print(indices)
# generator = HMM_Text_Generator(start, end, tag)
