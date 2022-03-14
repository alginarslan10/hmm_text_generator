from pomegranate import DiscreteDistribution, HiddenMarkovModel

from stat_calc import Pos_Data_Processor, Zemberek_Server_Pos_Tagger


class HMM_Text_Generator:
    def __init__(self, start_pos, end_pos, all_pos_tag, tmatrix):
        (
            start_indices,
            self.start_pos,
        ) = Pos_Data_Processor.convert_dictionary_to_matrix(start)
        (
            end_indices,
            self.end_pos,
        ) = Pos_Data_Processor.convert_dictionary_to_matrix(end)
        (
            self.tmatrix_indices,
            self.tmatrix,
        ) = Pos_Data_Processor.normalize_transition_matrix(tmatrix)

        if (start_indices != end_indices) or (start_indices != self.tmatrix_indices):
            print("StartPos:" + str(start_indices) + " ")
            print("EndPos: " + str(end_indices) + " ")
            print("Transition Matrix Pos: " + str(self.tmatrix_indices) + " ")
            raise Exception("POS orders are not compatible")

        self.all_pos_tag = all_pos_tag
        self.model = self.get_model()

    def __get_discrete_dist(self, dist: dict[str:int]):
        return DiscreteDistribution(dist)

    def get_model(self) -> HiddenMarkovModel:
        dist_list = []
        for pos in self.tmatrix_indices:
            prob_dict = Pos_Data_Processor.get_probability_dict(self.all_pos_tag[pos])
            dist = self.__get_discrete_dist(prob_dict)
            dist_list.append(dist)

        model = HiddenMarkovModel.from_matrix(
            self.tmatrix, dist_list, self.start_pos, self.end_pos
        )
        model.bake()
        return model

    def get_sentence(self):
        return self.model.sample()

# generator = HMM_Text_Generator(start, end, tag)
