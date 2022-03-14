from os import path
from pickle import dump, load
from string import punctuation
from timeit import timeit
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from pandas import DataFrame, json_normalize, read_json
from requests import post


class Zemberek_Server_Pos_Tagger:
    def __init__(self, host_name: str, host_port: int, file_path: str):
        self.host_name = host_name
        self.host_port = host_port
        self.file_path = file_path
        self.df = self.__read_to_df()

    def __read_to_df(self) -> DataFrame:
        try:
            df = read_json(self.file_path)
            df = json_normalize(df["messages"])
            return df
        except Exception as e:
            print("Could not read the data beacause of following reason:")
            print(str(e))
            exit()

    def is_text_empty(self, text):
        if text.replace(" ", "") == "":
            return True
        else:
            return False

    def remove_punctuation(self, text: str):
        new_text = "".join([i for i in text if i not in punctuation])
        return new_text

    def get_text_pos(self, text: str) -> list[str]:
        """__get_text_pos.
        Gets a text and tags the words with part of speech tags.

        Parameters
        ----------
        text : str
            Text desired to pos tagged.

        Returns
        -------
        pos_tag_list :list[tuple(str)]
            A list of tuples that contains word and pos tag of it.

        """
        if self.is_text_empty(text):
            raise Exception("Empty text")

        endpoint = self.host_name + ":" + str(self.host_port) + "/" + "find_pos"
        data = {"sentence": text}
        response = post(url=endpoint, data=data)

        if response.ok:
            return [pos_tag["pos"] for pos_tag in response.json()]
        else:
            raise Exception("Response not OK.Status Code:" + response.status_code)

    def get_df_pos(
        self,
        df: DataFrame,
        user_restriction_list=None,
    ) -> Tuple[dict[dict], dict, dict, list[str, np.array]]:
        """Gets the part of speech tags in the message column of the given dataframe.
         Also keeps track of the starting and ending part of speech tags

         Parameters
         ----------
         df : DataFrame
             DataFrame that contains data and meta-data for telegram messages

        user_restriction_list : list
             Optional restriction over messages. Only processes the messages
             that come from the users in the list.

         Returns
         -------
         pos_tag_dictionary : dict[dict]
             Part of speech tags' word encounter. First dictonary contains part
             of speech tags as key and words encountered dictionary as value.
             Nested dictionary contains words encountered as key, and how many
             times the word has been encountered as value.

         pos_start_dictionary :dict
             Keeps track of sentence starting pos tags. Pos tag name as key and
             encounter number as value.

         pos_end_dictionart :dict
             Keeps track of sentence ending pos tags. Pos tag name as key and
             encounter number as value

        transition_matrix : [str,[int]]
            Keeps track transitions between pos tags. String as the pos tag of
            the row and [int] as the columns.
        """

        # Check dataframe
        if df is None:
            return None

        pos_tag_dictionary = {}
        pos_start_dictionary = {}
        pos_end_dictionary = {}
        indices = []
        matrix = np.array([[]])

        for index, row in df.iterrows():
            post_type = row["type"]
            post_author = row["from"]
            text = row["text"]

            if post_type != "message":
                continue

            if type(text) is not str:
                continue

            text = self.remove_punctuation(text)

            if self.is_text_empty(text):
                continue

            text = text.lower()

            # Check if there is user restriction and if there is skip
            # non-restricted users.
            if user_restriction_list is not None:
                if post_author not in user_restriction_list:
                    continue

            word_list = text.split()
            pos_tag_list = self.get_text_pos(text)

            # Add the start part of the sentence if not existent
            if pos_tag_list[0] not in pos_start_dictionary:
                pos_start_dictionary[pos_tag_list[0]] = 1
            else:
                pos_start_dictionary[pos_tag_list[0]] += 1

            # Add the end part of the sentence if not existent
            if pos_tag_list[-1] not in pos_end_dictionary:
                pos_end_dictionary[pos_tag_list[-1]] = 1
            else:
                pos_end_dictionary[pos_tag_list[-1]] += 1

            for i, pos in enumerate(pos_tag_list):

                # Add pos tag if not in root dictionary
                if pos not in pos_tag_dictionary:
                    pos_tag_dictionary[pos] = {}

                    # If pos is not in dictionary is not in transition matrix
                    indices, matrix = Pos_Data_Processor.add_row_and_column_to_matrix(
                        [indices, matrix], pos
                    )

                # Add word if not in sub dictionary
                if word_list[i] not in pos_tag_dictionary[pos]:
                    pos_tag_dictionary[pos][word_list[i]] = 1

                else:
                    pos_tag_dictionary[pos][word_list[i]] += 1

            # Update transition matrix when all new pos rows and columns added
            indices, matrix = Pos_Data_Processor.update_transition_matrix(
                [indices, matrix], pos_tag_list
            )

        return (
            pos_tag_dictionary,
            pos_start_dictionary,
            pos_end_dictionary,
            [indices, matrix],
        )

    def get_df_pos_parallel(
        self, cheat_pickle=False
    ) -> Tuple[dict[str:dict], dict[str:int], dict[str:int], list[str, np.array]]:

        # Get inner functions to merge later on
        def merge_start_end_dict(new_dict: dict, old_dict: dict):

            for start in old_dict:

                if start in new_dict:
                    new_dict[start] += old_dict[start]
                else:
                    new_dict[start] = old_dict[start]

        def merge_nested_dict(new_dict: dict[dict], old_dict: dict[dict]):

            for pos in old_dict:

                if pos in new_dict:

                    for word in old_dict[pos]:
                        if word in new_dict[pos]:
                            new_dict[pos][word] += old_dict[pos][word]
                        else:
                            new_dict[pos][word] = old_dict[pos][word]

                else:
                    new_dict[pos] = old_dict[pos]

        def merge_transition_matrix(
            new_matrix_list: list[list[str], np.array],
            old_matrix_list: list[list[str], np.array],
        ) -> list[str, np.array]:

            new_matrix_indices, new_matrix = new_matrix_list
            old_matrix_indices, old_matrix = old_matrix_list

            for old_pos in old_matrix_indices:
                # If pos does not exist in new matrix,add
                if old_pos not in new_matrix_indices:
                    (
                        new_matrix_indices,
                        new_matrix,
                    ) = Pos_Data_Processor.add_row_and_column_to_matrix(
                        [new_matrix_indices, new_matrix], old_pos
                    )

            # TODO: Iterating same list twice not good, change if you can think of a
            # better solution, also O(n) of that must be around 2n^2
            # if index function is n, fix that bruv
            for old_row_index in range(len(old_matrix)):
                new_row_index = new_matrix_indices.index(
                    old_matrix_indices[old_row_index]
                )
                for old_col_index in range(len(old_matrix[old_row_index])):
                    new_col_index = new_matrix_indices.index(
                        old_matrix_indices[old_col_index]
                    )

                    new_matrix[new_row_index][new_col_index] += old_matrix[
                        old_row_index
                    ][old_col_index]

            return new_matrix_indices, new_matrix

        pos_start_dictionary_list = []
        pos_end_dictionary_list = []
        pos_tag_dictionary_list = []
        tmatrix_list_list = []

        if cheat_pickle is False or not (path.exists("messages.pickle")):
            # Optimal number of cores server can handle, hand-optimized :(
            jobs_in_parallel = 3
            df_list = np.array_split(self.df, jobs_in_parallel)

            (
                pos_tag_dictionary_list,
                pos_start_dictionary_list,
                pos_end_dictionary_list,
                tmatrix_list_list,
            ) = zip(
                *Parallel(n_jobs=jobs_in_parallel)(
                    [delayed(self.get_df_pos)(i) for i in df_list]
                )
            )

            # Unite the dictionaries
            new_start_dictionary = {}
            new_end_dictionary = {}
            new_tag_dictionary = {}
            new_indices = []
            new_tmatrix = np.array([[]])

            for i in range(len(pos_tag_dictionary_list)):
                merge_start_end_dict(new_start_dictionary, pos_start_dictionary_list[i])
                merge_start_end_dict(new_end_dictionary, pos_end_dictionary_list[i])

                merge_nested_dict(new_tag_dictionary, pos_tag_dictionary_list[i])
                new_indices, new_tmatrix = merge_transition_matrix(
                    [new_indices, new_tmatrix], tmatrix_list_list[i]
                )

            new_start_dictionary = Pos_Data_Processor.add_non_starting_ending_pos(
                new_start_dictionary, new_indices
            )
            new_end_dictionary = Pos_Data_Processor.add_non_starting_ending_pos(
                new_end_dictionary, new_indices
            )

            pickle_list = (
                new_tag_dictionary,
                new_start_dictionary,
                new_end_dictionary,
                [new_indices, new_tmatrix],
            )
            with open("messages.pickle", "wb") as f:
                dump(pickle_list, f)

        else:
            with open(cheat_pickle, "rb") as f:
                pickle_list = load(f)
            new_tag_dictionary = pickle_list[0]
            new_start_dictionary = pickle_list[1]
            new_end_dictionary = pickle_list[2]
            new_indices, new_tmatrix = pickle_list[3]

        return (
            new_tag_dictionary,
            new_start_dictionary,
            new_end_dictionary,
            [new_indices, new_tmatrix],
        )


class Pos_Data_Processor:
    def __init__(self):
        return

    @staticmethod
    def add_non_starting_ending_pos(
        missing_dict: dict[str:int], indices_list: list[str]
    ):
        for i in indices_list:
            if i not in missing_dict.keys():
                missing_dict[i] = 0

        return missing_dict

    @staticmethod
    def sort_by_indices(
        unsorted_list: list[list[str], np.array]
    ) -> list[list[str], np.array]:
        indices, array = unsorted_list

        temp_indices = 0
        temp_array = 0

        for i in range(1, len(indices)):
            temp_indices = indices[i]
            temp_array = array[i]
            j = i - 1

            while j >= 0 and temp_indices < indices[j]:
                indices[j + 1] = indices[j]
                array[j + 1] = array[j]
                j -= 1

            indices[j + 1] = temp_indices
            array[j + 1] = temp_array

        return [indices, array]

    @staticmethod
    def add_row_and_column_to_matrix(
        matrix_list: list[list[str], np.array], new_pos: str
    ) -> list[list[str], np.array]:
        indices, matrix = matrix_list

        if len(matrix[0]) != 0:
            # Add row
            row_size = len(matrix[0])
            row = [[0] * row_size]
            matrix = np.append(matrix, row, axis=0)

            # Add columns
            col = np.zeros((row_size + 1, 1))
            matrix = np.append(matrix, col, axis=1)

            indices.append(new_pos)

        # If matrix empty
        else:
            matrix = np.array([[0]])
            indices.append(new_pos)

        return [indices, matrix]

    @staticmethod
    def update_transition_matrix(
        transition_matrix_list: list[list[str], np.array],
        pos_list: [str],
    ) -> list[list[str], np.array]:
        indices = transition_matrix_list[0]
        transition_matrix = transition_matrix_list[1]

        for pos in range(len(pos_list) - 1):
            row = pos_list[pos]
            col = pos_list[pos + 1]

            row_index = indices.index(row)
            col_index = indices.index(col)

            transition_matrix[row_index][col_index] += 1

        return [indices, transition_matrix]

    @staticmethod
    def get_probability_dict(words_dict: dict) -> dict[str:float]:
        """Receiving the {word:encounter} of the words as dictionary, converts the
        word encounters to ratio of words to total encounter count (word/total
        encounter).

        Parameters
        ----------
        total_encounter : dict
            Word encounters given as dictionary

        Returns
        ----------
        prob_dictionary: dict
            Encounter probabilty of the words
        """
        total_word_encounter_sum = 0
        prob_dictionary = {}

        for encounter in words_dict.values():
            total_word_encounter_sum += encounter

        for word in words_dict:
            prob_dictionary[word] = words_dict[word] / total_word_encounter_sum

        return prob_dictionary

    @staticmethod
    def normalize_transition_matrix(
        matrix_list: list[list[str], np.array]
    ) -> list[list[str], np.array]:
        indices = matrix_list[0]
        matrix = matrix_list[1]
        row_summations = np.sum(matrix, axis=1)
        normalized_matrix = matrix / row_summations[:, None]

        return Pos_Data_Processor.sort_by_indices([indices, normalized_matrix])

    @staticmethod
    def convert_dictionary_to_matrix(
        dictionary: dict[str:int],
    ) -> list[list[str], np.array]:
        indices = []
        probability_vector = np.array([])
        sum_encounter = sum(dictionary.values())

        for pos in dictionary:
            indices.append(pos)
            probability_vector = np.append(
                probability_vector, [dictionary[pos] / sum_encounter], axis=0
            )

        return Pos_Data_Processor.sort_by_indices([indices, probability_vector])


def test_parallel_time(req_obj: Zemberek_Server_Pos_Tagger):
    req_obj.get_df_pos_parallel()


def test_non_parallel_time(req_obj: Zemberek_Server_Pos_Tagger):
    req_obj.get_df_pos(req_obj.df)


def parallel_speed_test(req_obj: Zemberek_Server_Pos_Tagger):

    parallel_res = timeit("test_parallel_time(req_obj)", globals=globals(), number=1)
    non_parallel_res = timeit(
        "test_non_parallel_time(req_obj)", globals=globals(), number=1
    )
    print(parallel_res / non_parallel_res)
