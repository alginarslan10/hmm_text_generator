from os import path
from pickle import dump, load
from string import punctuation
from timeit import timeit
from typing import Tuple

from joblib import Parallel, delayed
from numpy import array_split
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
            print(str(e))
            return None

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

    def add_row_and_column_to_matrix(
        self, matrix: list[[str, [int]]], new_pos: str
    ) -> list[[str, [int]]]:

        # Check if matrix empty
        if len(matrix) != 0:
            # Add column
            column_size = len(matrix[0][-1])
            column = [0] * column_size
            matrix.append([new_pos, column])

            for row_tuple in matrix:
                row_tuple[-1].append(0)

        else:
            new_column = [new_pos, [0]]
            matrix = matrix.append(new_column)

        return matrix

    def get_df_pos(
        self,
        df: DataFrame,
        user_restriction_list=None,
    ) -> Tuple[dict[dict], dict, dict, [[str, [int]]]]:
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

        """
        # Inner functions for transition matrix

        def update_transition_matrix(
            transition_matrix: list[[str, [int]]],
            pos_list: [str],
        ) -> list[[str, [int]]]:

            for pos in range(len(pos_list) - 1):
                row = pos_list[pos]
                col = pos_list[pos + 1]

                for i in range(len(transition_matrix)):
                    if transition_matrix[i][0] == row:
                        row_index = i

                    if transition_matrix[i][0] == col:
                        col_index = i

                transition_matrix[row_index][-1][col_index] += 1

            return transition_matrix

        # Check dataframe
        if df is None:
            return None

        pos_tag_dictionary = {}
        pos_start_dictionary = {}
        pos_end_dictionary = {}
        transition_matrix = []

        for index, row in df.iterrows():
            post_type = row["type"]
            post_author = row["from"]
            text = row["text"]

            if type(text) is not str:
                continue

            text = self.remove_punctuation(text)

            if post_type != "message":
                continue

            if self.is_text_empty(text):
                continue

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

            for index, pos in enumerate(pos_tag_list):

                # Add pos tag if not in root dictionary
                if pos not in pos_tag_dictionary:
                    pos_tag_dictionary[pos] = {}

                    # If pos is not in dictionary is not in transition matrix
                    transition_matrix = self.add_row_and_column_to_matrix(
                        transition_matrix
                    )

                # Add word if not in sub dictionary
                if word_list[index] not in pos_tag_dictionary[pos]:
                    pos_tag_dictionary[pos][word_list[index]] = 1

                else:
                    pos_tag_dictionary[pos][word_list[index]] += 1

            # Update transition matrix when all new pos rows and columns added
            transition_matrix = update_transition_matrix(
                transition_matrix, pos_tag_list
            )
        return (
            pos_tag_dictionary,
            pos_start_dictionary,
            pos_end_dictionary,
            transition_matrix,
        )

    def get_df_pos_parallel(
        self, cheat_pickle=False
    ) -> Tuple[dict[str:dict], dict[str:int], dict[str:int]]:

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
            new_matrix: [[str, [int]]], new_index: [str], old_matrix: [[str, [int]]]
        ):
            new_m_index_list = []
            old_m_index_list = []

            for i in new_matrix:
                new_m_index_list.append(i[0])

            for j in old_matrix:
                old_m_index_list.append(j[0])

                # If pos does not exist in new matrix,add
                if j not in new_m_index_list:
                    new_matrix = self.add_row_and_column_to_matrix(new_matrix, j)
                    new_m_index_list.append(j)

            return

        pos_start_dictionary_list = []
        pos_end_dictionary_list = []
        pos_tag_dictionary_list = []

        if not cheat_pickle or path.exists("messages.pickle"):
            # Optimal number of cores server can handle, hand-optimized :(
            jobs_in_parallel = 3
            df_list = array_split(self.df, jobs_in_parallel)

            (
                pos_tag_dictionary_list,
                pos_start_dictionary_list,
                pos_end_dictionary_list,
            ) = zip(
                *Parallel(n_jobs=jobs_in_parallel)(
                    [delayed(self.get_df_pos)(i) for i in df_list]
                )
            )

            # Unite the dictionaries
            new_start_dictionary = {}
            new_end_dictionary = {}
            new_tag_dictionary = {}

            for i in range(len(pos_tag_dictionary_list)):
                self.merge_start_end_dict(
                    new_start_dictionary, pos_start_dictionary_list[i]
                )
                self.merge_start_end_dict(
                    new_end_dictionary, pos_end_dictionary_list[i]
                )

                self.merge_nested_dict(new_tag_dictionary, pos_tag_dictionary_list[i])

            pickle_list = (
                new_tag_dictionary,
                new_start_dictionary,
                new_end_dictionary,
            )
            with open("messages.pickle", "wb") as f:
                dump(pickle_list, f)

        else:
            with open(cheat_pickle, "rb") as f:
                pickle_list = load(f)
            new_tag_dictionary = pickle_list[0]
            new_start_dictionary = pickle_list[1]
            new_end_dictionary = pickle_list[2]

            return new_tag_dictionary, new_start_dictionary, new_end_dictionary

    def get_probability_dict(self, total_encounter: dict) -> dict[str:float]:
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
        total_word_encounter = 0
        prob_dictionary = {}

        for encounter in total_encounter.values():
            total_word_encounter += encounter

        for word in total_encounter:
            prob_dictionary[word] = total_encounter[word] / total_word_encounter

        return prob_dictionary


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
