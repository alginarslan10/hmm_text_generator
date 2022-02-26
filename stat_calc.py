from json import loads
from multiprocessing import cpu_count
from string import punctuation
from typing import Tuple

from joblib import Parallel, delayed
from pandas import DataFrame, json_normalize, read_json
from requests import post


class Zemberek_Server_Pos_Tagger:
    def __init__(self, host_name: str, host_port: int, file_path: str):
        self.host_name = host_name
        self.host_port = host_port
        self.core_count = cpu_count()
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
        if (text == "") or (text == " "):
            raise Exception("Empty text")

        endpoint = self.host_name + ":" + str(self.host_port) + "/" + "find_pos"
        data = {"sentence": text}
        response = post(url=endpoint, data=data)

        if response.ok:
            return [pos_tag["pos"] for pos_tag in response.json()]
        else:
            raise Exception("Response not OK.Status Code:" + response.status_code)

    def is_text_empty(self, text):
        if text.replace(" ", "") == "":
            return True
        else:
            return False

    def get_df_pos(
        self, df: DataFrame, user_restriction_list=None
    ) -> Tuple[dict[dict], dict]:
        """Gets the part of speech tags in the message column of the given dataframe.
        Also keeps track of the starting part of speech tags

        Parameters
        ----------
        df : DataFrame
            DataFrame that contains data and meta-data for telegram messages
        user_restriction_list :
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



        """

        # Check dataframe
        if df is None:
            return None

        pos_tag_dictionary = {}
        pos_start_dictionary = {}

        for index, row in df.iterrows():
            post_type = row["type"]
            post_author = row["from"]
            text = row["text"]

            if type(text) is not str:
                continue

            text = self.__remove_punctuation(text)

            if post_type != "message":
                continue

            if self.__is_text_empty(text):
                continue

            # Check if there is user restriction and if there is skip
            # non-restricted users.
            if user_restriction_list is not None:
                if post_author not in user_restriction_list:
                    continue

            word_list = text.split()
            pos_tag_list = self.__get_text_pos(text)

            # Add the start part of the sentence if not existent
            if pos_tag_list[0] not in pos_start_dictionary:
                pos_start_dictionary[pos_tag_list[0]] = 1
            else:
                pos_start_dictionary[pos_tag_list[0]] += 1

            for index, pos in enumerate(pos_tag_list):

                # Add pos tag if not in root dictionary
                if pos not in pos_tag_dictionary:
                    pos_tag_dictionary[pos] = {}

                # Add word if not in sub dictionary
                if word_list[index] not in pos_tag_dictionary[pos]:
                    pos_tag_dictionary[pos][word_list[index]] = 1

                else:
                    pos_tag_dictionary[pos][word_list[index]] += 1

        return pos_tag_dictionary, pos_start_dictionary

    def get_df_pos_parallel(self) -> Tuple[dict[dict], dict]:
        pos_start_dictionary_list = []
        pos_tag_dictionary_list = []
        df_list = []
        df_split_size = len(self.df.index) // self.core_count

        for index in range(df_split_size):

            # If last part get until the end of dataframe
            if (index + 1) == df_split_size:
                new_df = self.df[index:]
                df_list.append(new_df)
                break

            new_df = df[index : index + 1]
            df_list.append(new_df)

        pos_tag_dictionary_list, pos_start_dictionary_list = zip(
            *Parallel(n_jobs=self.core_count, prefer="processes")(
                [delayed(self.get_df_pos)(i) for i in df_list]
            )
        )
        return pos_tag_dictionary_list, pos_start_dictionary_list


# TODO: DEBUG, sumtin wrong dunno what
def get_probability_dict(total_encounter: dict):
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
        total_word_encounter += total_encounter

    for word in total_encounter:
        prob_dictionary[word] = total_encounter[word] / total_word_encounter

    return prob_dictionary


file_path = "/home/algin/İndirilenler/Telegram Desktop/Data/result.json"
df = read_json(file_path)
req_obj = Zemberek_Server_Pos_Tagger("http://localhost", 4567, file_path)
pos_tag_dict, pos_start_dict = req_obj.get_df_pos_parallel()
# print("Pos_Tag :")
# print(pos_tag_dict)
# print("Pos_Start :")
# print(pos_start_dict)
