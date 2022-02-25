from json import loads
from multiprocessing import cpu_count
from string import punctuation

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

    def __remove_punctuation(self, text):
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

    def __is_text_empty(self, text):
        if text.replace(" ", "") == "":
            return True
        else:
            return False

    def get_df_pos(
        self, df: DataFrame, user_restriction_list=None
    ) -> tuple(dict[dict], dict):
        """Gets the part of speech tags in the messages column of the given dataframe.
        Also keeps track of the starting part of speech tags

        Parameters
        ----------
        df : DataFrame
            DataFrame that contains data and meta-data for telegram messages
        user_restriction_list :
            Optional restriction over messages. Only processes the messages
            that comes from the users in the list.

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

            word_list = text.split(" ")
            pos_tag_list = self.get_text_pos(text)

            # Add the start part of the sentence if not existent
            if pos_tag_list[0] not in pos_start_dictionary:
                pos_start_dictionary[pos_tag_list[0]] = 1
            else:
                pos_start_dictionary += 1

            for index, pos in enumerate(pos_tag_list):

                # Add pos tag if not in root dictionary
                if pos not in pos_tag_dictionary:
                    pos_tag_dictionary[pos] = {}

                # Add word if not in sub dictionary
                if word_list[index] not in pos_tag_dictionary[pos]:
                    pos_tag_dictionary[pos][word_list[index]] == 1

                else:
                    pos_tag_dictionary[pos][word_list[index]] += 1

        return pos_tag_dictionary, pos_start_dictionary


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

    for encounter in total_encounter.values():
        total_word_encounter += total_encounter

    prob_dictionary = {}

    for word in total_encounter:
        prob_dictionary[word] = total_encounter[word] / total_word_encounter

    return prob_dictionary


file_path = "/home/algin/İndirilenler/Telegram Desktop/Data/result.json"
df = read_json(file_path)
req_obj = Zemberek_Server_Pos_Tagger("http://localhost", 4567, file_path)
print(req_obj.get_text_pos("Senle ben bir değiliz"))
