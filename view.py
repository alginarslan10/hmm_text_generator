import sys
from configparser import ConfigParser

from hmm import HMM_Text_Generator
from stat_calc import Zemberek_Server_Pos_Tagger


def get_file_path() -> str:
    parser = ConfigParser()
    parser.read("test.ini")
    file_path = parser["telegram_data"]["RESULT_JSON_PATH"]
    return file_path


def get_hmm_model(file_path: str):
    req_obj = Zemberek_Server_Pos_Tagger("http://localhost", 4567, file_path)

    tag, start, end, transition_matrix_list = req_obj.get_df_pos_parallel(
        cheat_pickle="messages.pickle"
    )
    hmm_text = HMM_Text_Generator(start, end, tag, transition_matrix_list)
    return hmm_text


# Accept only one arg, which is min word count for text
def main():
    file_path = get_file_path()
    hmm = get_hmm_model(file_path)
    args = sys.argv[1:]
    argc = len(args)

    if argc == 0:
        print(hmm.get_sentence())
    else:
        try:
            min_word = int(args[0])
            text = hmm.get_at_least_n_word_text(min_word)
            print(text)
        except Exception as e:
            print(e)
            exit()


if __name__ == "__main__":
    main()
